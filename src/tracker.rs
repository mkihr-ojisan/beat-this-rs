use std::path::Path;

use burn::module::Module;
use burn::store::{ModuleSnapshot, PytorchStore};
use burn::tensor::{Device, Tensor, TensorData, TensorReadError, Transaction};

use crate::chunking::{self, Chunk};
use crate::error::{Error, Result};
use crate::mel::{LogMelSpectrogram, MEL_BANDS, Spectrogram};
use crate::model::{BeatThis, ModelOutput};
use crate::postprocess::{Beats, logits_to_beats};
use crate::resample::Decimator;

/// チェックポイントのキーからモジュールのパスへの置換規則。
const KEY_REMAPPINGS: &[(&str, &str)] = &[
    (r"^model\.", ""),
    (r"\.bn1d\.", ".batch_norm_1d."),
    (r"\.bn2d\.", ".batch_norm_2d."),
    (r"\.conv2d\.", ".convolution."),
    (r"\.attnF\.", ".attention_frequency."),
    (r"\.ffF\.", ".feed_forward_frequency."),
    (r"\.attnT\.", ".attention_time."),
    (r"\.ffT\.", ".feed_forward_time."),
    (r"layers\.(\d+)\.0\.", "layers.$1.attention."),
    (r"layers\.(\d+)\.1\.", "layers.$1.feed_forward."),
    (r"\.net\.0\.", ".norm."),
    (r"\.net\.1\.", ".linear_in."),
    (r"\.net\.4\.", ".linear_out."),
    (r"\.to_out\.0\.", ".to_out."),
    (r"\.rotary_embed\.freqs$", ".rotary_embedding.frequencies"),
    (r"\.beat_downbeat_lin\.", ".beat_downbeat_linear."),
];

/// フレームごとのロジット。フレーム間隔は 20 ms。
#[derive(Clone, Debug, PartialEq)]
pub struct FrameLogits {
    /// ビートのロジット。
    pub beat: Vec<f32>,
    /// ダウンビートのロジット。
    pub downbeat: Vec<f32>,
}

/// モノラル音声からビートとダウンビートを検出する。
pub struct BeatTracker {
    /// 重みを読み込み済みのモデル。
    model: BeatThis,
    /// 44.1 kHz から 22050 Hz への間引きフィルター。
    decimator: Decimator,
    /// スペクトログラムの計算器。
    spectrogram: LogMelSpectrogram,
    /// モデルを配置したデバイス。
    device: Device,
}

impl BeatTracker {
    /// `checkpoint` の重みを読み込む。
    ///
    /// * `checkpoint` - `final0.ckpt` 形式の PyTorch Lightning チェックポイントのパス。
    /// * `device` - 推論に使用するデバイス。
    ///
    /// チェックポイントを読み込めないか、必要な重みが存在しないときはエラーを返す。
    pub fn load(checkpoint: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let mut store = KEY_REMAPPINGS.iter().fold(
            PytorchStore::from_file(checkpoint.as_ref()).with_top_level_key("state_dict"),
            |store, (from, to)| store.with_key_remapping(*from, *to),
        );
        let mut model = BeatThis::new(device);
        model.load_from(&mut store)?;

        Ok(Self {
            model: model.valid(),
            decimator: Decimator::new(),
            spectrogram: LogMelSpectrogram::new(),
            device: device.clone(),
        })
    }

    /// `mono_44100` のビートとダウンビートを検出する。
    ///
    /// * `mono_44100` - 44.1 kHz のモノラル音声。
    ///
    /// 入力が約 23 ms 未満のときは [`Error::AudioTooShort`]、NaN または無限大を含むときは
    /// [`Error::NonFiniteSample`] を返す。
    pub fn track(&self, mono_44100: &[f32]) -> Result<Beats> {
        let logits = self.frame_logits(mono_44100)?;
        Ok(logits_to_beats(&logits.beat, &logits.downbeat))
    }

    /// `mono_44100` のフレームごとのロジットを計算する。
    ///
    /// * `mono_44100` - 44.1 kHz のモノラル音声。
    ///
    /// 入力が約 23 ms 未満のときは [`Error::AudioTooShort`]、NaN または無限大を含むときは
    /// [`Error::NonFiniteSample`] を返す。
    pub fn frame_logits(&self, mono_44100: &[f32]) -> Result<FrameLogits> {
        ensure_finite(mono_44100)?;

        let spectrogram = self
            .spectrogram
            .compute(&self.decimator.process(mono_44100))?;

        let chunks = chunking::split(spectrogram.frames);
        let mut beat_predictions = Vec::with_capacity(chunks.len());
        let mut downbeat_predictions = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let output = self.model.forward(self.chunk_tensor(&spectrogram, chunk));
            let logits = read_logits(output)?;
            beat_predictions.push(logits.beat);
            downbeat_predictions.push(logits.downbeat);
        }

        Ok(FrameLogits {
            beat: chunking::aggregate(&chunks, &beat_predictions, spectrogram.frames),
            downbeat: chunking::aggregate(&chunks, &downbeat_predictions, spectrogram.frames),
        })
    }

    /// `spectrogram` から `chunk` の範囲を抽出してパディングした、形状 `[1, frames, MEL_BANDS]` のテンソルを返す。
    ///
    /// * `spectrogram` - 曲全体のスペクトログラム。
    /// * `chunk` - 抽出する範囲。
    fn chunk_tensor(&self, spectrogram: &Spectrogram, chunk: &Chunk) -> Tensor<3> {
        let mut values = vec![0.0f32; chunk.len() * MEL_BANDS];
        let begin = chunk.source.start * MEL_BANDS;
        let end = chunk.source.end * MEL_BANDS;
        let output_begin = chunk.left_padding * MEL_BANDS;
        values[output_begin..output_begin + (end - begin)]
            .copy_from_slice(&spectrogram.values[begin..end]);
        Tensor::from_data(
            TensorData::new(values, [1, chunk.len(), MEL_BANDS]),
            &self.device,
        )
    }
}

/// `samples` がすべて有限であることを確認する。
///
/// * `samples` - 確認するサンプル列。
///
/// NaN または無限大を含むときは、その最初の位置を格納した [`Error::NonFiniteSample`] を返す。
fn ensure_finite(samples: &[f32]) -> Result<()> {
    match samples.iter().position(|sample| !sample.is_finite()) {
        Some(index) => Err(Error::NonFiniteSample { index }),
        None => Ok(()),
    }
}

/// `output` のビートとダウンビートのロジットを取得する。
///
/// * `output` - 各テンソルの形状が `[1, frames]` であるモデルの出力。
fn read_logits(output: ModelOutput) -> Result<FrameLogits> {
    // デバイスからのテンソルの取得を 1 回に集約する。
    let [beat, downbeat]: [TensorData; 2] = Transaction::default()
        .register(output.beat)
        .register(output.downbeat)
        .try_execute()
        .map_err(TensorReadError::from)?
        .try_into()
        .expect("a transaction returns one item per registered tensor");
    Ok(FrameLogits {
        beat: into_vector(beat)?,
        downbeat: into_vector(downbeat)?,
    })
}

/// `data` の値を 1 次元の配列として取得する。
///
/// * `data` - テンソルから取得した値。
fn into_vector(data: TensorData) -> Result<Vec<f32>> {
    Ok(data
        .try_into_vec_as::<f32>()
        .map_err(TensorReadError::from)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_samples_are_accepted() {
        assert!(ensure_finite(&[]).is_ok());
        assert!(ensure_finite(&[0.0, -1.0, f32::MAX]).is_ok());
    }

    #[test]
    fn first_non_finite_sample_is_reported() {
        for (samples, expected) in [
            ([f32::INFINITY, 0.0, 0.0], 0),
            ([0.0, f32::NAN, f32::INFINITY], 1),
            ([0.0, 0.0, f32::NEG_INFINITY], 2),
        ] {
            assert!(matches!(
                ensure_finite(&samples),
                Err(Error::NonFiniteSample { index }) if index == expected
            ));
        }
    }
}
