mod roformer;

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::norm::{BatchNorm, BatchNormConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::gelu;
use burn::tensor::{Device, Tensor};

use crate::mel::MEL_BANDS;
use roformer::{Attention, FeedForward, Transformer};

/// 時間方向の Transformer の次元数。
const TRANSFORMER_DIMENSION: usize = 512;
/// 時間方向の Transformer の層数。
const TRANSFORMER_LAYERS: usize = 6;
/// フィードフォワードの次元の倍率。
const FEED_FORWARD_MULTIPLIER: usize = 4;
/// 自己注意のヘッドあたりの次元数。
const HEAD_DIMENSION: usize = 32;
/// 先頭の畳み込みの出力チャンネル数。
const STEM_DIMENSION: usize = 32;
/// [`FrontendBlock`] の数。
const FRONTEND_BLOCKS: usize = 3;

/// 周波数方向と時間方向の部分 Transformer。
#[derive(Module, Debug)]
struct PartialFrequencyTimeTransformer {
    /// 周波数方向の自己注意。
    attention_frequency: Attention,
    /// 周波数方向のフィードフォワード。
    feed_forward_frequency: FeedForward,
    /// 時間方向の自己注意。
    attention_time: Attention,
    /// 時間方向のフィードフォワード。
    feed_forward_time: FeedForward,
}

impl PartialFrequencyTimeTransformer {
    /// 部分 Transformer を作成する。
    ///
    /// * `dimension` - チャンネル数。
    /// * `device` - パラメーターを配置するデバイス。
    fn new(dimension: usize, device: &Device) -> Self {
        Self {
            attention_frequency: Attention::new(dimension, HEAD_DIMENSION, device),
            feed_forward_frequency: FeedForward::new(dimension, FEED_FORWARD_MULTIPLIER, device),
            attention_time: Attention::new(dimension, HEAD_DIMENSION, device),
            feed_forward_time: FeedForward::new(dimension, FEED_FORWARD_MULTIPLIER, device),
        }
    }

    /// 形状を維持して周波数方向および時間方向の自己注意とフィードフォワードを適用する。
    ///
    /// * `input` - 形状 `[batch, channels, frequencies, times]` のテンソル。
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let [batch, channels, frequencies, times] = input.dims();

        let hidden = input
            .permute([0, 3, 2, 1])
            .reshape([batch * times, frequencies, channels]);
        let hidden = self.attention_frequency.forward(hidden.clone()) + hidden;
        let hidden = self.feed_forward_frequency.forward(hidden.clone()) + hidden;

        let hidden = hidden
            .reshape([batch, times, frequencies, channels])
            .permute([0, 2, 1, 3])
            .reshape([batch * frequencies, times, channels]);
        let hidden = self.attention_time.forward(hidden.clone()) + hidden;
        let hidden = self.feed_forward_time.forward(hidden.clone()) + hidden;

        hidden
            .reshape([batch, frequencies, times, channels])
            .permute([0, 3, 1, 2])
    }
}

/// 部分 Transformer の後に周波数方向へ畳み込むブロック。
#[derive(Module, Debug)]
struct FrontendBlock {
    /// 部分 Transformer。
    partial: PartialFrequencyTimeTransformer,
    /// 周波数の数を 2 分の 1 にする畳み込み。
    convolution: Conv2d,
    /// 畳み込みの出力の BatchNorm。
    norm: BatchNorm,
}

impl FrontendBlock {
    /// ブロックを作成する。
    ///
    /// * `input_dimension` - 入力のチャンネル数。
    /// * `output_dimension` - 出力のチャンネル数。
    /// * `device` - パラメーターを配置するデバイス。
    fn new(input_dimension: usize, output_dimension: usize, device: &Device) -> Self {
        Self {
            partial: PartialFrequencyTimeTransformer::new(input_dimension, device),
            convolution: Conv2dConfig::new([input_dimension, output_dimension], [2, 3])
                .with_stride([2, 1])
                .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
                .with_bias(false)
                .init(device),
            norm: BatchNormConfig::new(output_dimension).init(device),
        }
    }

    /// 周波数の数を 2 分の 1 にしてチャンネル数を変換する。
    ///
    /// * `input` - 形状 `[batch, channels, frequencies, times]` のテンソル。
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let hidden = self.partial.forward(input);
        gelu(self.norm.forward(self.convolution.forward(hidden)))
    }
}

/// スペクトログラムを入力とする先頭の畳み込み。
#[derive(Module, Debug)]
struct Stem {
    /// メル帯域ごとの BatchNorm。
    batch_norm_1d: BatchNorm,
    /// 周波数の数を 4 分の 1 にする畳み込み。
    convolution: Conv2d,
    /// 畳み込みの出力の BatchNorm。
    batch_norm_2d: BatchNorm,
}

impl Stem {
    /// 先頭の畳み込みを作成する。
    ///
    /// * `device` - パラメーターを配置するデバイス。
    fn new(device: &Device) -> Self {
        Self {
            batch_norm_1d: BatchNormConfig::new(MEL_BANDS).init(device),
            convolution: Conv2dConfig::new([1, STEM_DIMENSION], [4, 3])
                .with_stride([4, 1])
                .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
                .with_bias(false)
                .init(device),
            batch_norm_2d: BatchNormConfig::new(STEM_DIMENSION).init(device),
        }
    }

    /// `input` を `[batch, STEM_DIMENSION, frequencies, times]` へ変換する。
    ///
    /// * `input` - 形状 `[batch, times, MEL_BANDS]` のスペクトログラム。
    fn forward(&self, input: Tensor<3>) -> Tensor<4> {
        let normalized = self.batch_norm_1d.forward(input.swap_dims(1, 2));
        let hidden = self.convolution.forward(normalized.unsqueeze_dim::<4>(1));
        gelu(self.batch_norm_2d.forward(hidden))
    }
}

/// スペクトログラムを Transformer の次元へ射影するフロントエンド。
#[derive(Module, Debug)]
struct Frontend {
    /// 先頭の畳み込み。
    stem: Stem,
    /// 順に適用するブロック。
    blocks: Vec<FrontendBlock>,
    /// Transformer の次元への射影。
    linear: Linear,
}

impl Frontend {
    /// フロントエンドを作成する。
    ///
    /// * `device` - パラメーターを配置するデバイス。
    fn new(device: &Device) -> Self {
        let mut dimension = STEM_DIMENSION;
        let mut frequencies = MEL_BANDS / 4;
        let mut blocks = Vec::with_capacity(FRONTEND_BLOCKS);
        for _ in 0..FRONTEND_BLOCKS {
            blocks.push(FrontendBlock::new(dimension, dimension * 2, device));
            dimension *= 2;
            frequencies /= 2;
        }
        Self {
            stem: Stem::new(device),
            blocks,
            linear: LinearConfig::new(dimension * frequencies, TRANSFORMER_DIMENSION).init(device),
        }
    }

    /// `input` を `[batch, times, TRANSFORMER_DIMENSION]` へ変換する。
    ///
    /// * `input` - 形状 `[batch, times, MEL_BANDS]` のスペクトログラム。
    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let mut hidden = self.stem.forward(input);
        for block in &self.blocks {
            hidden = block.forward(hidden);
        }
        let [batch, channels, frequencies, times] = hidden.dims();
        let flattened =
            hidden
                .permute([0, 3, 1, 2])
                .reshape([batch, times, channels * frequencies]);
        self.linear.forward(flattened)
    }
}

/// ビートとダウンビートのロジットを出力する層。ビートのロジットにはダウンビートのロジットを加算する。
#[derive(Module, Debug)]
struct SumHead {
    /// ビートとダウンビートの 2 次元への射影。
    beat_downbeat_linear: Linear,
}

impl SumHead {
    /// 出力層を作成する。
    ///
    /// * `device` - パラメーターを配置するデバイス。
    fn new(device: &Device) -> Self {
        Self {
            beat_downbeat_linear: LinearConfig::new(TRANSFORMER_DIMENSION, 2).init(device),
        }
    }

    /// フレームごとのロジットを計算する。
    ///
    /// * `input` - 形状 `[batch, times, TRANSFORMER_DIMENSION]` のテンソル。
    fn forward(&self, input: Tensor<3>) -> ModelOutput {
        let logits = self.beat_downbeat_linear.forward(input);
        let [batch, times, _] = logits.dims();
        let beat = logits.clone().narrow(2, 0, 1).reshape([batch, times]);
        let downbeat = logits.narrow(2, 1, 1).reshape([batch, times]);
        ModelOutput {
            beat: beat + downbeat.clone(),
            downbeat,
        }
    }
}

/// モデルの出力。
pub(crate) struct ModelOutput {
    /// 形状 `[batch, times]` のビートのロジット。
    pub(crate) beat: Tensor<2>,
    /// 形状 `[batch, times]` のダウンビートのロジット。
    pub(crate) downbeat: Tensor<2>,
}

/// Beat This! のモデル本体。
#[derive(Module, Debug)]
pub(crate) struct BeatThis {
    /// スペクトログラムを Transformer の次元へ射影するフロントエンド。
    frontend: Frontend,
    /// 時間方向の Transformer。
    transformer_blocks: Transformer,
    /// 出力層。
    task_heads: SumHead,
}

impl BeatThis {
    /// 未学習のモデルを作成する。
    ///
    /// * `device` - パラメーターを配置するデバイス。
    pub(crate) fn new(device: &Device) -> Self {
        Self {
            frontend: Frontend::new(device),
            transformer_blocks: Transformer::new(
                TRANSFORMER_DIMENSION,
                TRANSFORMER_LAYERS,
                HEAD_DIMENSION,
                FEED_FORWARD_MULTIPLIER,
                device,
            ),
            task_heads: SumHead::new(device),
        }
    }

    /// スペクトログラムからフレームごとのロジットを計算する。
    ///
    /// * `spectrogram` - 形状 `[batch, times, MEL_BANDS]` のテンソル。
    pub(crate) fn forward(&self, spectrogram: Tensor<3>) -> ModelOutput {
        let hidden = self.frontend.forward(spectrogram);
        self.task_heads
            .forward(self.transformer_blocks.forward(hidden))
    }
}
