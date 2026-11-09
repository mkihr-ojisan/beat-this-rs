use std::f64::consts::PI;
use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

use crate::error::{Error, Result};

/// メル帯域の数。
pub(crate) const MEL_BANDS: usize = 128;
/// 入力のサンプルレート (Hz)。
const SAMPLE_RATE: f64 = 22050.0;
/// FFT の長さ。
const FFT_LENGTH: usize = 1024;
/// ホップ長 (サンプル)。
const HOP_LENGTH: usize = 441;
/// フレーム間隔 (ミリ秒)。
pub(crate) const FRAME_MILLISECONDS: f64 = HOP_LENGTH as f64 * 1000.0 / SAMPLE_RATE;
/// メル帯域の下限 (Hz)。
const MINIMUM_FREQUENCY: f64 = 30.0;
/// メル帯域の上限 (Hz)。
const MAXIMUM_FREQUENCY: f64 = 11000.0;
/// log1p の前に乗算する係数。
const LOG_MULTIPLIER: f32 = 1000.0;

/// 対数メルスペクトログラム。
pub(crate) struct Spectrogram {
    /// フレーム数。
    pub(crate) frames: usize,
    /// 時間を先頭の軸とした `frames × MEL_BANDS` の値。
    pub(crate) values: Vec<f32>,
}

/// 1 つのメル帯域の三角フィルター。
struct MelFilter {
    /// 最初の非ゼロのビン番号。
    first_bin: usize,
    /// `first_bin` から続くビンの重み。
    weights: Vec<f32>,
}

/// 22050 Hz の信号から対数メルスペクトログラムを計算する。
pub(crate) struct LogMelSpectrogram {
    /// 周期的な Hann 窓。
    window: Vec<f32>,
    /// 低い順に並んだメル帯域のフィルター。
    filters: Vec<MelFilter>,
    /// 長さ `FFT_LENGTH` の順変換。
    fft: Arc<dyn RealToComplex<f32>>,
}

impl LogMelSpectrogram {
    /// 対数メルスペクトログラムの計算器を作成する。
    pub(crate) fn new() -> Self {
        let window = (0..FFT_LENGTH)
            .map(|index| {
                let phase = 2.0 * PI * index as f64 / FFT_LENGTH as f64;
                (0.5 - 0.5 * phase.cos()) as f32
            })
            .collect();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT_LENGTH);
        Self {
            window,
            filters: mel_filters(),
            fft,
        }
    }

    /// `signal` のスペクトログラムを計算する。
    ///
    /// * `signal` - 22050 Hz のモノラル信号。
    ///
    /// `signal` が短すぎて両端の反射パディングを実行できないときは [`Error::AudioTooShort`] を返す。
    pub(crate) fn compute(&self, signal: &[f32]) -> Result<Spectrogram> {
        let padding = FFT_LENGTH / 2;
        if signal.len() <= padding {
            return Err(Error::AudioTooShort {
                samples: signal.len(),
                minimum: padding + 1,
            });
        }

        let frames = signal.len() / HOP_LENGTH + 1;
        let mut values = vec![0.0f32; frames * MEL_BANDS];
        let mut frame = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut magnitudes = vec![0.0f32; spectrum.len()];
        let frame_length_norm = (FFT_LENGTH as f32).sqrt();
        for (frame_index, output) in values.chunks_exact_mut(MEL_BANDS).enumerate() {
            let center = frame_index * HOP_LENGTH;
            let source = center
                .checked_sub(padding)
                .and_then(|begin| signal.get(begin..begin + FFT_LENGTH));
            if let Some(source) = source {
                for ((sample, input), window) in frame.iter_mut().zip(source).zip(&self.window) {
                    *sample = input * window;
                }
            } else {
                let begin = center as isize - padding as isize;
                for (offset, (sample, window)) in frame.iter_mut().zip(&self.window).enumerate() {
                    *sample = reflected_sample(signal, begin + offset as isize) * window;
                }
            }
            self.fft
                .process(&mut frame, &mut spectrum)
                .expect("buffer lengths come from the planner");
            for (magnitude, bin) in magnitudes.iter_mut().zip(&spectrum) {
                *magnitude = bin.norm() / frame_length_norm;
            }
            for (band, filter) in output.iter_mut().zip(&self.filters) {
                let bins = &magnitudes[filter.first_bin..filter.first_bin + filter.weights.len()];
                let energy: f32 = bins
                    .iter()
                    .zip(&filter.weights)
                    .map(|(magnitude, weight)| magnitude * weight)
                    .sum();
                *band = (LOG_MULTIPLIER * energy).ln_1p();
            }
        }
        Ok(Spectrogram { frames, values })
    }
}

/// `signal` の両端を反射パディングした信号のサンプルを返す。
///
/// * `signal` - 元の信号。
/// * `index` - `signal` の先頭を 0 とする位置。`-(signal.len() - 1)` 以上 `2 * (signal.len() - 1)` 以下。
fn reflected_sample(signal: &[f32], index: isize) -> f32 {
    let last = signal.len() as isize - 1;
    let distance = index.abs();
    let position = if distance > last {
        2 * last - distance
    } else {
        distance
    };
    signal[position as usize]
}

/// Slaney 尺度の三角フィルターバンクを作成する。
fn mel_filters() -> Vec<MelFilter> {
    let bin_count = FFT_LENGTH / 2 + 1;
    let nyquist = SAMPLE_RATE / 2.0;
    let bin_frequencies: Vec<f64> = (0..bin_count)
        .map(|index| nyquist * index as f64 / (bin_count - 1) as f64)
        .collect();

    let minimum_mel = hz_to_mel(MINIMUM_FREQUENCY);
    let maximum_mel = hz_to_mel(MAXIMUM_FREQUENCY);
    let points: Vec<f64> = (0..MEL_BANDS + 2)
        .map(|index| {
            let mel =
                minimum_mel + (maximum_mel - minimum_mel) * index as f64 / (MEL_BANDS + 1) as f64;
            mel_to_hz(mel)
        })
        .collect();

    points
        .windows(3)
        .map(|edges| {
            let [lower, center, upper] = [edges[0], edges[1], edges[2]];
            let weights: Vec<f32> = bin_frequencies
                .iter()
                .map(|frequency| {
                    let rising = (frequency - lower) / (center - lower);
                    let falling = (upper - frequency) / (upper - center);
                    rising.min(falling).max(0.0) as f32
                })
                .collect();
            let first_bin = weights
                .iter()
                .position(|weight| *weight > 0.0)
                .unwrap_or(bin_count);
            let end = weights
                .iter()
                .rposition(|weight| *weight > 0.0)
                .map_or(first_bin, |index| index + 1);
            MelFilter {
                first_bin,
                weights: weights[first_bin..end].to_vec(),
            }
        })
        .collect()
}

/// 周波数を Slaney 尺度のメル値に変換する。
///
/// * `frequency` - 周波数 (Hz)。
fn hz_to_mel(frequency: f64) -> f64 {
    if frequency >= 1000.0 {
        15.0 + (frequency / 1000.0).ln() / mel_log_step()
    } else {
        frequency * 3.0 / 200.0
    }
}

/// Slaney 尺度のメル値を周波数に変換する。
///
/// * `mel` - メル値。
fn mel_to_hz(mel: f64) -> f64 {
    if mel >= 15.0 {
        1000.0 * (mel_log_step() * (mel - 15.0)).exp()
    } else {
        mel * 200.0 / 3.0
    }
}

/// Slaney 尺度の対数部分における 1 メルあたりの増分を返す。
fn mel_log_step() -> f64 {
    6.4f64.ln() / 27.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_follows_hop_length() {
        let mel = LogMelSpectrogram::new();
        for (length, frames) in [(513, 2), (882, 3), (22050, 51)] {
            let spectrogram = mel.compute(&vec![0.0; length]).unwrap();
            assert_eq!(spectrogram.frames, frames);
            assert_eq!(spectrogram.values.len(), frames * MEL_BANDS);
        }
    }

    #[test]
    fn short_signal_is_rejected() {
        let mel = LogMelSpectrogram::new();
        assert!(matches!(
            mel.compute(&[0.0; 512]),
            Err(Error::AudioTooShort {
                samples: 512,
                minimum: 513
            })
        ));
    }

    #[test]
    fn silence_produces_zero_values() {
        let mel = LogMelSpectrogram::new();
        let spectrogram = mel.compute(&vec![0.0; 4410]).unwrap();
        assert!(spectrogram.values.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn filters_cover_the_expected_range() {
        let filters = mel_filters();
        assert_eq!(filters.len(), MEL_BANDS);
        // 最下位の帯域は先頭付近のビンから開始し、最上位の帯域は 11000 Hz 未満で終了する
        assert!(filters[0].first_bin <= 2);
        let last = filters.last().unwrap();
        let last_bin = last.first_bin + last.weights.len() - 1;
        assert!((last_bin as f64) * 11025.0 / 512.0 < MAXIMUM_FREQUENCY);
    }

    #[test]
    fn sine_peak_is_in_the_matching_band() {
        let mel = LogMelSpectrogram::new();
        let signal: Vec<f32> = (0..22050)
            .map(|index| (2.0 * PI * 1000.0 * index as f64 / SAMPLE_RATE).sin() as f32)
            .collect();
        let spectrogram = mel.compute(&signal).unwrap();
        let frame = &spectrogram.values[MEL_BANDS * 20..MEL_BANDS * 21];
        let peak = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap();
        let lower = hz_to_mel(MINIMUM_FREQUENCY);
        let upper = hz_to_mel(MAXIMUM_FREQUENCY);
        let expected = (0..MEL_BANDS)
            .min_by_key(|band| {
                let center = lower + (upper - lower) * (*band + 1) as f64 / (MEL_BANDS + 1) as f64;
                ((mel_to_hz(center) - 1000.0).abs() * 1e6) as u64
            })
            .unwrap();
        assert!(peak.abs_diff(expected) <= 1);
    }
}
