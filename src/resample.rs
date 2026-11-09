use std::f64::consts::PI;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// 入力のサンプルレート (Hz)。
const INPUT_SAMPLE_RATE: f64 = 44100.0;
/// FIR の中心を除く片側のタップ数。
const HALF_TAPS: usize = 186;
/// Kaiser 窓の形状パラメーター。
const KAISER_BETA: f64 = 12.25;
/// sinc のカットオフ周波数 (Hz)。
const CUTOFF_HZ: f64 = 10547.5;
/// FFT の長さ。
const FFT_LENGTH: usize = 4096;

/// 44.1 kHz の信号を 22050 Hz に間引く FIR フィルター。
pub(crate) struct Decimator {
    /// FIR を FFT した結果。
    taps_spectrum: Vec<Complex<f32>>,
    /// 長さ `FFT_LENGTH` の順変換。
    forward: Arc<dyn RealToComplex<f32>>,
    /// 長さ `FFT_LENGTH` の逆変換。
    inverse: Arc<dyn ComplexToReal<f32>>,
}

impl Decimator {
    /// フィルターを作成する。
    pub(crate) fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(FFT_LENGTH);
        let inverse = planner.plan_fft_inverse(FFT_LENGTH);

        let mut time_domain = forward.make_input_vec();
        let taps = kaiser_sinc_taps();
        time_domain[..taps.len()].copy_from_slice(&taps);
        let mut taps_spectrum = forward.make_output_vec();
        forward
            .process(&mut time_domain, &mut taps_spectrum)
            .expect("buffer lengths come from the planner");

        Self {
            taps_spectrum,
            forward,
            inverse,
        }
    }

    /// `input` を 22050 Hz へ変換する。
    ///
    /// 入力の範囲外は 0 として扱い、出力の先頭は入力の先頭と時刻が一致する。
    ///
    /// * `input` - 44.1 kHz のモノラル信号。
    ///
    /// 戻り値の長さは `input.len().div_ceil(2)`。
    pub(crate) fn process(&self, input: &[f32]) -> Vec<f32> {
        let taps_length = 2 * HALF_TAPS + 1;
        let block_length = FFT_LENGTH - taps_length + 1;
        let scale = 1.0 / FFT_LENGTH as f32;

        // 重畳加算で FIR との線形畳み込みを計算し、群遅延 HALF_TAPS を補正して 1 サンプルおきに抽出する
        let mut output = vec![0.0f32; input.len().div_ceil(2)];
        let mut time_domain = self.forward.make_input_vec();
        let mut spectrum = self.forward.make_output_vec();
        for (block_index, block) in input.chunks(block_length).enumerate() {
            time_domain[..block.len()].copy_from_slice(block);
            time_domain[block.len()..].fill(0.0);
            self.forward
                .process(&mut time_domain, &mut spectrum)
                .expect("buffer lengths come from the planner");
            for (bin, taps_bin) in spectrum.iter_mut().zip(&self.taps_spectrum) {
                *bin *= *taps_bin;
            }
            self.inverse
                .process(&mut spectrum, &mut time_domain)
                .expect("buffer lengths come from the planner");

            let block_start = block_index * block_length;
            for (index, value) in time_domain.iter().enumerate() {
                let Some(delayed) = (block_start + index).checked_sub(HALF_TAPS) else {
                    continue;
                };
                if delayed % 2 == 0
                    && let Some(sample) = output.get_mut(delayed / 2)
                {
                    *sample += *value * scale;
                }
            }
        }
        output
    }
}

/// 直流ゲインが 1 になるよう正規化した Kaiser 窓 sinc を返す。
fn kaiser_sinc_taps() -> Vec<f32> {
    let half = HALF_TAPS as f64;
    let normalized_cutoff = CUTOFF_HZ / INPUT_SAMPLE_RATE;
    let denominator = modified_bessel_i0(KAISER_BETA);
    let taps: Vec<f64> = (-(HALF_TAPS as i64)..=HALF_TAPS as i64)
        .map(|index| {
            let position = index as f64;
            let sinc = if index == 0 {
                2.0 * normalized_cutoff
            } else {
                (2.0 * PI * normalized_cutoff * position).sin() / (PI * position)
            };
            let ratio = position / half;
            let window =
                modified_bessel_i0(KAISER_BETA * (1.0 - ratio * ratio).sqrt()) / denominator;
            sinc * window
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    taps.iter().map(|tap| (*tap / sum) as f32).collect()
}

/// 第 1 種変形ベッセル関数 I0 の値を返す。
///
/// * `x` - 引数。
fn modified_bessel_i0(x: f64) -> f64 {
    let half_x_squared = (x / 2.0) * (x / 2.0);
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut order = 1.0;
    loop {
        term *= half_x_squared / (order * order);
        sum += term;
        if term < sum * 1e-17 {
            return sum;
        }
        order += 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIR の振幅特性をデシベルで返す。
    ///
    /// * `frequency` - 周波数 (Hz)。
    fn gain_db(frequency: f64) -> f64 {
        let taps = kaiser_sinc_taps();
        let (mut real, mut imaginary) = (0.0f64, 0.0f64);
        for (index, tap) in taps.iter().enumerate() {
            let position = index as f64 - HALF_TAPS as f64;
            let angle = 2.0 * PI * frequency * position / INPUT_SAMPLE_RATE;
            real += *tap as f64 * angle.cos();
            imaginary += *tap as f64 * angle.sin();
        }
        20.0 * (real.hypot(imaginary)).log10()
    }

    #[test]
    fn passband_and_stopband_follow_the_reference_filter() {
        assert!(gain_db(1000.0).abs() < 1e-3);
        assert!((gain_db(10300.0) + 0.21).abs() < 0.1);
        assert!((gain_db(10500.0) + 3.8).abs() < 0.2);
        assert!((gain_db(10950.0) + 72.0).abs() < 1.0);
    }

    #[test]
    fn output_length_rounds_up() {
        let decimator = Decimator::new();
        assert_eq!(decimator.process(&[]).len(), 0);
        assert_eq!(decimator.process(&[0.0; 3]).len(), 2);
        assert_eq!(decimator.process(&[0.0; 44100]).len(), 22050);
        assert_eq!(decimator.process(&[0.0; 44101]).len(), 22051);
    }

    #[test]
    fn dc_is_preserved_away_from_the_edges() {
        let decimator = Decimator::new();
        let output = decimator.process(&vec![1.0; 20000]);
        for value in &output[200..output.len() - 200] {
            assert!((value - 1.0).abs() < 1e-4, "{value}");
        }
    }

    #[test]
    fn impulse_stays_at_its_time() {
        let decimator = Decimator::new();
        let mut input = vec![0.0; 10000];
        input[5000] = 1.0;
        let output = decimator.process(&input);
        let peak = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index);
        assert_eq!(peak, Some(2500));
    }
}
