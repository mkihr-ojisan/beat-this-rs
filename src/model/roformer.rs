use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{gelu, sigmoid};
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Device, Int, Tensor};

/// ベクトルのノルムの下限。
const NORM_EPSILON: f64 = 1e-12;

/// 最終軸のベクトルを L2 正規化して学習済みの係数を乗算する層。
#[derive(Module, Debug)]
pub(super) struct RootMeanSquareNormalization {
    /// 次元ごとの係数。
    gamma: Param<Tensor<1>>,
    /// 次元数の平方根。
    scale: f64,
}

impl RootMeanSquareNormalization {
    /// 係数を 1 で初期化した層を作成する。
    ///
    /// * `dimension` - 最終軸の次元数。
    /// * `device` - パラメーターを配置するデバイス。
    pub(super) fn new(dimension: usize, device: &Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([dimension], device)),
            scale: (dimension as f64).sqrt(),
        }
    }

    /// `input` の最終軸を正規化する。
    ///
    /// * `input` - 最終軸の次元数が [`Self::new`] の `dimension` と等しいテンソル。
    pub(super) fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        let last_dimension = D - 1;
        let norm = input
            .clone()
            .powi_scalar(2)
            .sum_dim(last_dimension)
            .sqrt()
            .clamp_min(NORM_EPSILON);
        let mut gamma_shape = [1usize; D];
        gamma_shape[last_dimension] = self.gamma.dims()[0];
        (input / norm) * self.scale * self.gamma.val().reshape(gamma_shape)
    }
}

/// 全結合 2 層のフィードフォワード。
#[derive(Module, Debug)]
pub(super) struct FeedForward {
    /// 入力の正規化。
    norm: RootMeanSquareNormalization,
    /// 内部の次元へ射影する層。
    linear_in: Linear,
    /// 元の次元へ射影する層。
    linear_out: Linear,
}

impl FeedForward {
    /// フィードフォワードを作成する。
    ///
    /// * `dimension` - 入出力の次元数。
    /// * `multiplier` - 内部の次元数の `dimension` に対する倍率。
    /// * `device` - パラメーターを配置するデバイス。
    pub(super) fn new(dimension: usize, multiplier: usize, device: &Device) -> Self {
        let inner = dimension * multiplier;
        Self {
            norm: RootMeanSquareNormalization::new(dimension, device),
            linear_in: LinearConfig::new(dimension, inner).init(device),
            linear_out: LinearConfig::new(inner, dimension).init(device),
        }
    }

    /// 形状を維持して変換する。
    ///
    /// * `input` - 形状 `[batch, sequence, dimension]` のテンソル。
    pub(super) fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let hidden = self.linear_in.forward(self.norm.forward(input));
        self.linear_out.forward(gelu(hidden))
    }
}

/// 回転位置埋め込みの周波数。
#[derive(Module, Debug)]
struct RotaryEmbedding {
    /// 位置に乗算する角周波数。
    frequencies: Param<Tensor<1>>,
}

impl RotaryEmbedding {
    /// 周波数を 0 で初期化した埋め込みを作成する。
    ///
    /// * `frequency_count` - 周波数の数。
    /// * `device` - パラメーターを配置するデバイス。
    fn new(frequency_count: usize, device: &Device) -> Self {
        Self {
            frequencies: Param::from_tensor(Tensor::zeros([frequency_count], device)),
        }
    }

    /// 長さ `sequence_length` の系列の各位置に対する回転角の cos と sin を返す。形状は
    /// `[1, 1, sequence_length, 周波数の数, 1]`。
    ///
    /// * `sequence_length` - 系列長。
    /// * `device` - 計算に使用するデバイス。
    fn rotation(&self, sequence_length: usize, device: &Device) -> (Tensor<5>, Tensor<5>) {
        let positions = Tensor::<1, Int>::arange(0..sequence_length as i64, device)
            .float()
            .unsqueeze_dim::<2>(1);
        let angles = positions * self.frequencies.val().unsqueeze_dim::<2>(0);
        let [length, frequency_count] = angles.dims();
        let shape = [1, 1, length, frequency_count, 1];
        (
            angles.clone().cos().reshape(shape),
            angles.sin().reshape(shape),
        )
    }
}

/// `tensor` の最終軸の隣接する 2 要素を組として回転する。
///
/// * `tensor` - 形状 `[batch, heads, sequence, head_dimension]` のテンソル。
/// * `cosine` - [`RotaryEmbedding::rotation`] の cos。
/// * `sine` - [`RotaryEmbedding::rotation`] の sin。
fn rotate(tensor: Tensor<4>, cosine: &Tensor<5>, sine: &Tensor<5>) -> Tensor<4> {
    let [batch, heads, sequence, head_dimension] = tensor.dims();
    let pairs = tensor.reshape([batch, heads, sequence, head_dimension / 2, 2]);
    let first = pairs.clone().narrow(4, 0, 1);
    let second = pairs.narrow(4, 1, 1);
    let rotated_first = first.clone() * cosine.clone() - second.clone() * sine.clone();
    let rotated_second = second * cosine.clone() + first * sine.clone();
    Tensor::cat(vec![rotated_first, rotated_second], 4).reshape([
        batch,
        heads,
        sequence,
        head_dimension,
    ])
}

/// 回転位置埋め込みとゲートを使用する自己注意。
#[derive(Module, Debug)]
pub(super) struct Attention {
    /// クエリとキーに適用する回転位置埋め込み。
    rotary_embedding: RotaryEmbedding,
    /// 入力の正規化。
    norm: RootMeanSquareNormalization,
    /// クエリ、キー、バリューへの射影。
    to_qkv: Linear,
    /// ヘッドごとのゲートへの射影。
    to_gates: Linear,
    /// 出力の射影。
    to_out: Linear,
    /// ヘッドの数。
    heads: usize,
    /// ヘッドあたりの次元数。
    head_dimension: usize,
}

impl Attention {
    /// 自己注意を作成する。
    ///
    /// * `dimension` - 入出力の次元数。`head_dimension` で割り切れる必要がある。
    /// * `head_dimension` - ヘッドあたりの次元数。
    /// * `device` - パラメーターを配置するデバイス。
    pub(super) fn new(dimension: usize, head_dimension: usize, device: &Device) -> Self {
        let heads = dimension / head_dimension;
        Self {
            rotary_embedding: RotaryEmbedding::new(head_dimension / 2, device),
            norm: RootMeanSquareNormalization::new(dimension, device),
            to_qkv: LinearConfig::new(dimension, dimension * 3)
                .with_bias(false)
                .init(device),
            to_gates: LinearConfig::new(dimension, heads).init(device),
            to_out: LinearConfig::new(dimension, dimension)
                .with_bias(false)
                .init(device),
            heads,
            head_dimension,
        }
    }

    /// 形状を維持して系列方向の自己注意を適用する。
    ///
    /// * `input` - 形状 `[batch, sequence, dimension]` のテンソル。
    pub(super) fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let normalized = self.norm.forward(input);
        let [batch, sequence, _] = normalized.dims();
        let device = normalized.device();

        let mut qkv: Vec<Tensor<4>> = self
            .to_qkv
            .forward(normalized.clone())
            .chunk(3, 2)
            .into_iter()
            .map(|part| {
                part.reshape([batch, sequence, self.heads, self.head_dimension])
                    .swap_dims(1, 2)
            })
            .collect();
        let value = qkv.remove(2);
        let key = qkv.remove(1);
        let query = qkv.remove(0);

        let (cosine, sine) = self.rotary_embedding.rotation(sequence, &device);
        let query = rotate(query, &cosine, &sine);
        let key = rotate(key, &cosine, &sine);

        // スケールには既定値の 1/sqrt(head_dimension) を使用する
        let attended = attention(
            query,
            key,
            value,
            None,
            None,
            AttentionModuleOptions::default(),
        );

        let gates = sigmoid(self.to_gates.forward(normalized))
            .swap_dims(1, 2)
            .unsqueeze_dim::<4>(3);
        let gated = attended * gates;

        let output =
            gated
                .swap_dims(1, 2)
                .reshape([batch, sequence, self.heads * self.head_dimension]);
        self.to_out.forward(output)
    }
}

/// 自己注意とフィードフォワードの組。
#[derive(Module, Debug)]
pub(super) struct TransformerLayer {
    /// 自己注意。
    attention: Attention,
    /// フィードフォワード。
    feed_forward: FeedForward,
}

/// 全周波数を統合した時間方向の Transformer。
#[derive(Module, Debug)]
pub(super) struct Transformer {
    /// 順に適用する層。
    layers: Vec<TransformerLayer>,
    /// 出力の正規化。
    norm: RootMeanSquareNormalization,
}

impl Transformer {
    /// Transformer を作成する。
    ///
    /// * `dimension` - 入出力の次元数。
    /// * `depth` - 層の数。
    /// * `head_dimension` - 自己注意のヘッドあたりの次元数。
    /// * `multiplier` - フィードフォワード内部の次元数の倍率。
    /// * `device` - パラメーターを配置するデバイス。
    pub(super) fn new(
        dimension: usize,
        depth: usize,
        head_dimension: usize,
        multiplier: usize,
        device: &Device,
    ) -> Self {
        let layers = (0..depth)
            .map(|_| TransformerLayer {
                attention: Attention::new(dimension, head_dimension, device),
                feed_forward: FeedForward::new(dimension, multiplier, device),
            })
            .collect();
        Self {
            layers,
            norm: RootMeanSquareNormalization::new(dimension, device),
        }
    }

    /// 形状を維持して全層を適用する。
    ///
    /// * `input` - 形状 `[batch, times, dimension]` のテンソル。
    pub(super) fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let mut hidden = input;
        for layer in &self.layers {
            hidden = layer.attention.forward(hidden.clone()) + hidden;
            hidden = layer.feed_forward.forward(hidden.clone()) + hidden;
        }
        self.norm.forward(hidden)
    }
}
