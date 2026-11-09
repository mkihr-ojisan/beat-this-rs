/// ビート検出で発生するエラー。
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// チェックポイントの読み込みに失敗した。
    #[error("failed to load checkpoint")]
    Checkpoint(#[from] burn::store::PytorchStoreError),
    /// 入力が短すぎてスペクトログラムを計算できない。
    #[error("audio is too short: {samples} samples at 22050 Hz, at least {minimum} required")]
    AudioTooShort {
        /// 22050 Hz に変換した後の入力のサンプル数。
        samples: usize,
        /// 必要な最小サンプル数。
        minimum: usize,
    },
    /// 入力に NaN または無限大のサンプルが含まれる。
    #[error("audio contains a non-finite sample at index {index}")]
    NonFiniteSample {
        /// NaN または無限大である最初のサンプルの位置。
        index: usize,
    },
    /// 推論結果のテンソルを取得できなかった。
    #[error("failed to read tensor")]
    Tensor(#[from] burn::tensor::TensorReadError),
}

/// [`Error`] を返す [`std::result::Result`]。
pub type Result<T> = std::result::Result<T, Error>;
