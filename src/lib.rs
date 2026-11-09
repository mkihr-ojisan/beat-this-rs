//! [Beat This!](https://github.com/CPJKU/beat_this) のビート・ダウンビート検出を burn で実行するライブラリ。

mod chunking;
mod error;
mod mel;
mod model;
mod postprocess;
mod resample;
mod tracker;

pub use error::{Error, Result};
pub use postprocess::Beats;
pub use tracker::{BeatTracker, FrameLogits};
