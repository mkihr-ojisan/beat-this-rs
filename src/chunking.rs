use std::ops::Range;

/// チャンクのフレーム数。
const CHUNK_FRAMES: usize = 1500;
/// チャンクの両端で破棄するフレーム数。
const BORDER_FRAMES: usize = 6;
/// チャンク間の移動量 (フレーム)。
const STRIDE_FRAMES: usize = CHUNK_FRAMES - 2 * BORDER_FRAMES;
/// いずれのチャンクにも割り当てられなかったフレームのロジット。
const UNASSIGNED_LOGIT: f32 = -1000.0;

/// スペクトログラムから抽出するチャンクの範囲。
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Chunk {
    /// チャンクの先頭に対応するフレーム番号。先頭にパディングがあるときは負になる。
    start: i64,
    /// スペクトログラムから抽出するフレームの範囲。
    pub(crate) source: Range<usize>,
    /// 先頭に追加するゼロのフレーム数。
    pub(crate) left_padding: usize,
    /// 末尾に追加するゼロのフレーム数。
    pub(crate) right_padding: usize,
}

impl Chunk {
    /// パディングを含むチャンクのフレーム数を返す。
    pub(crate) fn len(&self) -> usize {
        self.left_padding + self.source.len() + self.right_padding
    }
}

/// `frames` フレームの曲をチャンクに分割する。
///
/// * `frames` - スペクトログラムのフレーム数。
pub(crate) fn split(frames: usize) -> Vec<Chunk> {
    let frames = frames as i64;
    let border = BORDER_FRAMES as i64;
    let chunk_frames = CHUNK_FRAMES as i64;

    let mut starts: Vec<i64> = (-border..frames - border).step_by(STRIDE_FRAMES).collect();
    if frames > STRIDE_FRAMES as i64
        && let Some(last) = starts.last_mut()
    {
        *last = frames - (chunk_frames - border);
    }

    starts
        .into_iter()
        .map(|start| Chunk {
            start,
            source: start.max(0) as usize..(start + chunk_frames).min(frames) as usize,
            left_padding: (-start).max(0) as usize,
            right_padding: border.min(start + chunk_frames - frames).max(0) as usize,
        })
        .collect()
}

/// チャンクごとの予測を曲全体の予測に集約する。チャンクが重複する部分では前のチャンクを優先する。
///
/// * `chunks` - [`split`] の結果。
/// * `predictions` - `chunks` と同じ順序のチャンクごとの予測。各要素の長さは [`Chunk::len`]。
/// * `frames` - スペクトログラムのフレーム数。
pub(crate) fn aggregate(chunks: &[Chunk], predictions: &[Vec<f32>], frames: usize) -> Vec<f32> {
    let mut piece = vec![UNASSIGNED_LOGIT; frames];
    let border = BORDER_FRAMES as i64;
    for (chunk, prediction) in chunks.iter().zip(predictions).rev() {
        let inner = &prediction[BORDER_FRAMES..prediction.len() - BORDER_FRAMES];
        let begin = (chunk.start + border) as usize;
        let end = frames.min((chunk.start + CHUNK_FRAMES as i64 - border) as usize);
        piece[begin..end].copy_from_slice(&inner[..end - begin]);
    }
    piece
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 指定した値の [`Chunk`] を作成する。
    ///
    /// * `start` - チャンクの先頭に対応するフレーム番号。
    /// * `source` - スペクトログラムから抽出するフレームの範囲。
    /// * `left` - 先頭に追加するゼロのフレーム数。
    /// * `right` - 末尾に追加するゼロのフレーム数。
    fn chunk(start: i64, source: Range<usize>, left: usize, right: usize) -> Chunk {
        Chunk {
            start,
            source,
            left_padding: left,
            right_padding: right,
        }
    }

    #[test]
    fn short_piece_is_a_single_chunk_padded_by_the_border() {
        assert_eq!(split(100), vec![chunk(-6, 0..100, 6, 6)]);
        assert_eq!(split(100)[0].len(), 112);
    }

    #[test]
    fn piece_of_exactly_one_stride_is_not_shifted() {
        assert_eq!(split(1488), vec![chunk(-6, 0..1488, 6, 6)]);
        assert_eq!(split(1488)[0].len(), 1500);
    }

    #[test]
    fn last_start_is_shifted_to_the_end() {
        assert_eq!(
            split(1489),
            vec![chunk(-6, 0..1489, 6, 5), chunk(-5, 0..1489, 5, 6)]
        );
        assert_eq!(
            split(1500),
            vec![chunk(-6, 0..1494, 6, 0), chunk(6, 6..1500, 0, 6)]
        );
        assert_eq!(
            split(3000),
            vec![
                chunk(-6, 0..1494, 6, 0),
                chunk(1482, 1482..2982, 0, 0),
                chunk(1506, 1506..3000, 0, 6)
            ]
        );
    }

    #[test]
    fn all_chunks_of_long_pieces_are_full_length() {
        for frames in [1489, 1494, 1500, 2976, 2977, 4500, 100_000] {
            assert!(
                split(frames)
                    .iter()
                    .all(|chunk| chunk.len() == CHUNK_FRAMES)
            );
        }
    }

    #[test]
    fn aggregate_covers_every_frame_and_keeps_the_first_chunk() {
        for frames in [1, 100, 1488, 1489, 1500, 2976, 2977, 3000, 4500] {
            let chunks = split(frames);
            let predictions: Vec<Vec<f32>> = chunks
                .iter()
                .enumerate()
                .map(|(index, chunk)| vec![index as f32; chunk.len()])
                .collect();
            let piece = aggregate(&chunks, &predictions, frames);
            assert_eq!(piece.len(), frames);
            assert!(piece.iter().all(|value| *value != UNASSIGNED_LOGIT));
            // 各フレームは、そのフレームを含む最初のチャンクの値になる
            for (frame, value) in piece.iter().enumerate() {
                let expected = chunks
                    .iter()
                    .position(|chunk| {
                        let begin = chunk.start + BORDER_FRAMES as i64;
                        let end = chunk.start + (CHUNK_FRAMES - BORDER_FRAMES) as i64;
                        (begin..end).contains(&(frame as i64))
                    })
                    .unwrap();
                assert_eq!(*value, expected as f32, "frames={frames}, frame={frame}");
            }
        }
    }
}
