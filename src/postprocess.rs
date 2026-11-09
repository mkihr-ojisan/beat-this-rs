/// ピーク検出の窓の片側フレーム数。
const PEAK_RADIUS: usize = 3;
/// 統合中のピークの平均位置と次のピーク位置の差 (フレーム数) がこの値以下のとき、次のピークを統合する。
const DEDUPLICATION_WIDTH: f64 = 1.0;

/// 検出されたビートとダウンビートの時刻。
#[derive(Clone, Debug, PartialEq)]
pub struct Beats {
    /// 昇順のビートの時刻 (ミリ秒)。
    pub beats_ms: Vec<f64>,
    /// 昇順のダウンビートの時刻 (ミリ秒)。すべて `beats_ms` に含まれる。ビートがないときはダウンビートの推定位置。
    pub downbeats_ms: Vec<f64>,
}

/// フレームごとのロジットからビートとダウンビートの時刻を算出する。
///
/// * `beat_logits` - ビートのロジット。
/// * `downbeat_logits` - ダウンビートのロジット。
pub(crate) fn logits_to_beats(beat_logits: &[f32], downbeat_logits: &[f32]) -> Beats {
    let beats_ms = frames_to_milliseconds(deduplicate_peaks(&pick_peaks(beat_logits)));
    let downbeats_ms = frames_to_milliseconds(deduplicate_peaks(&pick_peaks(downbeat_logits)));
    let downbeats_ms = snap_to_nearest_beat(downbeats_ms, &beats_ms);
    Beats {
        beats_ms,
        downbeats_ms,
    }
}

/// ロジットが正で、前後 `PEAK_RADIUS` フレームの最大値と一致するフレームを返す。
///
/// * `logits` - フレームごとのロジット。
fn pick_peaks(logits: &[f32]) -> Vec<usize> {
    (0..logits.len())
        .filter(|&frame| {
            let begin = frame.saturating_sub(PEAK_RADIUS);
            let end = logits.len().min(frame + PEAK_RADIUS + 1);
            let maximum = logits[begin..end]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            logits[frame] == maximum && logits[frame] > 0.0
        })
        .collect()
}

/// 隣接するピークを統合してフレーム番号の平均に置換する。
///
/// * `peaks` - 昇順のピークのフレーム番号。
fn deduplicate_peaks(peaks: &[usize]) -> Vec<f64> {
    let mut result = Vec::with_capacity(peaks.len());
    let mut iterator = peaks.iter().map(|&peak| peak as f64);
    let Some(first) = iterator.next() else {
        return result;
    };
    let (mut mean, mut count) = (first, 1.0);
    for peak in iterator {
        if peak - mean <= DEDUPLICATION_WIDTH {
            count += 1.0;
            mean += (peak - mean) / count;
        } else {
            result.push(mean);
            mean = peak;
            count = 1.0;
        }
    }
    result.push(mean);
    result
}

/// フレーム番号をミリ秒に変換する。
///
/// * `frames` - フレーム番号。
fn frames_to_milliseconds(frames: Vec<f64>) -> Vec<f64> {
    frames
        .into_iter()
        .map(|frame| frame * crate::mel::FRAME_MILLISECONDS)
        .collect()
}

/// ダウンビートを最も近いビートへ移動し、重複を除去する。ビートがないときは移動しない。
///
/// * `downbeats_ms` - 昇順のダウンビートの時刻 (ミリ秒)。
/// * `beats_ms` - 昇順のビートの時刻 (ミリ秒)。
fn snap_to_nearest_beat(mut downbeats_ms: Vec<f64>, beats_ms: &[f64]) -> Vec<f64> {
    for downbeat in &mut downbeats_ms {
        let distance = |beat: f64| (beat - *downbeat).abs();
        let insertion = beats_ms.partition_point(|beat| beat < downbeat);
        let end = beats_ms.len().min(insertion + 1);
        // min_by は距離が等しいときに前のビートを返す
        let nearest = beats_ms[insertion.saturating_sub(1)..end]
            .iter()
            .copied()
            .min_by(|a, b| distance(*a).total_cmp(&distance(*b)));
        if let Some(nearest) = nearest {
            *downbeat = nearest;
        }
    }
    downbeats_ms.dedup();
    downbeats_ms
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `peaks` 以外のフレームの値が -5.0 のロジットを作成する。
    ///
    /// * `length` - フレーム数。
    /// * `peaks` - フレーム番号とロジットの組。
    fn logits(length: usize, peaks: &[(usize, f32)]) -> Vec<f32> {
        let mut logits = vec![-5.0; length];
        for &(frame, value) in peaks {
            logits[frame] = value;
        }
        logits
    }

    #[test]
    fn peaks_must_be_local_maxima_and_positive() {
        let logits = logits(20, &[(5, 2.0), (8, 1.0), (15, -1.0)]);
        assert_eq!(pick_peaks(&logits), vec![5]);
    }

    #[test]
    fn peaks_at_the_edges_are_detected() {
        let logits = logits(10, &[(0, 1.0), (9, 2.0)]);
        assert_eq!(pick_peaks(&logits), vec![0, 9]);
    }

    #[test]
    fn peaks_further_than_the_radius_are_independent() {
        let logits = logits(20, &[(5, 2.0), (9, 1.0)]);
        assert_eq!(pick_peaks(&logits), vec![5, 9]);
    }

    #[test]
    fn deduplication_averages_adjacent_peaks() {
        assert_eq!(deduplicate_peaks(&[10, 11, 20]), vec![10.5, 20.0]);
        assert_eq!(deduplicate_peaks(&[10, 11, 12, 20]), vec![10.5, 12.0, 20.0]);
        assert_eq!(deduplicate_peaks(&[5]), vec![5.0]);
        assert_eq!(deduplicate_peaks(&[]), Vec::<f64>::new());
    }

    #[test]
    fn tied_plateau_becomes_half_frame() {
        let logits = logits(20, &[(5, 2.0), (6, 2.0)]);
        let beats = logits_to_beats(&logits, &[-5.0; 20]);
        assert_eq!(beats.beats_ms, vec![110.0]);
    }

    #[test]
    fn downbeats_snap_to_the_nearest_beat_and_are_deduplicated() {
        assert_eq!(
            snap_to_nearest_beat(vec![90.0, 110.0, 390.0], &[100.0, 200.0, 400.0]),
            vec![100.0, 400.0]
        );
    }

    #[test]
    fn tied_downbeat_moves_to_the_earlier_beat() {
        assert_eq!(
            snap_to_nearest_beat(vec![150.0], &[100.0, 200.0]),
            vec![100.0]
        );
    }

    #[test]
    fn downbeats_outside_the_beat_range_snap_to_the_edges() {
        assert_eq!(
            snap_to_nearest_beat(vec![50.0, 500.0], &[100.0, 200.0]),
            vec![100.0, 200.0]
        );
    }

    #[test]
    fn downbeats_without_beats_stay_in_place() {
        assert_eq!(snap_to_nearest_beat(vec![50.0], &[]), vec![50.0]);
    }

    #[test]
    fn frames_convert_to_milliseconds() {
        let logits_beat = logits(100, &[(50, 3.0)]);
        let logits_downbeat = logits(100, &[(52, 3.0)]);
        let beats = logits_to_beats(&logits_beat, &logits_downbeat);
        assert_eq!(beats.beats_ms, vec![1000.0]);
        assert_eq!(beats.downbeats_ms, vec![1000.0]);
    }
}
