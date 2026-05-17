use std::collections::VecDeque;

use crate::episode::{EpisodeRecord, EpisodeTrace};

/// Hot-tier episodic buffer: a bounded ring of recent episode traces.
///
/// Each trace carries the episode vector plus per-region encoder inputs needed for sleep replay.
/// When full, the oldest trace is evicted; the sleep loop is responsible for spilling to cold.
pub struct EpisodicBuffer {
    capacity: usize,
    buf: VecDeque<EpisodeTrace>,
    next_id: u64,
    /// Index into `buf` of the next trace to replay. Reset to 0 after each sleep cycle.
    pub replay_checkpoint: usize,
}

impl EpisodicBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buf: VecDeque::with_capacity(capacity),
            next_id: 0,
            replay_checkpoint: 0,
        }
    }

    /// Push a trace; returns the assigned id.
    /// `regional` contains the per-region encoder embeddings for this episode.
    pub fn push(&mut self, timestamp_secs: f64, values: Vec<f32>, regional: [Vec<f32>; 5]) -> u64 {
        if self.buf.len() == self.capacity {
            self.buf.pop_front();
            // Adjust checkpoint: if the evicted element was before the checkpoint,
            // the checkpoint shifts by one to stay aligned.
            if self.replay_checkpoint > 0 {
                self.replay_checkpoint -= 1;
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.buf.push_back(EpisodeTrace {
            record: EpisodeRecord {
                id,
                timestamp_secs,
                values,
            },
            regional,
            text_target: None,
        });
        id
    }

    /// Attach a text-bypass token sequence to the most recently pushed trace.
    ///
    /// Called when the user submits a query while a given episode is current;
    /// the query text serves as the decoder projection training target (ADR 0012).
    pub fn set_last_text_target(&mut self, target: Vec<u32>) {
        if let Some(trace) = self.buf.back_mut() {
            trace.text_target = Some(target);
        }
    }

    pub fn iter_traces(&self) -> impl Iterator<Item = &EpisodeTrace> {
        self.buf.iter()
    }

    /// Iterate EpisodeRecords (without regional data) — used for cold spill.
    pub fn iter_records(&self) -> impl Iterator<Item = &EpisodeRecord> {
        self.buf.iter().map(|t| &t.record)
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Maximum id currently in the buffer, or None if empty.
    pub fn max_id(&self) -> Option<u64> {
        self.buf.back().map(|t| t.record.id)
    }

    /// Push a trace with an explicit id (used when restoring from snapshot).
    pub(crate) fn push_with_id(
        &mut self,
        id: u64,
        timestamp_secs: f64,
        values: Vec<f32>,
        regional: [Vec<f32>; 5],
    ) {
        if self.buf.len() == self.capacity {
            self.buf.pop_front();
            if self.replay_checkpoint > 0 {
                self.replay_checkpoint -= 1;
            }
        }
        self.buf.push_back(EpisodeTrace {
            record: EpisodeRecord {
                id,
                timestamp_secs,
                values,
            },
            regional,
            text_target: None,
        });
    }

    /// Override the next-id counter (used when restoring from snapshot).
    pub(crate) fn next_id_counter(&self) -> u64 {
        self.next_id
    }

    pub(crate) fn set_next_id(&mut self, next: u64) {
        self.next_id = next;
    }

    /// Clear all traces and reset the replay checkpoint.
    ///
    /// Called by the sleep controller after a successful hot→cold spill, so the
    /// high-water trigger does not fire again on the same data.
    pub fn clear(&mut self) {
        self.buf.clear();
        self.replay_checkpoint = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trace(id_hint: usize) -> (f64, Vec<f32>, [Vec<f32>; 5]) {
        let v = id_hint as f32;
        (
            v as f64,
            vec![v; 4],
            [vec![v; 4], vec![], vec![], vec![], vec![]],
        )
    }

    #[test]
    fn evicts_oldest_at_capacity() {
        let mut buf = EpisodicBuffer::new(3);
        let (ts, vals, reg) = make_trace(0);
        let id0 = buf.push(ts, vals, reg);
        for i in 1..4 {
            let (ts, vals, reg) = make_trace(i);
            buf.push(ts, vals, reg);
        }
        assert_eq!(buf.len(), 3);
        assert!(buf.iter_traces().all(|t| t.record.id != id0));
    }

    #[test]
    fn ids_are_monotonic() {
        let mut buf = EpisodicBuffer::new(10);
        for i in 0..5 {
            let (ts, vals, reg) = make_trace(i);
            let id = buf.push(ts, vals, reg);
            assert_eq!(id, i as u64);
        }
    }

    #[test]
    fn regional_stored_correctly() {
        let mut buf = EpisodicBuffer::new(10);
        let regional = [vec![1.0f32; 4], vec![2.0f32; 5], vec![], vec![], vec![]];
        buf.push(0.0, vec![0.0; 4], regional.clone());
        let trace = buf.iter_traces().next().unwrap();
        assert_eq!(trace.regional[0], vec![1.0f32; 4]);
        assert_eq!(trace.regional[1], vec![2.0f32; 5]);
    }
}
