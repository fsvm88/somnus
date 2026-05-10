use std::collections::VecDeque;

use crate::episode::EpisodeRecord;

/// Hot-tier episodic buffer: a bounded ring of recent episodes.
///
/// When full, the oldest episode is evicted (no cold-archive promotion in Phase 4;
/// that is Phase 5 sleep-loop work).
pub struct EpisodicBuffer {
    capacity: usize,
    buf: VecDeque<EpisodeRecord>,
    next_id: u64,
}

impl EpisodicBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buf: VecDeque::with_capacity(capacity),
            next_id: 0,
        }
    }

    /// Push an episode; returns the assigned id.
    pub fn push(&mut self, timestamp_secs: f64, values: Vec<f32>) -> u64 {
        if self.buf.len() == self.capacity {
            self.buf.pop_front();
        }
        let id = self.next_id;
        self.next_id += 1;
        self.buf.push_back(EpisodeRecord {
            id,
            timestamp_secs,
            values,
        });
        id
    }

    pub fn iter(&self) -> impl Iterator<Item = &EpisodeRecord> {
        self.buf.iter()
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_oldest_at_capacity() {
        let mut buf = EpisodicBuffer::new(3);
        let id0 = buf.push(0.0, vec![0.0; 4]);
        let _id1 = buf.push(1.0, vec![1.0; 4]);
        let _id2 = buf.push(2.0, vec![2.0; 4]);
        assert_eq!(buf.len(), 3);
        // Push a 4th — id0 must be evicted.
        buf.push(3.0, vec![3.0; 4]);
        assert_eq!(buf.len(), 3);
        assert!(buf.iter().all(|r| r.id != id0));
    }

    #[test]
    fn ids_are_monotonic() {
        let mut buf = EpisodicBuffer::new(10);
        for i in 0..5 {
            let id = buf.push(i as f64, vec![i as f32; 4]);
            assert_eq!(id, i as u64);
        }
    }
}
