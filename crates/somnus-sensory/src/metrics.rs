use std::sync::atomic::{AtomicU64, Ordering};

/// Per-source frame counters. Embedded in `Arc<SensorMetrics>` so both the
/// capture side and the consuming main loop can observe them without locking.
#[derive(Debug, Default)]
pub struct SensorMetrics {
    pub frames_captured: AtomicU64,
    pub frames_dropped: AtomicU64,
}

impl SensorMetrics {
    pub fn increment_captured(&self) {
        self.frames_captured.fetch_add(1, Ordering::Relaxed);
    }

    pub fn increment_dropped(&self) {
        self.frames_dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Point-in-time snapshot. The two counters are read in separate atomic
    /// operations, so they are not mutually consistent — close enough for logging.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            frames_captured: self.frames_captured.load(Ordering::Relaxed),
            frames_dropped: self.frames_dropped.load(Ordering::Relaxed),
        }
    }
}

/// Immutable copy of `SensorMetrics` at a moment in time.
#[derive(Debug, Clone, Copy)]
pub struct MetricsSnapshot {
    pub frames_captured: u64,
    pub frames_dropped: u64,
}

impl MetricsSnapshot {
    /// Fraction of produced frames that were dropped (0.0–1.0).
    /// Returns 0.0 when no frames have been produced yet.
    pub fn drop_rate(&self) -> f64 {
        let total = self.frames_captured + self.frames_dropped;
        if total == 0 {
            0.0
        } else {
            self.frames_dropped as f64 / total as f64
        }
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_increment() {
        let m = SensorMetrics::default();
        m.increment_captured();
        m.increment_captured();
        m.increment_dropped();
        let s = m.snapshot();
        assert_eq!(s.frames_captured, 2);
        assert_eq!(s.frames_dropped, 1);
    }

    #[test]
    fn drop_rate_zero_when_empty() {
        let s = MetricsSnapshot {
            frames_captured: 0,
            frames_dropped: 0,
        };
        assert_eq!(s.drop_rate(), 0.0);
    }

    #[test]
    fn drop_rate_zero_when_no_drops() {
        let s = MetricsSnapshot {
            frames_captured: 100,
            frames_dropped: 0,
        };
        assert_eq!(s.drop_rate(), 0.0);
    }

    #[test]
    fn drop_rate_half() {
        let s = MetricsSnapshot {
            frames_captured: 50,
            frames_dropped: 50,
        };
        assert!(
            (s.drop_rate() - 0.5).abs() < 1e-9,
            "expected 0.5, got {}",
            s.drop_rate()
        );
    }

    #[test]
    fn drop_rate_all_dropped() {
        // Only drops counted (no captured): ratio is 1.0.
        // In practice captured is incremented first, but the math should still work.
        let s = MetricsSnapshot {
            frames_captured: 0,
            frames_dropped: 10,
        };
        assert!((s.drop_rate() - 1.0).abs() < 1e-9);
    }
}
