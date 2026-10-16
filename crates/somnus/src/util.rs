//! Small helpers shared by the generation paths.

/// Index of the largest logit (first of equals is not guaranteed); `0` for an empty slice.
/// NaN compares as equal to everything, so a NaN logit never panics the decoder.
pub(crate) fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::argmax;

    #[test]
    fn argmax_picks_strict_maximum() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
    }

    #[test]
    fn argmax_empty_is_zero() {
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn argmax_tie_returns_some_max_index() {
        let logits = [1.0f32, 2.0, 2.0, 0.5];
        let i = argmax(&logits);
        assert_eq!(logits[i as usize], 2.0, "tie broke to a non-max index {i}");
    }
}
