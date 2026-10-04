use crate::error::{Error, Result};

pub fn softmax(values: &[f64]) -> Result<Vec<f64>> {
    if values.len() < 2 || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Validation(
            "Need at least two finite scores".into(),
        ));
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = values.iter().map(|v| (v - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    Ok(weights.into_iter().map(|w| w / total).collect())
}

pub fn argmax(values: &[f64]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_uniform_on_equal() {
        let p = softmax(&[1.0, 1.0, 1.0]).unwrap();
        assert!(p.iter().all(|x| (x - 1.0 / 3.0).abs() < 1e-12));
    }

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.2]), Some(1));
    }
}
