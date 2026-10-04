//! Soft-label logistic probe math (no llama.cpp).

use crate::error::{Error, Result};
use crate::softmax::softmax;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogisticProbe {
    pub recipe: String,
    pub n_classes: usize,
    pub dim: usize,
    pub weights: Vec<Vec<f64>>, // [class][dim]
    pub bias: Vec<f64>,
    pub l2: f64,
}

impl LogisticProbe {
    pub fn zeros(dim: usize, n_classes: usize, recipe: &str, l2: f64) -> Self {
        Self {
            recipe: recipe.into(),
            n_classes,
            dim,
            weights: vec![vec![0.0; dim]; n_classes],
            bias: vec![0.0; n_classes],
            l2,
        }
    }

    pub fn logits(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.dim {
            return Err(Error::Validation("probe dim mismatch".into()));
        }
        let mut out = self.bias.clone();
        for c in 0..self.n_classes {
            let mut s = self.bias[c];
            for i in 0..self.dim {
                s += self.weights[c][i] * x[i];
            }
            out[c] = s;
        }
        Ok(out)
    }

    pub fn predict(&self, x: &[f64]) -> Result<Vec<f64>> {
        softmax(&self.logits(x)?)
    }
}

/// One epoch of gradient descent on soft labels (teacher probs).
pub fn train_epoch(
    probe: &mut LogisticProbe,
    xs: &[Vec<f64>],
    ys: &[Vec<f64>],
    lr: f64,
) -> Result<f64> {
    if xs.len() != ys.len() || xs.is_empty() {
        return Err(Error::Validation("empty distill set".into()));
    }
    let mut loss = 0.0;
    let n = xs.len() as f64;
    let mut dw = vec![vec![0.0; probe.dim]; probe.n_classes];
    let mut db = vec![0.0; probe.n_classes];
    for (x, y) in xs.iter().zip(ys.iter()) {
        let p = probe.predict(x)?;
        for c in 0..probe.n_classes {
            let g = p[c] - y.get(c).copied().unwrap_or(0.0);
            loss -= y.get(c).copied().unwrap_or(0.0).max(1e-12).ln() * p.get(c).copied().unwrap_or(0.0).max(1e-12).ln().abs().min(20.0);
            db[c] += g;
            for i in 0..probe.dim {
                dw[c][i] += g * x[i];
            }
        }
    }
    let _ = loss / n;
    for c in 0..probe.n_classes {
        probe.bias[c] -= lr * db[c] / n;
        for i in 0..probe.dim {
            let g = dw[c][i] / n + probe.l2 * probe.weights[c][i];
            probe.weights[c][i] -= lr * g;
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separable_two_class() {
        let mut p = LogisticProbe::zeros(2, 2, "probe-decree-v1", 1e-4);
        let xs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.1], vec![0.1, 1.0]];
        let ys = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.9, 0.1],
            vec![0.1, 0.9],
        ];
        for _ in 0..80 {
            train_epoch(&mut p, &xs, &ys, 0.5).unwrap();
        }
        let a = p.predict(&xs[0]).unwrap();
        assert!(a[0] > a[1]);
    }
}
