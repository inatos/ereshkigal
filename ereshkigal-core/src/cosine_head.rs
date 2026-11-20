//! Minimal 0.53M-class cosine readout head (DecideAnything / Kev-class).
//!
//! `RMSNorm_h → W_q, W_o ∈ R^{256×H} → RMSNorm → min(exp(θ),100) * (1/W) qᵀ o`
//!
//! Binary `.ehead` layout (little-endian):
//! - `0..8`   : magic `b"EHEAD01\0"`
//! - `8..12`  : `hidden: u32`
//! - `12..16` : `head_dim: u32` (must match [`HEAD_DIM`] = 256)
//! - `16..20` : `theta: f32`
//! - `20..24` : `reserved: u32` (`0`)
//! - `24..`   : `w_q` (`head_dim * hidden` f32, row-major `[head_dim, hidden]`),
//!              then `w_o` (`head_dim * hidden` f32, row-major `[head_dim, hidden]`).

use crate::error::{Error, Result};
use std::path::Path;

pub const HEAD_DIM: usize = 256;
pub const EHEAD_MAGIC: &[u8; 8] = b"EHEAD01\0";
pub const EHEAD_HEADER_BYTES: usize = 24;

#[derive(Clone, Debug)]
pub struct CosineHead {
    pub w_q: Vec<f32>, // HEAD_DIM * H
    pub w_o: Vec<f32>, // HEAD_DIM * H
    pub hidden: usize,
    pub theta: f32,
}

impl CosineHead {
    pub fn random_init(hidden: usize, seed: u64) -> Self {
        let mut w_q = vec![0f32; HEAD_DIM * hidden];
        let mut w_o = vec![0f32; HEAD_DIM * hidden];
        let mut s = seed.max(1);
        for v in w_q.iter_mut().chain(w_o.iter_mut()) {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((s >> 33) as f32 / u32::MAX as f32) * 0.02 - 0.01;
        }
        Self {
            w_q,
            w_o,
            hidden,
            theta: 0.0,
        }
    }

    pub fn load_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::load_ehead(&bytes)
    }

    pub fn save_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, self.to_ehead())?;
        Ok(())
    }

    pub fn load_ehead(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < EHEAD_HEADER_BYTES {
            return Err(Error::Validation(format!(
                "ehead too short: got {} bytes, need at least {EHEAD_HEADER_BYTES}",
                bytes.len()
            )));
        }
        if &bytes[0..8] != EHEAD_MAGIC {
            return Err(Error::Validation(
                "ehead invalid magic (expected EHEAD01\\0)".into(),
            ));
        }
        let hidden = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let head_dim = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let theta = f32::from_bits(u32::from_le_bytes(bytes[16..20].try_into().unwrap()));
        if hidden == 0 {
            return Err(Error::Validation("ehead hidden must be > 0".into()));
        }
        if head_dim != HEAD_DIM {
            return Err(Error::Validation(format!(
                "ehead head_dim={head_dim} unsupported (expected {HEAD_DIM})"
            )));
        }
        let need = EHEAD_HEADER_BYTES + 2 * HEAD_DIM * hidden * 4;
        if bytes.len() < need {
            return Err(Error::Validation(format!(
                "ehead need {need} bytes got {}",
                bytes.len()
            )));
        }
        let mut off = EHEAD_HEADER_BYTES;
        let mut w_q = vec![0f32; HEAD_DIM * hidden];
        let mut w_o = vec![0f32; HEAD_DIM * hidden];
        for v in w_q.iter_mut().chain(w_o.iter_mut()) {
            *v = f32::from_bits(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        Ok(Self {
            w_q,
            w_o,
            hidden,
            theta,
        })
    }

    pub fn to_ehead(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(EHEAD_HEADER_BYTES + 2 * HEAD_DIM * self.hidden * 4);
        out.extend_from_slice(EHEAD_MAGIC);
        out.extend_from_slice(&(self.hidden as u32).to_le_bytes());
        out.extend_from_slice(&(HEAD_DIM as u32).to_le_bytes());
        out.extend_from_slice(&self.theta.to_bits().to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for v in self.w_q.iter().chain(self.w_o.iter()) {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out
    }

    fn rms_norm(x: &[f32]) -> Vec<f32> {
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32;
        let inv = 1.0 / (mean_sq + 1e-6).sqrt();
        x.iter().map(|v| v * inv).collect()
    }

    fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let mut out = vec![0f32; out_dim];
        for i in 0..out_dim {
            let row = &w[i * in_dim..(i + 1) * in_dim];
            out[i] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        }
        out
    }

    /// Score one option embedding against query hidden state.
    pub fn score_pair(&self, h_query: &[f32], h_option: &[f32]) -> Result<f32> {
        if h_query.len() != self.hidden || h_option.len() != self.hidden {
            return Err(Error::Validation(format!(
                "hidden size mismatch: head expects {}, got query={} option={}",
                self.hidden,
                h_query.len(),
                h_option.len()
            )));
        }
        let hq = Self::rms_norm(h_query);
        let ho = Self::rms_norm(h_option);
        let q = Self::rms_norm(&Self::matvec(&self.w_q, &hq, HEAD_DIM, self.hidden));
        let o = Self::rms_norm(&Self::matvec(&self.w_o, &ho, HEAD_DIM, self.hidden));
        let dot: f32 = q.iter().zip(o.iter()).map(|(a, b)| a * b).sum();
        let scale = (self.theta.exp()).min(100.0) / (HEAD_DIM as f32);
        Ok(scale * dot)
    }

    /// Score multiple option embeddings against one query hidden state.
    pub fn score_options(&self, h_query: &[f32], h_options: &[Vec<f32>]) -> Result<Vec<f32>> {
        h_options
            .iter()
            .map(|ho| self.score_pair(h_query, ho))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_fixture_head() -> CosineHead {
        let hidden = 4usize;
        let mut w_q = Vec::with_capacity(HEAD_DIM * hidden);
        let mut w_o = Vec::with_capacity(HEAD_DIM * hidden);
        for i in 0..(HEAD_DIM * hidden) {
            // Exact dyadic rationals (1/16, 1/32) ensure bit-identical f32 bytes across Rust & Python.
            w_q.push(((i % 13) as f32 - 6.0) * 0.0625);
            w_o.push(((i % 17) as f32 - 8.0) * 0.03125);
        }
        CosineHead {
            w_q,
            w_o,
            hidden,
            theta: 0.5,
        }
    }

    #[test]
    fn ehead_roundtrip() {
        let h = CosineHead::random_init(64, 7);
        let bytes = h.to_ehead();
        assert_eq!(&bytes[0..8], EHEAD_MAGIC);
        let h2 = CosineHead::load_ehead(&bytes).unwrap();
        assert_eq!(h.hidden, h2.hidden);
        assert_eq!(h.theta, h2.theta);
        let s = h.score_pair(&vec![0.1; 64], &vec![0.2; 64]).unwrap();
        let s2 = h2.score_pair(&vec![0.1; 64], &vec![0.2; 64]).unwrap();
        assert!((s - s2).abs() < 1e-6);
    }

    #[test]
    fn ehead_rejects_bad_magic_and_short_buffer() {
        assert!(CosineHead::load_ehead(&[0u8; 8]).is_err());
        let mut bytes = CosineHead::random_init(4, 1).to_ehead();
        bytes[0] = b'X';
        let err = CosineHead::load_ehead(&bytes).unwrap_err().to_string();
        assert!(err.contains("EHEAD01"), "{err}");
    }

    #[test]
    fn ehead_deterministic_fixture_for_python_parity() {
        use sha2::{Digest, Sha256};

        let head = deterministic_fixture_head();
        let bytes = head.to_ehead();
        assert_eq!(bytes.len(), EHEAD_HEADER_BYTES + 2 * HEAD_DIM * 4 * 4);
        // Must match tools/ereshkigal/training/test_train_lora_split.py byte-for-byte.
        let sha = hex::encode(Sha256::digest(&bytes));
        assert_eq!(
            sha,
            "003875af19c9748f51c6c2fe8d0c9e4b1ec8e269e142b62d9a09fb2baab0bec1"
        );
        let loaded = CosineHead::load_ehead(&bytes).unwrap();
        let h_q = [0.25f32, -0.5, 0.75, 1.0];
        let h_opts = vec![
            vec![0.5f32, 0.25, -0.25, 0.75],
            vec![-0.5f32, 0.75, 0.25, -0.25],
        ];
        let scores = loaded.score_options(&h_q, &h_opts).unwrap();
        assert_eq!(scores.len(), 2);
        // Golden PyTorch CosineHeadModule scores (max|diff| < 1e-4 parity gate).
        let pytorch_golden = [-0.00168752_f32, 0.00767222_f32];
        for (got, want) in scores.iter().zip(pytorch_golden.iter()) {
            assert!(
                (got - want).abs() < 1e-4,
                "Rust vs PyTorch cosine parity diff >= 1e-4: got={got} want={want}"
            );
        }
    }
}
