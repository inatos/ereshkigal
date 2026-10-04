//! Lockfile pinning recipe, hashes, fitted parameters, metrics snapshot.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Lockfile {
    pub recipe: String,
    #[serde(default)]
    pub template_hash: String,
    #[serde(default)]
    pub gguf_sha256: String,
    #[serde(default)]
    pub adapter_sha256: Option<String>,
    #[serde(default)]
    pub tokenizer_revision: String,
    #[serde(default)]
    pub prompt_hashes: BTreeMap<String, String>,
    #[serde(default)]
    pub fitted: FittedParams,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub packages: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FittedParams {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub abstain_margin: BTreeMap<String, f64>,
    #[serde(default)]
    pub conformal_qhat: BTreeMap<String, f64>,
    #[serde(default)]
    pub letter_prior: Option<Vec<f64>>,
    #[serde(default)]
    pub node_pass_rate: BTreeMap<String, f64>,
}

impl Lockfile {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        serde_json::from_str(&s).map_err(Error::from)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let s = serde_json::to_string_pretty(self)?;
        std::fs::write(path, s)?;
        Ok(())
    }

    pub fn drift(&self, other: &Lockfile) -> Vec<String> {
        let mut d = Vec::new();
        if self.recipe != other.recipe {
            d.push(format!("recipe {} -> {}", self.recipe, other.recipe));
        }
        if self.gguf_sha256 != other.gguf_sha256 && !other.gguf_sha256.is_empty() {
            d.push("gguf sha256 drifted".into());
        }
        if self.template_hash != other.template_hash && !other.template_hash.is_empty() {
            d.push("template hash drifted".into());
        }
        for (k, v) in &self.prompt_hashes {
            if other.prompt_hashes.get(k) != Some(v) {
                d.push(format!("prompt hash {k} drifted"));
            }
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_detects_recipe() {
        let a = Lockfile {
            recipe: "a".into(),
            ..Default::default()
        };
        let b = Lockfile {
            recipe: "b".into(),
            ..Default::default()
        };
        assert!(!a.drift(&b).is_empty());
    }
}
