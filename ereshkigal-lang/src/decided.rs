//! Decision result types.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecideStatus {
    Chosen,
    Abstain,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Decided {
    pub id: String,
    pub decree: String,
    pub choice: Option<String>,
    pub choice_index: Option<usize>,
    pub probabilities: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibrated_probabilities: Option<Vec<f64>>,
    pub option_ids: Vec<String>,
    pub margin: f64,
    pub status: DecideStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conformal_set: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_cost_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path_probability: Option<f64>,
    pub prompt_sha256: String,
    pub backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<bool>,
}

impl Decided {
    pub fn choice_as<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        let c = self
            .choice
            .as_ref()
            .ok_or_else(|| Error::Validation("abstained; no choice".into()))?;
        serde_json::from_value(serde_json::Value::String(c.clone())).map_err(Error::from)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProgramResult {
    pub program: String,
    pub value: String,
    pub nodes: Vec<Decided>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path_probability: Option<f64>,
    #[serde(default)]
    pub skipped: Vec<String>,
    pub forwards: usize,
}
