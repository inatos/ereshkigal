use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const LETTERS: &str = "ABCDEFGHIJKLMNOP";
pub const PROMPT_VERSION: &str = "direct-options-v1";
pub const DIRECT_SYSTEM: &str = concat!(
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. ",
    "Respond with only its uppercase letter, with no explanation or reasoning."
);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionSpec {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRow {
    pub id: String,
    pub state: Value,
    pub question: String,
    pub options: Vec<OptionSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub source: String,
    pub revision: String,
    pub backend: String,
    pub dtype: String,
    pub gguf: GgufMeta,
    pub vocab_size: i32,
    pub threads: i32,
    pub n_gpu_layers: i32,
    pub max_prompt_tokens: usize,
    pub context_tokens: u32,
    pub decode_chunk: usize,
    pub serving_config: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufMeta {
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreResult {
    pub id: String,
    pub option_ids: Vec<String>,
    pub probabilities: Vec<f64>,
    pub option_logits: Vec<f64>,
    pub answer_token_ids: Vec<i32>,
    pub input_tokens: usize,
    pub prompt_sha256: String,
    pub prompt_version: String,
    pub model: ModelMeta,
    pub readout: String,
    pub probability_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forward_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix_forward_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub copy_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_token_mass: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_vocab_argmax_id: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct EncodedDecision {
    pub token_ids: Vec<i32>,
    pub slots: Vec<i32>,
    pub prompt: String,
    pub prompt_sha256: String,
}
