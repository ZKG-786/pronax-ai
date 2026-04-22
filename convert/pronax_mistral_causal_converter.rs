use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::convert::pronax_converter_core::{
    ConversionCoordinate, NeuralMetadataKV, NeuralModelConverter, NeuralSourceTensor,
};
use crate::fs::ggml::pronax_ggml_format::NeuralGgmlTensor;
use crate::tokenizer::pronax_vocabulary::{NeuralConversionTokenizer, SpecialTokenType};

/// 3D Spatial coordinate for Mistral3 Causal conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MistralCausalCoordinate {
    pub sequence_id: u64,
    pub attention_tier: u16,
    pub causal_depth: u8,
    pub scaling_factor: f32,
}

impl MistralCausalCoordinate {
    pub const fn new(seq: u64, tier: u16, depth: u8, factor: f32) -> Self {
        Self {
            sequence_id: seq,
            attention_tier: tier,
            causal_depth: depth,
            scaling_factor: factor,
        }
    }

    pub const fn causal_attention() -> Self {
        Self::new(0, 980, 24, 1.0)
    }

    pub const fn sliding_window() -> Self {
        Self::new(0, 960, 20, 0.998)
    }

    pub const fn rope_scaled() -> Self {
        Self::new(0, 940, 18, 0.995)
    }

    pub const fn ffn_transformer() -> Self {
        Self::new(0, 920, 16, 0.992)
    }

    pub const fn output_projection() -> Self {
        Self::new(0, 900, 14, 0.99)
    }

    /// Calculate importance score based on causal depth
    pub fn importance_score(&self) -> u64 {
        let seq_factor = 1000u64.saturating_sub(self.sequence_id);
        let tier_boost = self.attention_tier as u64 * 95;
        let depth_norm = self.causal_depth as u64 * 10;
        let scaling_boost = (self.scaling_factor * 1500.0) as u64;

        seq_factor + tier_boost + depth_norm + scaling_boost
    }

    /// Check if this coordinate represents sliding window attention
    pub fn is_sliding_window(&self) -> bool {
        self.attention_tier == 960
    }
}

/// RoPE scaling configuration for causal models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalRoPEScalingType {
    Yarn,
    Llama3,
    Linear,
    None,
}

impl CausalRoPEScalingType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "yarn" => Self::Yarn,
            "llama3" => Self::Llama3,
            "linear" => Self::Linear,
            _ => Self::None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Yarn => "yarn",
            Self::Llama3 => "llama3",
            Self::Linear => "linear",
            Self::None => "none",
        }
    }

    pub fn is_active(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Get default beta values for scaling type
    pub fn default_betas(&self) -> (f32, f32) {
        match self {
            Self::Yarn => (32.0, 1.0),
            Self::Llama3 => (32.0, 1.0),
            Self::Linear => (1.0, 1.0),
            Self::None => (1.0, 1.0),
        }
    }
}

/// Advanced RoPE parameters for causal decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalRoPEParameters {
    #[serde(rename = "beta_fast")]
    pub beta_fast: f32,
    #[serde(rename = "beta_slow")]
    pub beta_slow: f32,
    #[serde(rename = "factor")]
    pub factor: f32,
    #[serde(rename = "llama_4_scaling_beta")]
    pub llama4_scaling_beta: Option<f32>,
    #[serde(rename = "original_max_position_embeddings")]
    pub original_max_pos_emb: u32,
    #[serde(rename = "rope_type")]
    pub rope_type: String,
    #[serde(rename = "rope_theta")]
    pub rope_theta: f32,
    #[serde(rename = "mscale")]
    pub mscale: Option<f32>,
    #[serde(rename = "mscale_all_dim")]
    pub mscale_all_dim: Option<f32>,
}

impl CausalRoPEParameters {
    pub fn new() -> Self {
        Self {
            beta_fast: 32.0,
            beta_slow: 1.0,
            factor: 1.0,
            llama4_scaling_beta: None,
            original_max_pos_emb: 8192,
            rope_type: String::new(),
            rope_theta: 10000.0,
            mscale: None,
            mscale_all_dim: None,
        }
    }

    pub fn with_scaling_type(mut self, scaling_type: CausalRoPEScalingType) -> Self {
        let (beta_fast, beta_slow) = scaling_type.default_betas();
        self.rope_type = scaling_type.as_str().to_string();
        self.beta_fast = beta_fast;
        self.beta_slow = beta_slow;
        self
    }

    pub fn get_scaling_type(&self) -> CausalRoPEScalingType {
        CausalRoPEScalingType::from_str(&self.rope_type)
    }

    pub fn has_advanced_scaling(&self) -> bool {
        self.factor != 1.0 ||
        self.beta_fast != 32.0 ||
        self.beta_slow != 1.0 ||
        self.llama4_scaling_beta.is_some() ||
        self.mscale.is_some() ||
        self.mscale_all_dim.is_some()
    }

    /// Calculate effective context length after scaling
    pub fn effective_context_length(&self, base_context: u32) -> u32 {
        if self.factor > 1.0 {
            (base_context as f32 * self.factor) as u32
        } else {
            base_context
        }
    }
}

impl Default for CausalRoPEParameters {
    fn default() -> Self {
        Self::new()
    }
}

/// Causal language model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct NeuralMistralCausalConfig {
    #[serde(rename = "num_hidden_layers")]
    pub num_hidden_layers: u32,
    #[serde(rename = "max_position_embeddings")]
    pub max_position_embeddings: u32,
    #[serde(rename = "hidden_size")]
    pub hidden_size: u32,
    #[serde(rename = "intermediate_size")]
    pub intermediate_size: u32,
    #[serde(rename = "num_attention_heads")]
    pub num_attention_heads: u32,
    #[serde(rename = "num_key_value_heads")]
    pub num_key_value_heads: u32,
    #[serde(rename = "rope_theta")]
    pub rope_theta: f32,
    #[serde(rename = "rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(rename = "head_dim")]
    pub head_dim: u32,
    #[serde(rename = "sliding_window")]
    pub sliding_window: Option<u32>,
    #[serde(rename = "hidden_act")]
    pub hidden_act: String,
    #[serde(rename = "vocab_size")]
    pub vocab_size: u32,
    #[serde(rename = "rope_parameters")]
    pub rope_params: CausalRoPEParameters,
}

impl NeuralMistralCausalConfig {
    pub fn new() -> Self {
        Self {
            num_hidden_layers: 32,
            max_position_embeddings: 32768,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            head_dim: 128,
            sliding_window: Some(4096),
            hidden_act: "silu".to_string(),
            vocab_size: 32768,
            rope_params: CausalRoPEParameters::default(),
        }
    }

    /// Create configuration for 7B model
    pub fn mistral_7b() -> Self {
        Self::new()
    }

    /// Create configuration for smaller variant
    pub fn mistral_small() -> Self {
        let mut config = Self::new();
        config.num_hidden_layers = 24;
        config.hidden_size = 2048;
        config.intermediate_size = 7168;
        config.num_attention_heads = 16;
        config.num_key_value_heads = 4;
        config.head_dim = 128;
        config
    }

    /// Check if using Grouped Query Attention
    pub fn uses_gqa(&self) -> bool {
        self.num_key_value_heads > 0 && self.num_key_value_heads != self.num_attention_heads
    }

    /// Get GQA group size
    pub fn gqa_group_size(&self) -> u32 {
        if self.uses_gqa() {
            self.num_attention_heads / self.num_key_value_heads
        } else {
            1
        }
    }

    /// Check if sliding window attention is enabled
    pub fn has_sliding_window(&self) -> bool {
        self.sliding_window.is_some()
    }

    /// Get sliding window size
    pub fn sliding_window_size(&self) -> u32 {
        self.sliding_window.unwrap_or(0)
    }

    /// Calculate effective head dimension
    pub fn effective_head_dim(&self) -> u32 {
        if self.head_dim > 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_attention_heads
        }
    }

    /// Calculate rope dimension count
    pub fn rope_dimension_count(&self) -> u32 {
        self.effective_head_dim()
    }

    /// Get effective rope theta
    pub fn effective_rope_theta(&self) -> f32 {
        if self.rope_params.rope_theta > 0.0 {
            self.rope_params.rope_theta
        } else {
            self.rope_theta
        }
    }

    /// Calculate total parameter count
    pub fn estimate_parameters(&self) -> u64 {
        let vocab_size = self.vocab_size as u64;
        let hidden = self.hidden_size as u64;
        let intermediate = self.intermediate_size as u64;
        let layers = self.num_hidden_layers as u64;
        let heads = self.num_attention_heads as u64;
        let kv_heads = self.num_key_value_heads as u64;
        let head_dim = self.effective_head_dim() as u64;

        // Embedding
        let token_embd = vocab_size * hidden;

        // Attention per layer
        let q_proj = hidden * (heads * head_dim);
        let k_proj = hidden * (kv_heads * head_dim);
        let v_proj = k_proj;
        let o_proj = heads * head_dim * hidden;

        // FFN per layer
        let gate_proj = hidden * intermediate;
        let up_proj = gate_proj;
        let down_proj = intermediate * hidden;

        // Norms per layer
        let attn_norm = hidden;
        let ffn_norm = hidden;

        let per_layer = q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj + attn_norm + ffn_norm;
        let all_layers = layers * per_layer;

        // Output norm and lm_head
        let output_norm = hidden;
        let lm_head = vocab_size * hidden;

        token_embd + all_layers + output_norm + lm_head
    }
}

impl Default for NeuralMistralCausalConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Mistral3 Causal model parameters
#[derive(Debug, Clone, Deserialize)]
pub struct NeuralMistralCausalParameters {
    #[serde(flatten)]
    pub config: NeuralMistralCausalConfig,
}

impl NeuralMistralCausalParameters {
    pub fn new() -> Self {
        Self {
            config: NeuralMistralCausalConfig::new(),
        }
    }

    pub fn with_config(config: NeuralMistralCausalConfig) -> Self {
        Self { config }
    }

    /// Check if model uses advanced rope scaling
    pub fn uses_advanced_rope(&self) -> bool {
        self.config.rope_params.has_advanced_scaling()
    }

    /// Get effective context length considering rope scaling
    pub fn effective_context_length(&self) -> u32 {
        self.config.rope_params.effective_context_length(self.config.max_position_embeddings)
    }
}

impl Default for NeuralMistralCausalParameters {
    fn default() -> Self {
        Self::new()
    }
}

/// Attention tensor repacker for interleaved heads
pub struct CausalAttentionRepacker;

impl CausalAttentionRepacker {
    /// Repack attention weights with interleaved → grouped transformation
    /// 
    /// The transformation: [out, in] where out = heads * 2 * (head_dim/2)
    /// is reshaped to [heads, 2, head_dim/2, in] then permuted to [heads, head_dim/2, 2, in]
    pub fn repack_weights(
        data: &[f32],
        shape: &[u64],
        num_heads: u32,
    ) -> Result<Vec<f32>, String> {
        if shape.len() < 2 {
            return Err("Invalid shape: expected at least 2 dimensions".to_string());
        }

        let heads = num_heads as usize;
        let dim0 = shape[0] as usize;
        let dim1 = shape[1] as usize;

        // Calculate inner dimension: dim0 = heads * 2 * inner_dim
        let inner_dim = dim0 / heads / 2;
        if inner_dim == 0 || dim0 % (heads * 2) != 0 {
            return Ok(data.to_vec());
        }

        let mut result = vec![0.0f32; data.len()];

        // Transpose: [heads, 2, inner_dim, dim1] -> [heads, inner_dim, 2, dim1]
        for h in 0..heads {
            for i in 0..inner_dim {
                for j in 0..2 {
                    for k in 0..dim1 {
                        // Source index: [h, j, i, k]
                        let src_idx = h * 2 * inner_dim * dim1 + j * inner_dim * dim1 + i * dim1 + k;
                        // Target index: [h, i, j, k]
                        let dst_idx = h * inner_dim * 2 * dim1 + i * 2 * dim1 + j * dim1 + k;

                        if src_idx < data.len() && dst_idx < result.len() {
                            result[dst_idx] = data[src_idx];
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Check if tensor name indicates it needs repacking
    pub fn needs_repack(name: &str) -> bool {
        name.ends_with(".attn_q.weight") || name.ends_with(".attn_k.weight")
    }

    /// Determine number of heads for repacking based on tensor name
    pub fn get_head_count(name: &str, num_q_heads: u32, num_kv_heads: u32) -> u32 {
        if name.ends_with(".attn_q.weight") {
            num_q_heads
        } else if name.ends_with(".attn_k.weight") {
            if num_kv_heads > 0 {
                num_kv_heads
            } else {
                num_q_heads
            }
        } else {
            num_q_heads
        }
    }

    /// Validate tensor shape for repacking
    pub fn validate_shape(shape: &[u64], num_heads: u32) -> bool {
        if shape.len() < 2 {
            return false;
        }
        let dim0 = shape[0] as usize;
        let heads = num_heads as usize;
        dim0 >= heads * 2 && dim0 % (heads * 2) == 0
    }
}

/// Sliding window attention handler
#[derive(Debug, Clone)]
pub struct SlidingWindowHandler {
    pub window_size: u32,
    pub is_enabled: bool,
    pub layer_pattern: SlidingWindowPattern,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlidingWindowPattern {
    AllLayers,      // Every layer uses sliding window
    Alternating,    // Every other layer
    FirstHalf,      // First half of layers
    LastHalf,       // Last half of layers
    None,
}

impl SlidingWindowHandler {
    pub fn new(window_size: Option<u32>) -> Self {
        let enabled = window_size.is_some() && window_size.unwrap() > 0;
        Self {
            window_size: window_size.unwrap_or(0),
            is_enabled: enabled,
            layer_pattern: SlidingWindowPattern::AllLayers,
        }
    }

    pub fn with_pattern(mut self, pattern: SlidingWindowPattern) -> Self {
        self.layer_pattern = pattern;
        self
    }

    /// Check if specific layer should use sliding window
    pub fn use_sliding_for_layer(&self, layer_idx: u32, total_layers: u32) -> bool {
        if !self.is_enabled {
            return false;
        }

        match self.layer_pattern {
            SlidingWindowPattern::AllLayers => true,
            SlidingWindowPattern::Alternating => layer_idx % 2 == 0,
            SlidingWindowPattern::FirstHalf => layer_idx < total_layers / 2,
            SlidingWindowPattern::LastHalf => layer_idx >= total_layers / 2,
            SlidingWindowPattern::None => false,
        }
    }

    /// Get coordinate for sliding window tensors
    pub fn get_coordinate(&self) -> MistralCausalCoordinate {
        if self.is_enabled {
            MistralCausalCoordinate::sliding_window()
        } else {
            MistralCausalCoordinate::causal_attention()
        }
    }
}

/// Mistral3 Causal converter
#[derive(Debug, Clone)]
pub struct NeuralMistralCausalConverter {
    params: NeuralMistralCausalParameters,
    coordinate: MistralCausalCoordinate,
    tensor_replacements: HashMap<String, String>,
    sliding_window: SlidingWindowHandler,
}

impl NeuralMistralCausalConverter {
    /// Create new converter
    pub fn new(params: NeuralMistralCausalParameters) -> Self {
        let sliding_window = SlidingWindowHandler::new(params.config.sliding_window);
        let mut converter = Self {
            params,
            coordinate: MistralCausalCoordinate::causal_attention(),
            tensor_replacements: HashMap::new(),
            sliding_window,
        };

        converter.initialize_replacements();
        converter
    }

    /// Initialize tensor name replacements
    fn initialize_replacements(&mut self) {
        // Model prefix removal
        self.tensor_replacements.insert("model.norm".to_string(), "output_norm".to_string());
        self.tensor_replacements.insert("model.".to_string(), "".to_string());

        // Layer naming
        self.tensor_replacements.insert("layers".to_string(), "blk".to_string());
        self.tensor_replacements.insert("transformer.layers".to_string(), "blk".to_string());

        // Pre-layer norm
        self.tensor_replacements.insert("ln_pre".to_string(), "encoder_norm".to_string());

        // Attention norms
        self.tensor_replacements.insert("input_layernorm".to_string(), "attn_norm".to_string());
        self.tensor_replacements.insert("post_attention_layernorm".to_string(), "ffn_norm".to_string());

        // Embeddings
        self.tensor_replacements.insert("embed_tokens".to_string(), "token_embd".to_string());

        // Self attention
        self.tensor_replacements.insert("self_attn.q_proj".to_string(), "attn_q".to_string());
        self.tensor_replacements.insert("self_attn.k_proj".to_string(), "attn_k".to_string());
        self.tensor_replacements.insert("self_attn.v_proj".to_string(), "attn_v".to_string());
        self.tensor_replacements.insert("self_attn.o_proj".to_string(), "attn_output".to_string());

        // MLP
        self.tensor_replacements.insert("mlp.down_proj".to_string(), "ffn_down".to_string());
        self.tensor_replacements.insert("mlp.gate_proj".to_string(), "ffn_gate".to_string());
        self.tensor_replacements.insert("mlp.up_proj".to_string(), "ffn_up".to_string());

        // Alternative attention naming (for compatibility)
        self.tensor_replacements.insert("attention.q_proj".to_string(), "attn_q".to_string());
        self.tensor_replacements.insert("attention.k_proj".to_string(), "attn_k".to_string());
        self.tensor_replacements.insert("attention.v_proj".to_string(), "attn_v".to_string());
        self.tensor_replacements.insert("attention.o_proj".to_string(), "attn_output".to_string());
        self.tensor_replacements.insert("attention_norm".to_string(), "attn_norm".to_string());

        // Alternative FFN naming
        self.tensor_replacements.insert("feed_forward.gate_proj".to_string(), "ffn_gate".to_string());
        self.tensor_replacements.insert("feed_forward.down_proj".to_string(), "ffn_down".to_string());
        self.tensor_replacements.insert("feed_forward.up_proj".to_string(), "ffn_up".to_string());

        // Multi-modal projector (for compatibility with vision variants)
        self.tensor_replacements.insert("multi_modal_projector".to_string(), "mm".to_string());

        // Output head
        self.tensor_replacements.insert("lm_head".to_string(), "output".to_string());
    }

    /// Apply tensor name replacement
    pub fn replace_tensor_name(&self, name: &str) -> String {
        let mut result = name.to_string();
        for (from, to) in &self.tensor_replacements {
            result = result.replace(from, to);
        }
        result
    }

    /// Get all tensor replacements
    pub fn replacement_pairs(&self) -> Vec<(String, String)> {
        self.tensor_replacements
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Get coordinate for tensor based on name
    fn coordinate_for_tensor(&self, name: &str) -> MistralCausalCoordinate {
        if name.contains("attn_q") || name.contains("attn_k") || name.contains("attn_v") {
            if self.sliding_window.is_enabled && name.contains("sliding") {
                MistralCausalCoordinate::sliding_window()
            } else {
                MistralCausalCoordinate::causal_attention()
            }
        } else if name.contains("rope") || name.contains("scaling") {
            MistralCausalCoordinate::rope_scaled()
        } else if name.contains("ffn") || name.contains("mlp") {
            MistralCausalCoordinate::ffn_transformer()
        } else if name.contains("output") || name.contains("lm_head") {
            MistralCausalCoordinate::output_projection()
        } else {
            self.coordinate
        }
    }

    /// Get parameter info
    pub fn parameter_info(&self) -> MistralCausalParameterInfo {
        let config = &self.params.config;
        MistralCausalParameterInfo {
            num_layers: config.num_hidden_layers,
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.effective_head_dim(),
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            max_context_length: config.max_position_embeddings,
            effective_context_length: self.params.effective_context_length(),
            sliding_window: config.sliding_window,
            uses_gqa: config.uses_gqa(),
            gqa_group_size: config.gqa_group_size(),
            rope_scaling_type: config.rope_params.get_scaling_type(),
            rope_factor: config.rope_params.factor,
            rope_theta: config.effective_rope_theta(),
            uses_advanced_rope: self.params.uses_advanced_rope(),
            estimated_parameters: config.estimate_parameters(),
            coordinate: self.coordinate,
        }
    }

    /// Generate model summary
    pub fn model_summary(&self) -> String {
        let info = self.parameter_info();
        let params_b = info.estimated_parameters as f64 / 1e9;

        format!(
            "Mistral3-Causal {} layers, {} hidden, {} heads (GQA:{}), {} context, {:.1}B params, RoPE:{}",
            info.num_layers,
            info.hidden_size,
            info.num_attention_heads,
            info.gqa_group_size,
            info.effective_context_length,
            params_b,
            info.rope_scaling_type.as_str()
        )
    }
}

/// Parameter info for Mistral3 Causal
#[derive(Debug, Clone)]
pub struct MistralCausalParameterInfo {
    pub num_layers: u32,
    pub hidden_size: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub max_context_length: u32,
    pub effective_context_length: u32,
    pub sliding_window: Option<u32>,
    pub uses_gqa: bool,
    pub gqa_group_size: u32,
    pub rope_scaling_type: CausalRoPEScalingType,
    pub rope_factor: f32,
    pub rope_theta: f32,
    pub uses_advanced_rope: bool,
    pub estimated_parameters: u64,
    pub coordinate: MistralCausalCoordinate,
}

impl NeuralModelConverter for NeuralMistralCausalConverter {
    fn to_metadata_kv(&self, tokenizer: &NeuralConversionTokenizer) -> NeuralMetadataKV {
        let mut kv = NeuralMetadataKV::new();
        let config = &self.params.config;
        let rope = &config.rope_params;

        // Base architecture
        kv.insert("general.architecture", "mistral3".to_string());
        kv.insert("general.type", "causal_lm".to_string());

        // Vocabulary
        kv.insert("mistral3.vocab_size", config.vocab_size);

        // Model dimensions
        kv.insert("mistral3.block_count", config.num_hidden_layers);
        kv.insert("mistral3.context_length", config.max_position_embeddings);
        kv.insert("mistral3.embedding_length", config.hidden_size);
        kv.insert("mistral3.feed_forward_length", config.intermediate_size);

        // Attention configuration
        kv.insert("mistral3.attention.head_count", config.num_attention_heads);
        kv.insert("mistral3.attention.head_count_kv", config.num_kv_heads);
        kv.insert("mistral3.attention.key_length", config.effective_head_dim());
        kv.insert("mistral3.attention.value_length", config.effective_head_dim());
        kv.insert("mistral3.attention.layer_norm_rms_epsilon", config.rms_norm_eps);

        // Sliding window
        if let Some(window) = config.sliding_window {
            kv.insert("mistral3.attention.sliding_window", window);
        }

        // RoPE configuration
        kv.insert("mistral3.rope.dimension_count", config.rope_dimension_count());
        kv.insert("mistral3.rope.freq_base", config.effective_rope_theta());

        // RoPE scaling
        if rope.get_scaling_type().is_active() {
            kv.insert("mistral3.rope.scaling.type", rope.rope_type.clone());
            kv.insert("mistral3.rope.scaling.factor", rope.factor);
            kv.insert("mistral3.rope.scaling.beta_fast", rope.beta_fast);
            kv.insert("mistral3.rope.scaling.beta_slow", rope.beta_slow);

            if let Some(mscale) = rope.mscale {
                kv.insert("mistral3.rope.scaling.mscale", mscale);
            }
            if let Some(mscale_all) = rope.mscale_all_dim {
                kv.insert("mistral3.rope.scaling.mscale_all_dim", mscale_all);
            }
            if rope.original_max_pos_emb > 0 {
                kv.insert("mistral3.rope.scaling.original_context_length", rope.original_max_pos_emb);
            }
            if let Some(beta) = rope.llama4_scaling_beta {
                kv.insert("mistral3.rope.scaling_beta", beta);
            }
        }

        // GQA indicator
        if config.uses_gqa() {
            kv.insert("mistral3.attention.use_gqa", true);
            kv.insert("mistral3.attention.gqa_group_size", config.gqa_group_size());
        }

        // Coordinate metadata
        kv.insert("pronax.coordinate.sequence", self.coordinate.sequence_id);
        kv.insert("pronax.coordinate.tier", self.coordinate.attention_tier);
        kv.insert("pronax.coordinate.depth", self.coordinate.causal_depth);
        kv.insert("pronax.coordinate.scaling", self.coordinate.scaling_factor);

        // Tokenizer metadata
        let tokenizer_kv = tokenizer.to_kv();
        kv.merge(tokenizer_kv);

        kv.set_architecture("mistral3_causal");
        kv
    }

    fn convert_tensors(&self, tensors: &[NeuralSourceTensor]) -> Vec<NeuralGgmlTensor> {
        let mut converted = Vec::with_capacity(tensors.len());
        let config = &self.params.config;

        for (idx, tensor) in tensors.iter().enumerate() {
            let name = self.replace_tensor_name(&tensor.name);

            // Apply attention repacking for Q/K weights
            let mut data = tensor.data.clone();
            if CausalAttentionRepacker::needs_repack(&name) {
                let head_count = CausalAttentionRepacker::get_head_count(
                    &name,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                );

                // Only repack if shape is valid
                if CausalAttentionRepacker::validate_shape(&tensor.shape, head_count) {
                    let float_data = Self::bytes_to_f32_slice(&data);
                    if let Ok(repacked) =
                        CausalAttentionRepacker::repack_weights(&float_data, &tensor.shape, head_count)
                    {
                        data = Self::f32_slice_to_bytes(&repacked);
                    }
                }
            }

            // Get coordinate for this tensor
            let coord = self.coordinate_for_tensor(&name);

            let converted_tensor = NeuralGgmlTensor::new(name, tensor.data_type, tensor.shape.clone(), data)
                .with_coordinate(ConversionCoordinate::new(
                    idx as u64,
                    coord.attention_tier,
                    coord.causal_depth,
                    coord.scaling_factor,
                ));

            converted.push(converted_tensor);
        }

        converted
    }

    fn name_replacements(&self) -> Vec<(String, String)> {
        self.replacement_pairs()
    }

    fn special_token_types(&self) -> Vec<SpecialTokenType> {
        vec![
            SpecialTokenType::Bos,
            SpecialTokenType::Eos,
            SpecialTokenType::Pad,
            SpecialTokenType::Unknown,
        ]
    }

    fn architecture(&self) -> &str {
        "mistral3_causal"
    }

    fn coordinate(&self) -> ConversionCoordinate {
        ConversionCoordinate::new(
            self.coordinate.sequence_id,
            self.coordinate.attention_tier,
            self.coordinate.causal_depth,
            self.coordinate.scaling_factor,
        )
    }
}

impl NeuralMistralCausalConverter {
    /// Helper: Convert bytes to f32 slice
    fn bytes_to_f32_slice(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// Helper: Convert f32 slice to bytes
    fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|&f| f.to_le_bytes().to_vec()).collect()
    }
}

/// Factory function
pub fn create_mistral_causal_converter(params: NeuralMistralCausalParameters) -> NeuralMistralCausalConverter {
    NeuralMistralCausalConverter::new(params)
}

/// Create default 7B model converter
pub fn create_mistral_7b_converter() -> NeuralMistralCausalConverter {
    NeuralMistralCausalConverter::new(NeuralMistralCausalParameters::new())
}

/// Create small variant converter
pub fn create_mistral_small_converter() -> NeuralMistralCausalConverter {
    let params = NeuralMistralCausalParameters::with_config(NeuralMistralCausalConfig::mistral_small());
    NeuralMistralCausalConverter::new(params)
}

/// Create with Yarn scaling
pub fn create_mistral_with_yarn(scale_factor: f32) -> NeuralMistralCausalConverter {
    let mut params = NeuralMistralCausalParameters::new();
    params.config.rope_params = CausalRoPEParameters::new()
        .with_scaling_type(CausalRoPEScalingType::Yarn);
    params.config.rope_params.factor = scale_factor;
    NeuralMistralCausalConverter::new(params)
}

/// Calculate attention memory requirements
pub fn calculate_attention_memory(
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    num_kv_heads: u32,
) -> u64 {
    let q_size = batch_size as u64 * seq_len as u64 * num_heads as u64 * head_dim as u64 * 4;
    let k_size = batch_size as u64 * seq_len as u64 * num_kv_heads as u64 * head_dim as u64 * 4;
    let v_size = k_size;
    let attn_weights = batch_size as u64 * num_heads as u64 * seq_len as u64 * seq_len as u64 * 4;

    q_size + k_size + v_size + attn_weights
}

/// Estimate KV cache size
pub fn estimate_kv_cache_size(
    num_layers: u32,
    batch_size: u32,
    max_seq_len: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> u64 {
    let per_layer = batch_size as u64 * max_seq_len as u64 * num_kv_heads as u64 * head_dim as u64 * 4 * 2; // K + V
    num_layers as u64 * per_layer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_coordinate() {
        let coord = MistralCausalCoordinate::causal_attention();
        assert_eq!(coord.attention_tier, 980);
        assert_eq!(coord.causal_depth, 24);

        let score = coord.importance_score();
        assert!(score > 0);
    }

    #[test]
    fn test_sliding_window_coordinate() {
        let coord = MistralCausalCoordinate::sliding_window();
        assert!(coord.is_sliding_window());
        assert_eq!(coord.attention_tier, 960);
    }

    #[test]
    fn test_rope_scaling_type() {
        assert!(CausalRoPEScalingType::Yarn.is_active());
        assert!(CausalRoPEScalingType::Llama3.is_active());
        assert!(!CausalRoPEScalingType::None.is_active());

        let (beta_fast, beta_slow) = CausalRoPEScalingType::Yarn.default_betas();
        assert_eq!(beta_fast, 32.0);
        assert_eq!(beta_slow, 1.0);
    }

    #[test]
    fn test_rope_params() {
        let params = CausalRoPEParameters::new();
        assert!(!params.has_advanced_scaling());

        let yarn_params = CausalRoPEParameters::new()
            .with_scaling_type(CausalRoPEScalingType::Yarn);
        assert!(yarn_params.get_scaling_type().is_active());
        assert_eq!(yarn_params.rope_type, "yarn");
    }

    #[test]
    fn test_effective_context_length() {
        let mut params = CausalRoPEParameters::new();
        params.factor = 4.0;
        assert_eq!(params.effective_context_length(8192), 32768);
    }

    #[test]
    fn test_causal_config() {
        let config = NeuralMistralCausalConfig::new();
        assert!(config.uses_gqa());
        assert!(config.has_sliding_window());
        assert_eq!(config.gqa_group_size(), 4);
        assert_eq!(config.effective_head_dim(), 128);
    }

    #[test]
    fn test_small_config() {
        let config = NeuralMistralCausalConfig::mistral_small();
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.hidden_size, 2048);
    }

    #[test]
    fn test_parameter_estimation() {
        let config = NeuralMistralCausalConfig::mistral_small();
        let params = config.estimate_parameters();
        assert!(params > 1_000_000_000); // > 1B
    }

    #[test]
    fn test_attention_repacker() {
        let heads = 16u32;
        let inner_dim = 4usize;
        let dim1 = 64usize;
        let dim0 = heads as usize * 2 * inner_dim;

        let data_len = dim0 * dim1;
        let data: Vec<f32> = (0..data_len).map(|i| i as f32).collect();
        let shape = vec![dim0 as u64, dim1 as u64];

        assert!(CausalAttentionRepacker::validate_shape(&shape, heads));

        let result = CausalAttentionRepacker::repack_weights(&data, &shape, heads);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), data_len);
    }

    #[test]
    fn test_needs_repack() {
        assert!(CausalAttentionRepacker::needs_repack("blk.0.attn_q.weight"));
        assert!(CausalAttentionRepacker::needs_repack("blk.0.attn_k.weight"));
        assert!(!CausalAttentionRepacker::needs_repack("blk.0.attn_v.weight"));
        assert!(!CausalAttentionRepacker::needs_repack("blk.0.attn_output.weight"));
    }

    #[test]
    fn test_get_head_count() {
        assert_eq!(CausalAttentionRepacker::get_head_count("blk.0.attn_q.weight", 32, 8), 32);
        assert_eq!(CausalAttentionRepacker::get_head_count("blk.0.attn_k.weight", 32, 8), 8);
        assert_eq!(CausalAttentionRepacker::get_head_count("blk.0.attn_k.weight", 32, 0), 32);
    }

    #[test]
    fn test_sliding_window_handler() {
        let handler = SlidingWindowHandler::new(Some(4096));
        assert!(handler.is_enabled);
        assert_eq!(handler.window_size, 4096);
        assert!(handler.use_sliding_for_layer(0, 32));
        assert!(handler.use_sliding_for_layer(1, 32));

        let handler_none = SlidingWindowHandler::new(None);
        assert!(!handler_none.is_enabled);
        assert!(!handler_none.use_sliding_for_layer(0, 32));
    }

    #[test]
    fn test_sliding_window_patterns() {
        let handler = SlidingWindowHandler::new(Some(4096))
            .with_pattern(SlidingWindowPattern::Alternating);
        
        assert!(handler.use_sliding_for_layer(0, 32));
        assert!(!handler.use_sliding_for_layer(1, 32));
        assert!(handler.use_sliding_for_layer(2, 32));
    }

    #[test]
    fn test_tensor_replacement() {
        let params = NeuralMistralCausalParameters::new();
        let converter = NeuralMistralCausalConverter::new(params);

        let replaced = converter.replace_tensor_name("model.layers.0.self_attn.q_proj.weight");
        assert!(replaced.contains("blk"));
        assert!(replaced.contains("attn_q"));
        assert!(!replaced.contains("model.layers"));
    }

    #[test]
    fn test_parameter_info() {
        let params = NeuralMistralCausalParameters::new();
        let converter = NeuralMistralCausalConverter::new(params);
        let info = converter.parameter_info();

        assert!(info.uses_gqa);
        assert!(info.uses_sliding_window.is_some());
        assert_eq!(info.gqa_group_size, 4);
        assert!(info.estimated_parameters > 1_000_000_000);
    }

    #[test]
    fn test_model_summary() {
        let params = NeuralMistralCausalParameters::new();
        let converter = NeuralMistralCausalConverter::new(params);
        let summary = converter.model_summary();

        assert!(summary.contains("Mistral3-Causal"));
        assert!(summary.contains("layers"));
        assert!(summary.contains("B params"));
    }

    #[test]
    fn test_converter_trait() {
        let params = NeuralMistralCausalParameters::new();
        let converter = NeuralMistralCausalConverter::new(params);

        assert_eq!(converter.architecture(), "mistral3_causal");
        assert_eq!(converter.coordinate().tier, 980);
    }

    #[test]
    fn test_memory_calculations() {
        let memory = calculate_attention_memory(1, 4096, 32, 128, 8);
        assert!(memory > 0);

        let kv_cache = estimate_kv_cache_size(32, 1, 32768, 8, 128);
        assert!(kv_cache > 0);
    }

    #[test]
    fn test_factory_functions() {
        let default_converter = create_mistral_7b_converter();
        assert_eq!(default_converter.architecture(), "mistral3_causal");

        let small_converter = create_mistral_small_converter();
        let info = small_converter.parameter_info();
        assert_eq!(info.num_layers, 24);

        let yarn_converter = create_mistral_with_yarn(4.0);
        let yarn_info = yarn_converter.parameter_info();
        assert!(yarn_info.uses_advanced_rope);
        assert_eq!(yarn_info.rope_factor, 4.0);
    }
}
