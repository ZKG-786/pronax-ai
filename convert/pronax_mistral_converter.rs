
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::convert::pronax_converter_core::{
    ConversionCoordinate, NeuralMetadataKV, NeuralModelConverter, NeuralSourceTensor,
};
use crate::fs::ggml::pronax_ggml_format::NeuralGgmlTensor;
use crate::tokenizer::pronax_vocabulary::{NeuralConversionTokenizer, SpecialTokenType};

/// 3D Spatial coordinate for Mistral3 conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Mistral3Coordinate {
    pub sequence_id: u64,
    pub vision_tier: u16,
    pub rope_depth: u8,
    pub scaling_score: f32,
}

impl Mistral3Coordinate {
    pub const fn new(seq: u64, tier: u16, depth: u8, score: f32) -> Self {
        Self {
            sequence_id: seq,
            vision_tier: tier,
            rope_depth: depth,
            scaling_score: score,
        }
    }

    pub const fn text_attention() -> Self {
        Self::new(0, 960, 20, 0.999)
    }

    pub const fn vision_encoder() -> Self {
        Self::new(0, 940, 18, 0.997)
    }

    pub const fn rope_scaled() -> Self {
        Self::new(0, 920, 16, 0.995)
    }

    pub const fn multimodal_projector() -> Self {
        Self::new(0, 900, 14, 0.993)
    }

    pub const fn ffn_transformer() -> Self {
        Self::new(0, 880, 12, 0.99)
    }

    /// Calculate importance score
    pub fn importance_score(&self) -> u64 {
        let seq_factor = 1000u64.saturating_sub(self.sequence_id);
        let tier_boost = self.vision_tier as u64 * 90;
        let depth_norm = self.rope_depth as u64 * 8;
        let scaling_boost = (self.scaling_score * 1200.0) as u64;

        seq_factor + tier_boost + depth_norm + scaling_boost
    }
}

/// RoPE scaling type for Mistral3
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuralRoPEScalingType {
    Linear,
    Yarn,
    Llama3,
    None,
}

impl NeuralRoPEScalingType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "linear" => Self::Linear,
            "yarn" => Self::Yarn,
            "llama3" => Self::Llama3,
            _ => Self::None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Yarn => "yarn",
            Self::Llama3 => "llama3",
            Self::None => "none",
        }
    }

    pub fn is_active(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Advanced RoPE scaling parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralMistralRoPEParams {
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

impl NeuralMistralRoPEParams {
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

    pub fn get_scaling_type(&self) -> NeuralRoPEScalingType {
        NeuralRoPEScalingType::from_str(&self.rope_type)
    }

    pub fn has_advanced_scaling(&self) -> bool {
        self.beta_fast != 32.0 ||
        self.beta_slow != 1.0 ||
        self.factor != 1.0 ||
        self.llama4_scaling_beta.is_some() ||
        self.mscale.is_some() ||
        self.mscale_all_dim.is_some()
    }
}

impl Default for NeuralMistralRoPEParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Text model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct NeuralMistral3TextConfig {
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
    pub rope_params: NeuralMistralRoPEParams,
}

impl NeuralMistral3TextConfig {
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
            rope_params: NeuralMistralRoPEParams::default(),
        }
    }

    pub fn uses_gqa(&self) -> bool {
        self.num_key_value_heads > 0 && self.num_key_value_heads != self.num_attention_heads
    }

    pub fn has_sliding_window(&self) -> bool {
        self.sliding_window.is_some()
    }

    pub fn effective_head_dim(&self) -> u32 {
        if self.head_dim > 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_attention_heads
        }
    }

    pub fn rope_dimension_count(&self) -> u32 {
        self.effective_head_dim()
    }

    pub fn effective_rope_theta(&self) -> f32 {
        if self.rope_params.rope_theta > 0.0 {
            self.rope_params.rope_theta
        } else {
            self.rope_theta
        }
    }
}

impl Default for NeuralMistral3TextConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Vision model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct NeuralMistral3VisionConfig {
    #[serde(rename = "num_attention_heads")]
    pub num_attention_heads: u32,
    #[serde(rename = "num_hidden_layers")]
    pub num_hidden_layers: u32,
    #[serde(rename = "hidden_size")]
    pub hidden_size: u32,
    #[serde(rename = "intermediate_size")]
    pub intermediate_size: u32,
    #[serde(rename = "image_size")]
    pub image_size: u32,
    #[serde(rename = "num_channels")]
    pub num_channels: u32,
    #[serde(rename = "patch_size")]
    pub patch_size: u32,
    #[serde(rename = "head_dim")]
    pub head_dim: u32,
    #[serde(rename = "hidden_act")]
    pub hidden_act: String,
    #[serde(rename = "rope_theta")]
    pub rope_theta: f32,
    #[serde(rename = "rope_parameters")]
    pub rope_params: NeuralVisionRoPEParams,
}

/// Vision-specific RoPE params
#[derive(Debug, Clone, Deserialize)]
pub struct NeuralVisionRoPEParams {
    #[serde(rename = "rope_theta")]
    pub rope_theta: f32,
}

impl Default for NeuralVisionRoPEParams {
    fn default() -> Self {
        Self { rope_theta: 10000.0 }
    }
}

impl NeuralMistral3VisionConfig {
    pub fn new() -> Self {
        Self {
            num_attention_heads: 16,
            num_hidden_layers: 24,
            hidden_size: 1024,
            intermediate_size: 4096,
            image_size: 336,
            num_channels: 3,
            patch_size: 14,
            head_dim: 64,
            hidden_act: "gelu".to_string(),
            rope_theta: 10000.0,
            rope_params: NeuralVisionRoPEParams::default(),
        }
    }

    pub fn num_patches(&self) -> u32 {
        (self.image_size / self.patch_size).pow(2)
    }

    pub fn effective_rope_theta(&self) -> f32 {
        if self.rope_params.rope_theta > 0.0 {
            self.rope_params.rope_theta
        } else {
            self.rope_theta
        }
    }
}

impl Default for NeuralMistral3VisionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Mistral3 model parameters
#[derive(Debug, Clone, Deserialize)]
pub struct NeuralMistral3Parameters {
    #[serde(rename = "image_token_index")]
    pub image_token_index: u32,
    #[serde(rename = "spatial_merge_size")]
    pub spatial_merge_size: u32,
    #[serde(rename = "vision_feature_layer")]
    pub vision_feature_layer: i32,
    #[serde(rename = "text_config")]
    pub text: NeuralMistral3TextConfig,
    #[serde(rename = "vision_config")]
    pub vision: NeuralMistral3VisionConfig,
    #[serde(rename = "multimodal_projector_bias")]
    pub multimodal_projector_bias: bool,
    #[serde(rename = "projector_hidden_act")]
    pub projector_hidden_act: String,
}

impl NeuralMistral3Parameters {
    pub fn new() -> Self {
        Self {
            image_token_index: 10,
            spatial_merge_size: 2,
            vision_feature_layer: -2,
            text: NeuralMistral3TextConfig::default(),
            vision: NeuralMistral3VisionConfig::default(),
            multimodal_projector_bias: false,
            projector_hidden_act: String::new(),
        }
    }

    pub fn is_multimodal(&self) -> bool {
        self.vision.num_hidden_layers > 0
    }

    pub fn uses_advanced_rope(&self) -> bool {
        self.text.rope_params.has_advanced_scaling()
    }
}

impl Default for NeuralMistral3Parameters {
    fn default() -> Self {
        Self::new()
    }
}

/// Attention weight repacker for Q/K projections
pub struct MistralAttentionRepacker;

impl MistralAttentionRepacker {
    /// Repack attention weights with interleaved head reshaping
    pub fn repack_attention_weights(
        data: &[f32],
        shape: &[u64],
        num_heads: u32,
    ) -> Result<Vec<f32>, String> {
        if shape.len() < 2 {
            return Err("Invalid shape for attention repack".to_string());
        }

        let heads = num_heads as usize;
        let dim0 = shape[0] as usize;
        let dim1 = shape[1] as usize;

        // Calculate reshaped dimensions: [heads, 2, dim0/heads/2, dim1]
        let inner_dim = dim0 / heads / 2;
        if inner_dim == 0 {
            return Ok(data.to_vec());
        }

        // Step 1: Reshape to [heads, 2, inner_dim, dim1] and permute
        let mut result = vec![0.0f32; data.len()];
        let half_inner = inner_dim;

        for h in 0..heads {
            for i in 0..half_inner {
                for j in 0..2 {
                    for k in 0..dim1 {
                        // Source: interleaved [h, j, i, k] where j is even/odd
                        let src_idx = h * 2 * half_inner * dim1 + j * half_inner * dim1 + i * dim1 + k;
                        // Target: [h, i, j, k] - grouped even then odd
                        let dst_idx = h * 2 * half_inner * dim1 + i * 2 * dim1 + j * dim1 + k;
                        if src_idx < data.len() {
                            result[dst_idx] = data[src_idx];
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Check if tensor needs repacking
    pub fn needs_repack(name: &str, is_vision: bool) -> bool {
        // Skip vision tensors
        if is_vision || name.starts_with("v.") {
            return false;
        }
        name.ends_with(".attn_q.weight") || name.ends_with(".attn_k.weight")
    }

    /// Determine number of heads for repacking
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
}

/// Sliding window attention configuration
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    pub window_size: u32,
    pub is_active: bool,
}

impl SlidingWindowConfig {
    pub fn new(window_size: Option<u32>) -> Self {
        Self {
            window_size: window_size.unwrap_or(0),
            is_active: window_size.is_some() && window_size.unwrap() > 0,
        }
    }
}

/// Mistral3 converter
#[derive(Debug, Clone)]
pub struct NeuralMistral3Converter {
    params: NeuralMistral3Parameters,
    coordinate: Mistral3Coordinate,
    tensor_replacements: HashMap<String, String>,
    sliding_window: SlidingWindowConfig,
}

impl NeuralMistral3Converter {
    /// Create new converter
    pub fn new(params: NeuralMistral3Parameters) -> Self {
        let sliding_window = SlidingWindowConfig::new(params.text.sliding_window);
        let mut converter = Self {
            params,
            coordinate: Mistral3Coordinate::text_attention(),
            tensor_replacements: HashMap::new(),
            sliding_window,
        };

        converter.initialize_replacements();
        converter
    }

    /// Initialize tensor name replacements
    fn initialize_replacements(&mut self) {
        // Language model prefix removal
        self.tensor_replacements.insert("language_model.model.norm".to_string(), "output_norm".to_string());
        self.tensor_replacements.insert("language_model.model.".to_string(), "".to_string());
        self.tensor_replacements.insert("language_model.".to_string(), "".to_string());

        // Layer naming
        self.tensor_replacements.insert("layers".to_string(), "blk".to_string());
        self.tensor_replacements.insert("transformer.layers".to_string(), "blk".to_string());

        // Vision tower
        self.tensor_replacements.insert("vision_tower".to_string(), "v".to_string());
        self.tensor_replacements.insert("ln_pre".to_string(), "encoder_norm".to_string());

        // Attention norms
        self.tensor_replacements.insert("input_layernorm".to_string(), "attn_norm".to_string());
        self.tensor_replacements.insert("post_attention_layernorm".to_string(), "ffn_norm".to_string());

        // Embeddings
        self.tensor_replacements.insert("embed_tokens".to_string(), "token_embd".to_string());

        // Text attention
        self.tensor_replacements.insert("self_attn.q_proj".to_string(), "attn_q".to_string());
        self.tensor_replacements.insert("self_attn.k_proj".to_string(), "attn_k".to_string());
        self.tensor_replacements.insert("self_attn.v_proj".to_string(), "attn_v".to_string());
        self.tensor_replacements.insert("self_attn.o_proj".to_string(), "attn_output".to_string());

        // Text FFN
        self.tensor_replacements.insert("mlp.down_proj".to_string(), "ffn_down".to_string());
        self.tensor_replacements.insert("mlp.gate_proj".to_string(), "ffn_gate".to_string());
        self.tensor_replacements.insert("mlp.up_proj".to_string(), "ffn_up".to_string());

        // Vision attention (alternative naming)
        self.tensor_replacements.insert("attention.q_proj".to_string(), "attn_q".to_string());
        self.tensor_replacements.insert("attention.k_proj".to_string(), "attn_k".to_string());
        self.tensor_replacements.insert("attention.v_proj".to_string(), "attn_v".to_string());
        self.tensor_replacements.insert("attention.o_proj".to_string(), "attn_output".to_string());
        self.tensor_replacements.insert("attention_norm".to_string(), "attn_norm".to_string());

        // Vision FFN
        self.tensor_replacements.insert("feed_forward.gate_proj".to_string(), "ffn_gate".to_string());
        self.tensor_replacements.insert("feed_forward.down_proj".to_string(), "ffn_down".to_string());
        self.tensor_replacements.insert("feed_forward.up_proj".to_string(), "ffn_up".to_string());

        // Multi-modal projector
        self.tensor_replacements.insert("multi_modal_projector".to_string(), "mm".to_string());

        // Output
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

    /// Get all tensor replacements as pairs
    pub fn replacement_pairs(&self) -> Vec<(String, String)> {
        self.tensor_replacements
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Check if tensor is from vision model
    pub fn is_vision_tensor(&self, name: &str) -> bool {
        name.starts_with("v.") || name.starts_with("vision_tower")
    }

    /// Get coordinate for tensor
    fn coordinate_for_tensor(&self, name: &str) -> Mistral3Coordinate {
        if self.is_vision_tensor(name) {
            Mistral3Coordinate::vision_encoder()
        } else if name.contains("mm.") || name.contains("multi_modal_projector") {
            Mistral3Coordinate::multimodal_projector()
        } else if name.contains("rope") || name.contains("scaling") {
            Mistral3Coordinate::rope_scaled()
        } else if name.contains("ffn") {
            Mistral3Coordinate::ffn_transformer()
        } else {
            Mistral3Coordinate::text_attention()
        }
    }

    /// Get parameter info
    pub fn parameter_info(&self) -> Mistral3ParameterInfo {
        Mistral3ParameterInfo {
            text_layers: self.params.text.num_hidden_layers,
            vision_layers: self.params.vision.num_hidden_layers,
            hidden_size: self.params.text.hidden_size,
            vision_hidden_size: self.params.vision.hidden_size,
            num_attention_heads: self.params.text.num_attention_heads,
            num_kv_heads: self.params.text.num_key_value_heads,
            head_dim: self.params.text.effective_head_dim(),
            sliding_window: self.params.text.sliding_window,
            uses_gqa: self.params.text.uses_gqa(),
            rope_scaling_type: self.params.text.rope_params.get_scaling_type(),
            rope_factor: self.params.text.rope_params.factor,
            image_token_index: self.params.image_token_index,
            spatial_merge_size: self.params.spatial_merge_size,
            vision_feature_layer: self.params.vision_feature_layer,
            multimodal_projector_bias: self.params.multimodal_projector_bias,
            is_multimodal: self.params.is_multimodal(),
            uses_advanced_rope: self.params.uses_advanced_rope(),
            coordinate: self.coordinate,
        }
    }
}

/// Mistral3 parameter info
#[derive(Debug, Clone)]
pub struct Mistral3ParameterInfo {
    pub text_layers: u32,
    pub vision_layers: u32,
    pub hidden_size: u32,
    pub vision_hidden_size: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub sliding_window: Option<u32>,
    pub uses_gqa: bool,
    pub rope_scaling_type: NeuralRoPEScalingType,
    pub rope_factor: f32,
    pub image_token_index: u32,
    pub spatial_merge_size: u32,
    pub vision_feature_layer: i32,
    pub multimodal_projector_bias: bool,
    pub is_multimodal: bool,
    pub uses_advanced_rope: bool,
    pub coordinate: Mistral3Coordinate,
}

impl NeuralModelConverter for NeuralMistral3Converter {
    fn to_metadata_kv(&self, tokenizer: &NeuralConversionTokenizer) -> NeuralMetadataKV {
        let mut kv = NeuralMetadataKV::new();
        let p = &self.params;
        let text = &p.text;
        let vision = &p.vision;

        // Base architecture
        kv.insert("general.architecture", "mistral3".to_string());
        kv.insert("general.type", "model".to_string());

        // Text model metadata
        kv.insert("mistral3.vocab_size", text.vocab_size);
        kv.insert("mistral3.block_count", text.num_hidden_layers);
        kv.insert("mistral3.context_length", text.max_position_embeddings);
        kv.insert("mistral3.embedding_length", text.hidden_size);
        kv.insert("mistral3.feed_forward_length", text.intermediate_size);
        kv.insert("mistral3.attention.head_count", text.num_attention_heads);
        kv.insert("mistral3.attention.head_count_kv", text.num_key_value_heads);
        kv.insert("mistral3.attention.layer_norm_rms_epsilon", text.rms_norm_eps);
        kv.insert("mistral3.attention.key_length", text.effective_head_dim());
        kv.insert("mistral3.attention.value_length", text.effective_head_dim());

        // RoPE configuration
        let rope_dim = text.rope_dimension_count();
        kv.insert("mistral3.rope.dimension_count", rope_dim);
        kv.insert("mistral3.rope.freq_base", text.effective_rope_theta());

        // RoPE scaling
        let rope_params = &text.rope_params;
        if rope_params.get_scaling_type().is_active() {
            kv.insert("mistral3.rope.scaling.type", rope_params.rope_type.clone());
            kv.insert("mistral3.rope.scaling.factor", rope_params.factor);
            kv.insert("mistral3.rope.scaling.beta_fast", rope_params.beta_fast);
            kv.insert("mistral3.rope.scaling.beta_slow", rope_params.beta_slow);

            if let Some(mscale) = rope_params.mscale {
                kv.insert("mistral3.rope.scaling.mscale", mscale);
            }
            if let Some(mscale_all) = rope_params.mscale_all_dim {
                kv.insert("mistral3.rope.scaling.mscale_all_dim", mscale_all);
            }
            if rope_params.original_max_pos_emb > 0 {
                kv.insert("mistral3.rope.scaling.original_context_length", rope_params.original_max_pos_emb);
            }
            if let Some(beta) = rope_params.llama4_scaling_beta {
                kv.insert("mistral3.rope.scaling_beta", beta);
            }
        }

        // Sliding window
        if let Some(window) = text.sliding_window {
            kv.insert("mistral3.attention.sliding_window", window);
        }

        // Vision model metadata
        kv.insert("mistral3.vision.block_count", vision.num_hidden_layers);
        kv.insert("mistral3.vision.embedding_length", vision.hidden_size);
        kv.insert("mistral3.vision.feed_forward_length", vision.intermediate_size);
        kv.insert("mistral3.vision.attention.head_count", vision.num_attention_heads);
        kv.insert("mistral3.vision.attention.key_length", vision.head_dim);
        kv.insert("mistral3.vision.image_size", vision.image_size);
        kv.insert("mistral3.vision.patch_size", vision.patch_size);
        kv.insert("mistral3.vision.num_channels", vision.num_channels);
        kv.insert("mistral3.vision.rope.freq_base", vision.effective_rope_theta());

        // Multi-modal configuration
        kv.insert("mistral3.image_token_index", p.image_token_index);
        kv.insert("mistral3.spatial_merge_size", p.spatial_merge_size);
        kv.insert("mistral3.mm.projector_bias", p.multimodal_projector_bias);

        if !p.projector_hidden_act.is_empty() {
            kv.insert("mistral3.mm.projector_hidden_act", p.projector_hidden_act.clone());
        }

        // Coordinate metadata
        kv.insert("pronax.coordinate.sequence", self.coordinate.sequence_id);
        kv.insert("pronax.coordinate.tier", self.coordinate.vision_tier);
        kv.insert("pronax.coordinate.depth", self.coordinate.rope_depth);
        kv.insert("pronax.coordinate.scaling", self.coordinate.scaling_score);

        // Tokenizer metadata
        let tokenizer_kv = tokenizer.to_kv();
        kv.merge(tokenizer_kv);

        kv.set_architecture("mistral3");
        kv
    }

    fn convert_tensors(&self, tensors: &[NeuralSourceTensor]) -> Vec<NeuralGgmlTensor> {
        let mut converted = Vec::with_capacity(tensors.len());

        for (idx, tensor) in tensors.iter().enumerate() {
            let name = self.replace_tensor_name(&tensor.name);
            let is_vision = self.is_vision_tensor(&name);

            // Apply attention repacking for non-vision Q/K weights
            let mut data = tensor.data.clone();
            if MistralAttentionRepacker::needs_repack(&name, is_vision) {
                let head_count = MistralAttentionRepacker::get_head_count(
                    &name,
                    self.params.text.num_attention_heads,
                    self.params.text.num_key_value_heads,
                );

                let float_data = Self::bytes_to_f32_slice(&data);
                if let Ok(repacked) =
                    MistralAttentionRepacker::repack_attention_weights(&float_data, &tensor.shape, head_count)
                {
                    data = Self::f32_slice_to_bytes(&repacked);
                }
            }

            // Get coordinate for this tensor
            let coord = self.coordinate_for_tensor(&name);

            let converted_tensor = NeuralGgmlTensor::new(name, tensor.data_type, tensor.shape.clone(), data)
                .with_coordinate(ConversionCoordinate::new(
                    idx as u64,
                    coord.vision_tier,
                    coord.rope_depth,
                    coord.scaling_score,
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
            SpecialTokenType::Image,
            SpecialTokenType::Unknown,
        ]
    }

    fn architecture(&self) -> &str {
        "mistral3"
    }

    fn coordinate(&self) -> ConversionCoordinate {
        ConversionCoordinate::new(
            self.coordinate.sequence_id,
            self.coordinate.vision_tier,
            self.coordinate.rope_depth,
            self.coordinate.scaling_score,
        )
    }
}

impl NeuralMistral3Converter {
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
pub fn create_mistral3_converter(params: NeuralMistral3Parameters) -> NeuralMistral3Converter {
    NeuralMistral3Converter::new(params)
}

/// Convenience function with default parameters
pub fn create_default_mistral3_converter() -> NeuralMistral3Converter {
    NeuralMistral3Converter::new(NeuralMistral3Parameters::default())
}

/// Calculate vision grid dimensions
pub fn calculate_mistral_vision_grid(image_size: u32, patch_size: u32) -> (u32, u32) {
    let grid = image_size / patch_size;
    (grid, grid)
}

/// Calculate number of vision tokens with spatial merge
pub fn calculate_spatial_tokens(image_size: u32, patch_size: u32, spatial_merge: u32) -> u32 {
    let patches = image_size / patch_size;
    let merged_grid = patches / spatial_merge;
    merged_grid * merged_grid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral3_coordinate() {
        let coord = Mistral3Coordinate::text_attention();
        assert_eq!(coord.vision_tier, 960);
        assert_eq!(coord.scaling_score, 0.999);

        let score = coord.importance_score();
        assert!(score > 0);
    }

    #[test]
    fn test_rope_scaling_type() {
        assert!(NeuralRoPEScalingType::Linear.is_active());
        assert!(NeuralRoPEScalingType::Yarn.is_active());
        assert!(!NeuralRoPEScalingType::None.is_active());

        assert_eq!(NeuralRoPEScalingType::from_str("yarn"), NeuralRoPEScalingType::Yarn);
        assert_eq!(NeuralRoPEScalingType::from_str("llama3"), NeuralRoPEScalingType::Llama3);
    }

    #[test]
    fn test_rope_params() {
        let params = NeuralMistralRoPEParams::new();
        assert_eq!(params.beta_fast, 32.0);
        assert_eq!(params.beta_slow, 1.0);
        assert!(!params.has_advanced_scaling());

        let mut advanced = NeuralMistralRoPEParams::new();
        advanced.factor = 4.0;
        assert!(advanced.has_advanced_scaling());
    }

    #[test]
    fn test_text_config() {
        let config = NeuralMistral3TextConfig::new();
        assert!(config.uses_gqa());
        assert!(config.has_sliding_window());
        assert_eq!(config.effective_head_dim(), 128);
    }

    #[test]
    fn test_vision_config() {
        let config = NeuralMistral3VisionConfig::new();
        let patches = config.num_patches();
        assert_eq!(patches, (336 / 14).pow(2));
        assert_eq!(config.num_channels, 3);
    }

    #[test]
    fn test_mistral3_parameters() {
        let params = NeuralMistral3Parameters::new();
        assert!(params.is_multimodal());
        assert!(!params.uses_advanced_rope());
        assert_eq!(params.spatial_merge_size, 2);
        assert_eq!(params.vision_feature_layer, -2);
    }

    #[test]
    fn test_sliding_window() {
        let window = SlidingWindowConfig::new(Some(4096));
        assert!(window.is_active);
        assert_eq!(window.window_size, 4096);

        let no_window = SlidingWindowConfig::new(None);
        assert!(!no_window.is_active);
    }

    #[test]
    fn test_attention_repacker() {
        // Create test data: [32 * 2 * 4, 128] = [256, 128]
        let heads = 32u32;
        let data_len = 256 * 128;
        let data: Vec<f32> = (0..data_len).map(|i| i as f32).collect();
        let shape = vec![256u64, 128];

        let result = MistralAttentionRepacker::repack_attention_weights(&data, &shape, heads);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), data_len);
    }

    #[test]
    fn test_needs_repack() {
        assert!(MistralAttentionRepacker::needs_repack("blk.0.attn_q.weight", false));
        assert!(MistralAttentionRepacker::needs_repack("blk.0.attn_k.weight", false));
        assert!(!MistralAttentionRepacker::needs_repack("blk.0.attn_q.weight", true)); // vision
        assert!(!MistralAttentionRepacker::needs_repack("v.blk.0.attn_q.weight", false)); // starts with v
    }

    #[test]
    fn test_tensor_replacement() {
        let params = NeuralMistral3Parameters::new();
        let converter = NeuralMistral3Converter::new(params);

        let replaced = converter.replace_tensor_name("language_model.model.layers.0.self_attn.q_proj.weight");
        assert!(replaced.contains("blk"));
        assert!(replaced.contains("attn_q"));
        assert!(!replaced.contains("language_model"));

        let vision_replaced = converter.replace_tensor_name("vision_tower.encoder.layers.0.attn.q_proj.weight");
        assert!(vision_replaced.starts_with("v."));
    }

    #[test]
    fn test_is_vision_tensor() {
        let params = NeuralMistral3Parameters::new();
        let converter = NeuralMistral3Converter::new(params);

        assert!(converter.is_vision_tensor("v.patch_embed.weight"));
        assert!(converter.is_vision_tensor("vision_tower.encoder.norm.weight"));
        assert!(!converter.is_vision_tensor("blk.0.attn_q.weight"));
    }

    #[test]
    fn test_parameter_info() {
        let params = NeuralMistral3Parameters::new();
        let converter = NeuralMistral3Converter::new(params);
        let info = converter.parameter_info();

        assert!(info.uses_gqa);
        assert!(info.is_multimodal);
        assert_eq!(info.rope_factor, 1.0);
        assert_eq!(info.image_token_index, 10);
    }

    #[test]
    fn test_vision_grid_calculation() {
        let (w, h) = calculate_mistral_vision_grid(336, 14);
        assert_eq!(w, 24);
        assert_eq!(h, 24);
    }

    #[test]
    fn test_spatial_tokens() {
        let tokens = calculate_spatial_tokens(336, 14, 2);
        // 24 patches / 2 merge = 12 merged grid
        assert_eq!(tokens, 144); // 12 * 12
    }

    #[test]
    fn test_converter_trait() {
        let params = NeuralMistral3Parameters::new();
        let converter = NeuralMistral3Converter::new(params);

        assert_eq!(converter.architecture(), "mistral3");
        assert_eq!(converter.coordinate().tier, 960);
    }

    #[test]
    fn test_get_head_count() {
        assert_eq!(MistralAttentionRepacker::get_head_count("blk.0.attn_q.weight", 32, 8), 32);
        assert_eq!(MistralAttentionRepacker::get_head_count("blk.0.attn_k.weight", 32, 8), 8);
        assert_eq!(MistralAttentionRepacker::get_head_count("blk.0.attn_k.weight", 32, 0), 32);
    }
}