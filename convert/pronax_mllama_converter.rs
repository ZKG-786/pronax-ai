use std::collections::HashMap;
use std::io;
use std::sync::Arc;

use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

use crate::fs::ggml::pronax_ggml_types::{GgmlTensor, SpatialTensorMetadata};
use crate::tokenizer::pronax_bpe_tokenizer::NeuralTokenizer;

/// 3D spatial MLLaMA conversion context (multimodal Llama)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MllamaSpatialContext {
    /// Width - text hidden dimension
    pub semantic_width: u32,
    /// Height - transformer layers with cross-attention
    pub transformer_height: u32,
    /// Depth - vision transformer depth with tiles
    pub vision_depth: u32,
    /// Cross-attention layer indices
    pub cross_attention_indices: [i32; 8],
    /// Conversion guidance scale
    pub conversion_guidance: f32,
}

impl MllamaSpatialContext {
    pub const fn new(width: u32, height: u32, depth: u32, guidance: f32) -> Self {
        Self { semantic_width: width, transformer_height: height, vision_depth: depth, cross_attention_indices: [-1; 8], conversion_guidance: guidance }
    }
    pub const fn with_cross_attn(width: u32, height: u32, depth: u32, indices: [i32; 8], guidance: f32) -> Self {
        Self { semantic_width: width, transformer_height: height, vision_depth: depth, cross_attention_indices: indices, conversion_guidance: guidance }
    }
    pub const fn standard() -> Self { Self::new(4096, 32, 32, 1.0) }
    pub const fn mllama_base() -> Self { Self::new(4096, 32, 32, 0.9) }
    pub const fn mllama_large() -> Self { Self::new(5120, 40, 40, 1.0) }
    pub fn to_metadata(&self) -> SpatialTensorMetadata { SpatialTensorMetadata::new(self.semantic_width, self.transformer_height, self.vision_depth) }
}

impl Default for MllamaSpatialContext { fn default() -> Self { Self::standard() } }

/// Gated positional embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralGatedPositionalEmbedding {
    pub use_tile_embedding: bool, pub use_pre_post_tile: bool,
    pub max_num_tiles: u32, pub tile_size: u32,
    pub gate_activation: Arc<str>,
}

impl Default for NeuralGatedPositionalEmbedding {
    fn default() -> Self { Self { use_tile_embedding: true, use_pre_post_tile: true, max_num_tiles: 4, tile_size: 224, gate_activation: Arc::from("tanh") } }
}

/// Vision model configuration with tiles and global layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralMllamaVisionConfig {
    pub num_hidden_layers: u32, pub num_global_layers: u32,
    pub intermediate_layers_indices: Vec<i32>,
    pub hidden_size: u32, pub intermediate_size: u32,
    pub attention_heads: u32, pub image_size: u32,
    pub patch_size: u32, pub num_channels: u32,
    pub max_num_tiles: u32, pub norm_epsilon: f32,
    pub rope_theta: f32, pub gated_position: NeuralGatedPositionalEmbedding,
}

impl Default for NeuralMllamaVisionConfig {
    fn default() -> Self {
        Self { num_hidden_layers: 32, num_global_layers: 8, intermediate_layers_indices: vec![4, 12, 20, 28],
            hidden_size: 1280, intermediate_size: 5120, attention_heads: 16, image_size: 224,
            patch_size: 14, num_channels: 3, max_num_tiles: 4, norm_epsilon: 1e-6,
            rope_theta: 10000.0, gated_position: NeuralGatedPositionalEmbedding::default() }
    }
}

/// Cross-attention layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCrossAttentionConfig {
    pub layer_indices: Vec<i32>, pub num_heads: u32,
    pub head_dim: u32, pub use_gate: bool,
}

impl Default for NeuralCrossAttentionConfig {
    fn default() -> Self { Self { layer_indices: vec![], num_heads: 32, head_dim: 128, use_gate: true } }
}

/// Text model configuration extending Llama
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralMllamaTextConfig {
    pub hidden_size: u32, pub num_hidden_layers: u32,
    pub num_attention_heads: u32, pub num_key_value_heads: u32,
    pub intermediate_size: u32, pub rms_norm_eps: f32,
    pub rope_theta: f32, pub vocab_size: u32,
    pub cross_attention: NeuralCrossAttentionConfig,
}

impl Default for NeuralMllamaTextConfig {
    fn default() -> Self {
        Self { hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 8,
            intermediate_size: 14336, rms_norm_eps: 1e-6, rope_theta: 500000.0, vocab_size: 128256,
            cross_attention: NeuralCrossAttentionConfig::default() }
    }
}

/// MLLaMA model parameters with 3D spatial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralMllamaModelParams {
    pub model_type: Arc<str>, pub text_config: NeuralMllamaTextConfig,
    pub vision_config: NeuralMllamaVisionConfig,
    pub spatial_context: MllamaSpatialContext,
}

impl NeuralMllamaModelParams {
    pub fn base() -> Self {
        let cross_layers = vec![3, 7, 11, 15, 19, 23, 27, 31];
        let mut cross_indices = [-1i32; 8];
        for (i, &idx) in cross_layers.iter().enumerate() { if i < 8 { cross_indices[i] = idx; } }

        let text = NeuralMllamaTextConfig { cross_attention: NeuralCrossAttentionConfig { layer_indices: cross_layers, ..Default::default() }, ..Default::default() };
        let spatial = MllamaSpatialContext::with_cross_attn(4096, 32, 32, cross_indices, 0.9);

        Self { model_type: Arc::from("mllama"), text_config: text, vision_config: NeuralMllamaVisionConfig::default(), spatial_context: spatial }
    }

    pub fn is_cross_attention_layer(&self, layer_idx: i32) -> bool {
        self.text_config.cross_attention.layer_indices.contains(&layer_idx)
    }

    pub fn spatial_coordinate(&self) -> (u32, u32, u32, i32) {
        (self.text_config.hidden_size, self.text_config.num_hidden_layers, self.vision_config.hidden_size, self.text_config.cross_attention.layer_indices.len() as i32)
    }

    pub fn has_global_vision_layers(&self) -> bool { self.vision_config.num_global_layers > 0 }

    /// Check if vision layer is an intermediate layer
    pub fn is_intermediate_vision_layer(&self, layer_idx: i32) -> bool {
        self.vision_config.intermediate_layers_indices.contains(&layer_idx)
    }

    /// Calculate number of vision patches per tile
    pub fn patches_per_tile(&self) -> u32 {
        let v = &self.vision_config;
        (v.image_size / v.patch_size).pow(2)
    }

    /// Calculate total vision tokens (all tiles)
    pub fn total_vision_tokens(&self) -> u32 {
        self.patches_per_tile() * self.vision_config.max_num_tiles
    }

    /// Calculate effective context length with cross-attention
    pub fn effective_context_length(&self) -> u32 {
        // Text context + vision tokens from cross-attention
        self.text_config.num_hidden_layers * 2 + self.total_vision_tokens()
    }
}

impl Default for NeuralMllamaModelParams { fn default() -> Self { Self::base() } }

/// MLLaMA converter
#[derive(Debug, Clone)]
pub struct NeuralMllamaConverter {
    pub params: NeuralMllamaModelParams,
    pub tokenizer: Option<NeuralTokenizer>,
}

impl NeuralMllamaConverter {
    pub fn new(params: NeuralMllamaModelParams) -> Self { Self { params, tokenizer: None } }
    pub fn with_tokenizer(mut self, tokenizer: NeuralTokenizer) -> Self { self.tokenizer = Some(tokenizer); self }

    pub fn generate_metadata(&self, _tokenizer: &NeuralTokenizer) -> HashMap<Arc<str>, NeuralMllamaValue> {
        let mut kv = HashMap::new();
        let t = &self.params.text_config;
        let v = &self.params.vision_config;

        kv.insert(Arc::from("general.architecture"), NeuralMllamaValue::String(Arc::from("mllama")));

        // Text model (llama-compatible)
        kv.insert(Arc::from("mllama.block_count"), NeuralMllamaValue::U32(t.num_hidden_layers));
        kv.insert(Arc::from("mllama.embedding_length"), NeuralMllamaValue::U32(t.hidden_size));
        kv.insert(Arc::from("mllama.feed_forward_length"), NeuralMllamaValue::U32(t.intermediate_size));
        kv.insert(Arc::from("mllama.attention.head_count"), NeuralMllamaValue::U32(t.num_attention_heads));
        kv.insert(Arc::from("mllama.attention.head_count_kv"), NeuralMllamaValue::U32(t.num_key_value_heads));
        kv.insert(Arc::from("mllama.attention.layer_norm_rms_epsilon"), NeuralMllamaValue::F32(t.rms_norm_eps));
        kv.insert(Arc::from("mllama.rope.freq_base"), NeuralMllamaValue::F32(t.rope_theta));
        kv.insert(Arc::from("mllama.vocab_size"), NeuralMllamaValue::U32(t.vocab_size));

        // Cross-attention
        kv.insert(Arc::from("mllama.attention.cross_attention_layers"), NeuralMllamaValue::I32Array(t.cross_attention.layer_indices.clone()));

        // Vision model
        kv.insert(Arc::from("mllama.vision.block_count"), NeuralMllamaValue::U32(v.num_hidden_layers));
        kv.insert(Arc::from("mllama.vision.global.block_count"), NeuralMllamaValue::U32(v.num_global_layers));
        kv.insert(Arc::from("mllama.vision.intermediate_layers_indices"), NeuralMllamaValue::I32Array(v.intermediate_layers_indices.clone()));
        kv.insert(Arc::from("mllama.vision.embedding_length"), NeuralMllamaValue::U32(v.hidden_size));
        kv.insert(Arc::from("mllama.vision.feed_forward_length"), NeuralMllamaValue::U32(v.intermediate_size));
        kv.insert(Arc::from("mllama.vision.attention.head_count"), NeuralMllamaValue::U32(v.attention_heads));
        kv.insert(Arc::from("mllama.vision.attention.layer_norm_epsilon"), NeuralMllamaValue::F32(v.norm_epsilon));
        kv.insert(Arc::from("mllama.vision.image_size"), NeuralMllamaValue::U32(v.image_size));
        kv.insert(Arc::from("mllama.vision.patch_size"), NeuralMllamaValue::U32(v.patch_size));
        kv.insert(Arc::from("mllama.vision.max_num_tiles"), NeuralMllamaValue::U32(v.max_num_tiles));
        kv.insert(Arc::from("mllama.vision.num_channels"), NeuralMllamaValue::U32(v.num_channels));

        kv
    }

    pub fn get_replacements(&self) -> Vec<(Arc<str>, Arc<str>)> {
        let mut out = Vec::with_capacity(48);

        // Base llama replacements
        out.extend(vec![
            (Arc::from("model.embed_tokens"), Arc::from("token_embd")),
            (Arc::from("model.layers"), Arc::from("blk")),
            (Arc::from("model.norm"), Arc::from("output_norm")),
            (Arc::from("lm_head"), Arc::from("output")),
            (Arc::from("self_attn.q_proj"), Arc::from("attn_q")),
            (Arc::from("self_attn.k_proj"), Arc::from("attn_k")),
            (Arc::from("self_attn.v_proj"), Arc::from("attn_v")),
            (Arc::from("self_attn.o_proj"), Arc::from("attn_output")),
            (Arc::from("mlp.gate_proj"), Arc::from("ffn_gate")),
            (Arc::from("mlp.up_proj"), Arc::from("ffn_up")),
            (Arc::from("mlp.down_proj"), Arc::from("ffn_down")),
            (Arc::from("input_layernorm"), Arc::from("attn_norm")),
            (Arc::from("post_attention_layernorm"), Arc::from("ffn_norm")),
        ]);

        // MLLaMA-specific
        out.extend(vec![
            (Arc::from("language_model."), Arc::from("")),
            (Arc::from("gate_attn"), Arc::from("attn_gate")),
            (Arc::from("gate_ffn"), Arc::from("ffn_gate")),
            (Arc::from("cross_attn."), Arc::from("cross_attn_")),
            (Arc::from("vision_model"), Arc::from("v")),
            (Arc::from("class_embedding"), Arc::from("class_embd")),
            (Arc::from("patch_embedding"), Arc::from("patch_embd")),
            (Arc::from("gated_positional_embedding.tile_embedding"), Arc::from("tile_position_embd")),
            (Arc::from("gated_positional_embedding.embedding"), Arc::from("position_embd.weight")),
            (Arc::from("gated_positional_embedding"), Arc::from("position_embd")),
            (Arc::from("embedding.weight"), Arc::from("weight")),
            (Arc::from("pre_tile_positional_embedding"), Arc::from("pre_tile_position_embd")),
            (Arc::from("post_tile_positional_embedding"), Arc::from("post_tile_position_embd")),
            (Arc::from("layernorm_pre"), Arc::from("pre_ln")),
            (Arc::from("layernorm_post"), Arc::from("post_ln")),
            (Arc::from("global_transformer.layers"), Arc::from("global.blk")),
            (Arc::from("transformer.layers"), Arc::from("blk")),
            (Arc::from("mlp.fc1"), Arc::from("ffn_up")),
            (Arc::from("mlp.fc2"), Arc::from("ffn_down")),
            (Arc::from("multi_modal_projector"), Arc::from("mm.0")),
        ]);
        out
    }

    pub fn convert_tensor_name(&self, name: &str) -> Arc<str> {
        let mut result = Arc::from(name);
        for (old, new) in self.get_replacements() { result = Arc::from(result.replace(old.as_ref(), new.as_ref())); }
        result
    }

    pub fn convert_tensors(&self, tensors: &[GgmlTensor]) -> Vec<GgmlTensor> {
        let mut out = Vec::new();
        let mut text_tensors = Vec::new();

        for t in tensors {
            let name = t.name.clone();

            if !name.starts_with("v.") && !name.starts_with("mm.") {
                text_tensors.push(t.clone());
            } else if name == "v.position_embd.gate" {
                for gate_name in ["v.position_embd.gate", "v.tile_position_embd.gate"] {
                    let mut tt = t.clone();
                    tt.name = gate_name.to_string();
                    tt.needs_repack = true;
                    tt.repack_type = Some(NeuralRepackType::GatedEmbedding(gate_name.to_string()));
                    out.push(tt);
                }
            } else {
                let mut tt = t.clone();

                if name == "v.pre_tile_position_embd.gate" || name == "v.post_tile_position_embd.gate" {
                    tt.needs_repack = true;
                    tt.repack_type = Some(NeuralRepackType::GatedEmbedding(name.clone()));
                } else if name.ends_with("attn_q.weight") || name.ends_with("attn_k.weight") {
                    tt.needs_repack = true;
                    tt.repack_type = Some(NeuralRepackType::VisionAttention { heads: self.params.vision_config.attention_heads });
                } else if name.ends_with("attn_gate") || name.ends_with("ffn_gate") {
                    tt.needs_repack = true;
                    tt.repack_type = Some(NeuralRepackType::GatedActivation);
                }

                out.push(tt);
            }
        }

        // Convert text model tensors
        for t in text_tensors {
            let mut converted = t.clone();
            converted.name = self.convert_tensor_name(&t.name).to_string();
            out.push(converted);
        }

        out
    }

    /// Repack function for gated embeddings and attention
    pub fn repack_gated(&self, data: &[f32], shape: &[u64], name: &str) -> Result<Vec<f32>, NeuralMllamaError> {
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        if name.ends_with("attn_q.weight") || name.ends_with("attn_k.weight") {
            let heads = self.params.vision_config.attention_heads as usize;
            let dim0 = dims[0];
            if dim0 % (heads * 2) != 0 { return Ok(data.to_vec()); }

            let head_dim = dim0 / heads / 2;
            let new_shape = vec![heads, 2, head_dim, dims[1]];

            let array = ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec()).map_err(|e| NeuralMllamaError::RepackError(e.to_string()))?;
            let reshaped = array.into_shape_with_order(IxDyn(&new_shape)).map_err(|e| NeuralMllamaError::RepackError(e.to_string()))?;
            let permuted = reshaped.permuted_axes([0, 2, 1, 3]);
            let flattened = permuted.into_shape_with_order(IxDyn(&dims)).map_err(|e| NeuralMllamaError::RepackError(e.to_string()))?;
            let transposed = flattened.t();

            Ok(transposed.iter().copied().collect())
        } else {
            // Gated embedding with tanh activation
            let mut result: Vec<f32> = data.iter().map(|&v| v.tanh()).collect();

            // For position embeddings: apply (1 - tanh(x)) transformation
            // This is specific to MLLaMA's gated positional embeddings
            if name == "v.position_embd.gate" || name == "v.tile_position_embd.gate" ||
               name == "v.pre_tile_position_embd.gate" || name == "v.post_tile_position_embd.gate" {
                result = result.iter().map(|&v| 1.0 - v).collect();
            }

            Ok(result)
        }
    }

    /// Apply tanh gating with optional 1-x transform
    pub fn apply_tanh_gate(&self, data: &[f32], invert: bool) -> Vec<f32> {
        if invert {
            data.iter().map(|&v| 1.0 - v.tanh()).collect()
        } else {
            data.iter().map(|&v| v.tanh()).collect()
        }
    }

    /// Repack attention weights with interleaved head permutation
    pub fn repack_attention_interleaved(
        &self,
        data: &[f32],
        shape: &[u64],
        num_heads: u32,
    ) -> Result<Vec<f32>, NeuralMllamaError> {
        if shape.len() < 2 {
            return Ok(data.to_vec());
        }

        let heads = num_heads as usize;
        let dim0 = shape[0] as usize;
        let dim1 = shape[1] as usize;

        // Check if divisible by heads * 2
        if dim0 % (heads * 2) != 0 {
            return Ok(data.to_vec());
        }

        let inner_dim = dim0 / heads / 2;
        let mut result = vec![0.0f32; data.len()];

        // Reshape [dim0, dim1] -> [heads, 2, inner_dim, dim1]
        // Permute to [heads, inner_dim, 2, dim1]
        for h in 0..heads {
            for i in 0..inner_dim {
                for j in 0..2 {
                    for k in 0..dim1 {
                        // Source: [h, j, i, k]
                        let src_idx = h * 2 * inner_dim * dim1 + j * inner_dim * dim1 + i * dim1 + k;
                        // Target: [h, i, j, k]
                        let dst_idx = h * 2 * inner_dim * dim1 + i * 2 * dim1 + j * dim1 + k;

                        if src_idx < data.len() && dst_idx < result.len() {
                            result[dst_idx] = data[src_idx];
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    pub fn estimate_size(&self) -> u64 {
        let vocab_size = self.tokenizer.as_ref().map(|t| t.vocab_size()).unwrap_or(128256) as u64;
        let t = &self.params.text_config;
        let v = &self.params.vision_config;

        // Text model
        let token_embd = vocab_size * t.hidden_size as u64 * 4;
        let q_proj = t.hidden_size as u64 * (t.num_attention_heads * 128) as u64 * 4;
        let k_proj = t.hidden_size as u64 * (t.num_key_value_heads * 128) as u64 * 4;
        let v_proj = k_proj;
        let o_proj = (t.num_attention_heads * 128) as u64 * t.hidden_size as u64 * 4;
        let ffn_gate = t.hidden_size as u64 * t.intermediate_size as u64 * 4;
        let ffn_up = ffn_gate;
        let ffn_down = t.intermediate_size as u64 * t.hidden_size as u64 * 4;
        let text_layer = q_proj + k_proj + v_proj + o_proj + ffn_gate + ffn_up + ffn_down + 3 * t.hidden_size as u64 * 4;
        let text_total = token_embd + t.num_hidden_layers as u64 * text_layer + t.hidden_size as u64 * 4;

        // Vision model
        let patch_embed = (v.patch_size * v.patch_size * v.num_channels) as u64 * v.hidden_size as u64 * 4;
        let vision_qkv = v.hidden_size as u64 * (v.attention_heads * 3 * v.hidden_size / v.attention_heads) as u64 * 4;
        let vision_out = (v.attention_heads * v.hidden_size / v.attention_heads) as u64 * v.hidden_size as u64 * 4;
        let vision_ffn = v.hidden_size as u64 * v.intermediate_size as u64 * 4 * 3;
        let vision_layer = vision_qkv + vision_out + vision_ffn + 2 * v.hidden_size as u64 * 4;
        let vision_total = patch_embed + v.num_hidden_layers as u64 * vision_layer + v.num_global_layers as u64 * vision_layer;

        text_total + vision_total
    }

    pub fn spatial_metadata(&self) -> SpatialTensorMetadata { self.params.spatial_context.to_metadata() }

    pub fn model_summary(&self) -> String {
        let t = &self.params.text_config;
        let v = &self.params.vision_config;
        format!("MLLaMA {} - {}B params, Text: {} layers ({} cross-attn), Vision: {} layers ({} global), Tiles: {}",
            self.params.model_type, self.estimate_size() / 4 / 1_000_000_000,
            t.num_hidden_layers, t.cross_attention.layer_indices.len(),
            v.num_hidden_layers, v.num_global_layers, v.max_num_tiles)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuralRepackType {
    VisionAttention { heads: u32 }, GatedEmbedding(String), GatedActivation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NeuralMllamaValue {
    String(Arc<str>), U32(u32), F32(f32), I32Array(Vec<i32>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum NeuralMllamaError {
    ConfigReadError(String), ConfigParseError(String), TensorConversionError(String),
    RepackError(String), IoError(String),
}

impl std::fmt::Display for NeuralMllamaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConfigReadError(e) => write!(f, "Config read error: {}", e),
            Self::ConfigParseError(e) => write!(f, "Config parse error: {}", e),
            Self::TensorConversionError(e) => write!(f, "Tensor conversion error: {}", e),
            Self::RepackError(e) => write!(f, "Repack error: {}", e),
            Self::IoError(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for NeuralMllamaError {}
impl From<io::Error> for NeuralMllamaError { fn from(e: io::Error) -> Self { NeuralMllamaError::IoError(e.to_string()) } }

pub type MllamaConverter = NeuralMllamaConverter;
pub type MllamaParams = NeuralMllamaModelParams;
pub type MllamaError = NeuralMllamaError;
pub type MllamaContext = MllamaSpatialContext;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_context() {
        let ctx = MllamaContext::mllama_base();
        assert_eq!(ctx.semantic_width, 4096);
        assert_eq!(ctx.cross_attention_indices[0], 3);
    }

    #[test]
    fn test_vision_config() {
        let cfg = NeuralMllamaVisionConfig::default();
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.max_num_tiles, 4);
        assert_eq!(cfg.num_global_layers, 8);
    }

    #[test]
    fn test_cross_attention() {
        let params = NeuralMllamaModelParams::base();
        assert!(params.is_cross_attention_layer(3));
        assert!(params.is_cross_attention_layer(31));
        assert!(!params.is_cross_attention_layer(4));
        assert!(!params.is_cross_attention_layer(0));
    }

    #[test]
    fn test_intermediate_vision_layers() {
        let params = NeuralMllamaModelParams::base();
        assert!(params.is_intermediate_vision_layer(4));
        assert!(params.is_intermediate_vision_layer(12));
        assert!(!params.is_intermediate_vision_layer(0));
        assert!(!params.is_intermediate_vision_layer(5));
    }

    #[test]
    fn test_tile_calculations() {
        let params = NeuralMllamaModelParams::base();
        let patches = params.patches_per_tile();
        assert_eq!(patches, (224 / 14).pow(2)); // 256 patches per tile

        let total_tokens = params.total_vision_tokens();
        assert_eq!(total_tokens, patches * 4); // 4 tiles max
    }

    #[test]
    fn test_model_params() {
        let params = NeuralMllamaModelParams::base();
        assert_eq!(params.text_config.cross_attention.layer_indices.len(), 8);
        assert!(params.has_global_vision_layers());
    }

    #[test]
    fn test_gated_position() {
        let gated = NeuralGatedPositionalEmbedding::default();
        assert!(gated.use_tile_embedding);
        assert!(gated.use_pre_post_tile);
        assert_eq!(gated.gate_activation.as_ref(), "tanh");
        assert_eq!(gated.max_num_tiles, 4);
    }

    #[test]
    fn test_tensor_replacements() {
        let converter = NeuralMllamaConverter::new(NeuralMllamaModelParams::base());
        let reps = converter.get_replacements();
        assert!(reps.iter().any(|(k, _)| k.as_ref().contains("vision_model")));
        assert!(reps.iter().any(|(k, _)| k.as_ref().contains("language_model")));
        assert!(reps.iter().any(|(k, _)| k.as_ref().contains("cross_attn")));
    }

    #[test]
    fn test_tensor_name_conversion() {
        let converter = NeuralMllamaConverter::new(NeuralMllamaModelParams::base());

        let vision_converted = converter.convert_tensor_name("vision_model.patch_embedding.weight");
        assert!(vision_converted.as_ref().contains("v.patch_embd"));

        let cross_converted = converter.convert_tensor_name("cross_attn.q_proj.weight");
        assert!(cross_converted.as_ref().contains("cross_attn_q"));
    }

    #[test]
    fn test_estimate_size() {
        let converter = NeuralMllamaConverter::new(NeuralMllamaModelParams::base());
        let size = converter.estimate_size();
        assert!(size > 5_000_000_000);
    }

    #[test]
    fn test_model_summary() {
        let converter = NeuralMllamaConverter::new(NeuralMllamaModelParams::base());
        let summary = converter.model_summary();
        assert!(summary.contains("MLLaMA"));
        assert!(summary.contains("cross-attn"));
        assert!(summary.contains("global"));
    }

    #[test]
    fn test_tanh_gate() {
        let converter = NeuralMllamaConverter::new(NeuralMllamaModelParams::base());
        let data = vec![0.0f32, 1.0f32, -1.0f32, 2.0f32];

        let gated = converter.apply_tanh_gate(&data, false);
        assert!(gated[0].abs() < 0.01); // tanh(0) = 0
        assert!(gated[1] > 0.7 && gated[1] < 0.8); // tanh(1) ≈ 0.76

        let inverted = converter.apply_tanh_gate(&data, true);
        assert!(inverted[0] > 0.99); // 1 - tanh(0) = 1
    }

    #[test]
    fn test_repack_attention() {
        let converter = NeuralMllamaConverter::new(NeuralMllamaModelParams::base());

        // Create test data: [16 * 2 * 4, 64] = [128, 64] for 16 heads
        let heads = 16u32;
        let head_dim_half = 4usize;
        let dim1 = 64usize;
        let dim0 = heads as usize * 2 * head_dim_half;

        let data_len = dim0 * dim1;
        let data: Vec<f32> = (0..data_len).map(|i| i as f32).collect();
        let shape = vec![dim0 as u64, dim1 as u64];

        let result = converter.repack_attention_interleaved(&data, &shape, heads);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), data_len);
    }

    #[test]
    fn test_repack_gated_embedding() {
        let converter = NeuralMllamaConverter::new(NeuralMllamaModelParams::base());

        let data = vec![0.0f32, 0.5f32, 1.0f32, 2.0f32];
        let shape = vec![4, 1];

        // Test position embedding gate
        let result = converter.repack_gated(&data, &shape, "v.position_embd.gate");
        assert!(result.is_ok());

        let gated = result.unwrap();
        // Check that values are transformed with (1 - tanh(x))
        assert!(gated[0] > 0.99); // 1 - tanh(0) ≈ 1
        assert!(gated[1] < 1.0); // 1 - tanh(0.5) < 1
    }

    #[test]
    fn test_effective_context_length() {
        let params = NeuralMllamaModelParams::base();
        let context = params.effective_context_length();
        assert!(context > 0);
        assert!(context > params.text_config.num_hidden_layers);
    }

    #[test]
    fn test_metadata_generation() {
        let converter = NeuralMllamaConverter::new(NeuralMllamaModelParams::base());
        let tokenizer = NeuralTokenizer::default();
        let metadata = converter.generate_metadata(&tokenizer);

        assert!(metadata.contains_key(&Arc::from("general.architecture")));
        assert!(metadata.contains_key(&Arc::from("mllama.vision.block_count")));
        assert!(metadata.contains_key(&Arc::from("mllama.attention.cross_attention_layers")));
    }
}