use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{Array1, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::fs::ggml::pronax_ggml_types::{GgmlTensor, SpatialTensorMetadata};
use crate::tokenizer::pronax_bpe_tokenizer::NeuralTokenizer;

/// 3D spatial Gemma2 conversion context
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Gemma2SpatialContext {
    /// Width - hidden dimension
    pub hidden_width: u32,
    /// Height - number of transformer layers
    pub transformer_height: u32,
    /// Depth - sliding window attention span
    pub attention_depth: u32,
    /// Conversion guidance scale
    pub conversion_guidance: f32,
}

impl Gemma2SpatialContext {
    pub const fn new(width: u32, height: u32, depth: u32, guidance: f32) -> Self {
        Self {
            hidden_width: width,
            transformer_height: height,
            attention_depth: depth,
            conversion_guidance: guidance,
        }
    }

    pub const fn standard() -> Self {
        Self::new(3584, 42, 4096, 1.0)
    }

    pub const fn gemma_2b() -> Self {
        Self::new(2048, 18, 2048, 0.9)
    }

    pub const fn gemma_4b() -> Self {
        Self::new(2560, 34, 4096, 1.0)
    }

    pub const fn gemma_9b() -> Self {
        Self::new(3584, 42, 4096, 1.0)
    }

    pub const fn gemma_27b() -> Self {
        Self::new(6144, 46, 4096, 1.1)
    }

    pub fn to_metadata(&self) -> SpatialTensorMetadata {
        SpatialTensorMetadata::new(self.hidden_width, self.transformer_height, self.attention_depth)
    }
}

impl Default for Gemma2SpatialContext {
    fn default() -> Self {
        Self::standard()
    }
}

/// Gemma model architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NeuralGemmaArchitecture {
    /// Original Gemma 1.x
    Gemma = 1,
    /// Gemma 2 with softcapping
    Gemma2 = 2,
}

impl NeuralGemmaArchitecture {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gemma => "gemma",
            Self::Gemma2 => "gemma2",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gemma2" | "gemma-2" => Self::Gemma2,
            _ => Self::Gemma,
        }
    }
}

/// Special token IDs for Gemma models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeuralGemmaSpecialTokens {
    pub eot_token_id: u32,      // End of turn
    pub middle_token_id: u32,   // Code infill middle
    pub prefix_token_id: u32,   // Code infill prefix
    pub suffix_token_id: u32,   // Code infill suffix
    pub bos_token_id: u32,      // Beginning of sequence
    pub eos_token_id: u32,      // End of sequence
    pub pad_token_id: u32,      // Padding
}

impl NeuralGemmaSpecialTokens {
    pub const fn gemma_default() -> Self {
        Self {
            eot_token_id: 107,
            middle_token_id: 68,
            prefix_token_id: 67,
            suffix_token_id: 69,
            bos_token_id: 2,
            eos_token_id: 1,
            pad_token_id: 0,
        }
    }

    pub const fn gemma2_default() -> Self {
        Self {
            eot_token_id: 107,
            middle_token_id: 68,
            prefix_token_id: 67,
            suffix_token_id: 69,
            bos_token_id: 2,
            eos_token_id: 1,
            pad_token_id: 0,
        }
    }
}

impl Default for NeuralGemmaSpecialTokens {
    fn default() -> Self {
        Self::gemma2_default()
    }
}

/// Gemma/Gemma2 model hyperparameters with 3D spatial metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralGemmaModelParams {
    // Architecture type
    pub architecture: NeuralGemmaArchitecture,
    
    // Model dimensions
    pub max_position_embeddings: u32,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub intermediate_size: u32,
    
    // Attention configuration
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub rms_norm_eps: f32,
    pub head_dim: u32,
    
    // Gemma2 specific features
    pub sliding_window: Option<u32>,
    pub attention_logit_softcap: Option<f32>,
    pub final_logit_softcap: Option<f32>,
    
    // Special tokens
    pub special_tokens: NeuralGemmaSpecialTokens,
    
    // 3D spatial metadata
    pub spatial_context: Gemma2SpatialContext,
}

impl NeuralGemmaModelParams {
    pub fn gemma2() -> Self {
        Self {
            architecture: NeuralGemmaArchitecture::Gemma2,
            max_position_embeddings: 8192,
            hidden_size: 3584,
            num_hidden_layers: 42,
            intermediate_size: 14336,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-6,
            head_dim: 256,
            sliding_window: Some(4096),
            attention_logit_softcap: Some(50.0),
            final_logit_softcap: Some(30.0),
            special_tokens: NeuralGemmaSpecialTokens::gemma2_default(),
            spatial_context: Gemma2SpatialContext::gemma_9b(),
        }
    }

    pub fn gemma_2b() -> Self {
        Self {
            architecture: NeuralGemmaArchitecture::Gemma,
            max_position_embeddings: 8192,
            hidden_size: 2048,
            num_hidden_layers: 18,
            intermediate_size: 16384,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            head_dim: 256,
            sliding_window: None,
            attention_logit_softcap: None,
            final_logit_softcap: None,
            special_tokens: NeuralGemmaSpecialTokens::gemma_default(),
            spatial_context: Gemma2SpatialContext::gemma_2b(),
        }
    }

    pub fn gemma_4b() -> Self {
        Self {
            architecture: NeuralGemmaArchitecture::Gemma,
            max_position_embeddings: 8192,
            hidden_size: 2560,
            num_hidden_layers: 34,
            intermediate_size: 10240,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            head_dim: 256,
            sliding_window: None,
            attention_logit_softcap: None,
            final_logit_softcap: None,
            special_tokens: NeuralGemmaSpecialTokens::gemma_default(),
            spatial_context: Gemma2SpatialContext::gemma_4b(),
        }
    }

    pub fn gemma_27b() -> Self {
        Self {
            architecture: NeuralGemmaArchitecture::Gemma2,
            max_position_embeddings: 8192,
            hidden_size: 6144,
            num_hidden_layers: 46,
            intermediate_size: 24576,
            num_attention_heads: 32,
            num_key_value_heads: 16,
            rms_norm_eps: 1e-6,
            head_dim: 192,
            sliding_window: Some(4096),
            attention_logit_softcap: Some(50.0),
            final_logit_softcap: Some(30.0),
            special_tokens: NeuralGemmaSpecialTokens::gemma2_default(),
            spatial_context: Gemma2SpatialContext::gemma_27b(),
        }
    }

    pub fn update_spatial_context(&mut self) {
        self.spatial_context = Gemma2SpatialContext::new(
            self.hidden_size,
            self.num_hidden_layers,
            self.sliding_window.unwrap_or(self.max_position_embeddings),
            1.0,
        );
    }

    pub fn spatial_coordinate(&self) -> (u32, u32, u32) {
        (
            self.hidden_size,
            self.num_hidden_layers,
            self.sliding_window.unwrap_or(self.max_position_embeddings),
        )
    }

    pub fn is_gemma2(&self) -> bool {
        matches!(self.architecture, NeuralGemmaArchitecture::Gemma2)
    }

    pub fn has_softcapping(&self) -> bool {
        self.attention_logit_softcap.is_some() || self.final_logit_softcap.is_some()
    }

    pub fn effective_sliding_window(&self) -> u32 {
        self.sliding_window.unwrap_or(self.max_position_embeddings)
    }
}

impl Default for NeuralGemmaModelParams {
    fn default() -> Self {
        Self::gemma2()
    }
}

/// Tensor repack function type for normalization weights
pub type NeuralRepackFn = Arc<dyn Fn(&str, &[f32], &[u64]) -> Result<Vec<f32>, NeuralGemmaError> + Send + Sync>;

/// Gemma/Gemma2 model converter with 3D spatial metadata
#[derive(Debug, Clone)]
pub struct NeuralGemmaConverter {
    pub params: NeuralGemmaModelParams,
    pub tokenizer: Option<NeuralTokenizer>,
    pub tensor_cache: HashMap<Arc<str>, GgmlTensor>,
}

impl NeuralGemmaConverter {
    pub fn new(params: NeuralGemmaModelParams) -> Self {
        Self {
            params,
            tokenizer: None,
            tensor_cache: HashMap::new(),
        }
    }

    pub fn with_tokenizer(mut self, tokenizer: NeuralTokenizer) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Generate key-value metadata for GGUF
    pub fn generate_metadata(&self, tokenizer: &NeuralTokenizer) -> HashMap<Arc<str>, NeuralGemmaValue> {
        let mut kv = HashMap::new();
        let arch_str = self.params.architecture.as_str();

        // Architecture
        kv.insert(Arc::from("general.architecture"), NeuralGemmaValue::String(Arc::from(arch_str)));

        // Model dimensions
        kv.insert(Arc::from(format!("{}.context_length", arch_str)), NeuralGemmaValue::U32(self.params.max_position_embeddings));
        kv.insert(Arc::from(format!("{}.embedding_length", arch_str)), NeuralGemmaValue::U32(self.params.hidden_size));
        kv.insert(Arc::from(format!("{}.block_count", arch_str)), NeuralGemmaValue::U32(self.params.num_hidden_layers));
        kv.insert(Arc::from(format!("{}.feed_forward_length", arch_str)), NeuralGemmaValue::U32(self.params.intermediate_size));

        // Attention configuration
        kv.insert(Arc::from(format!("{}.attention.head_count", arch_str)), NeuralGemmaValue::U32(self.params.num_attention_heads));
        kv.insert(Arc::from(format!("{}.attention.head_count_kv", arch_str)), NeuralGemmaValue::U32(self.params.num_key_value_heads));
        kv.insert(Arc::from(format!("{}.attention.layer_norm_rms_epsilon", arch_str)), NeuralGemmaValue::F32(self.params.rms_norm_eps));
        kv.insert(Arc::from(format!("{}.attention.key_length", arch_str)), NeuralGemmaValue::U32(self.params.head_dim));
        kv.insert(Arc::from(format!("{}.attention.value_length", arch_str)), NeuralGemmaValue::U32(self.params.head_dim));

        // Gemma2 specific features
        if let Some(sliding_window) = self.params.sliding_window {
            kv.insert(Arc::from(format!("{}.attention.sliding_window", arch_str)), NeuralGemmaValue::U32(sliding_window));
        }
        if let Some(attn_softcap) = self.params.attention_logit_softcap {
            kv.insert(Arc::from(format!("{}.attn_logit_softcapping", arch_str)), NeuralGemmaValue::F32(attn_softcap));
        }
        if let Some(final_softcap) = self.params.final_logit_softcap {
            kv.insert(Arc::from(format!("{}.final_logit_softcapping", arch_str)), NeuralGemmaValue::F32(final_softcap));
        }

        // Special tokens
        kv.insert(Arc::from("tokenizer.ggml.eot_token_id"), NeuralGemmaValue::U32(self.params.special_tokens.eot_token_id));
        kv.insert(Arc::from("tokenizer.ggml.middle_token_id"), NeuralGemmaValue::U32(self.params.special_tokens.middle_token_id));
        kv.insert(Arc::from("tokenizer.ggml.prefix_token_id"), NeuralGemmaValue::U32(self.params.special_tokens.prefix_token_id));
        kv.insert(Arc::from("tokenizer.ggml.suffix_token_id"), NeuralGemmaValue::U32(self.params.special_tokens.suffix_token_id));

        kv
    }

    /// Get tensor name replacements
    pub fn get_replacements(&self) -> Vec<(Arc<str>, Arc<str>)> {
        let base_replacements = vec![
            (Arc::from("model.embed_tokens"), Arc::from("token_embd")),
            (Arc::from("model.norm"), Arc::from("output_norm")),
            (Arc::from("model.layers"), Arc::from("blk")),
            (Arc::from("input_layernorm"), Arc::from("attn_norm")),
            (Arc::from("self_attn.q_proj"), Arc::from("attn_q")),
            (Arc::from("self_attn.k_proj"), Arc::from("attn_k")),
            (Arc::from("self_attn.v_proj"), Arc::from("attn_v")),
            (Arc::from("self_attn.o_proj"), Arc::from("attn_output")),
            (Arc::from("mlp.gate_proj"), Arc::from("ffn_gate")),
            (Arc::from("mlp.down_proj"), Arc::from("ffn_down")),
            (Arc::from("mlp.up_proj"), Arc::from("ffn_up")),
        ];

        if self.params.is_gemma2() {
            // Gemma2 has additional norm layers
            let mut replacements = base_replacements;
            replacements.push((Arc::from("post_attention_layernorm"), Arc::from("post_attention_norm")));
            replacements.push((Arc::from("pre_feedforward_layernorm"), Arc::from("ffn_norm")));
            replacements.push((Arc::from("post_feedforward_layernorm"), Arc::from("post_ffw_norm")));
            replacements
        } else {
            // Original Gemma
            let mut replacements = base_replacements;
            replacements.push((Arc::from("post_attention_layernorm"), Arc::from("ffn_norm")));
            replacements
        }
    }

    /// Convert tensor name to GGML format
    pub fn convert_tensor_name(&self, name: &str) -> Arc<str> {
        let mut result = Arc::from(name);
        
        for (old, new) in self.get_replacements() {
            result = Arc::from(result.replace(old.as_ref(), new.as_ref()));
        }
        
        result
    }

    /// Convert and repack tensors
    pub fn convert_tensors(&self, tensors: &[GgmlTensor]) -> Result<Vec<GgmlTensor>, NeuralGemmaError> {
        let mut out = Vec::new();
        
        for t in tensors {
            let mut converted = t.clone();
            converted.name = self.convert_tensor_name(&t.name).to_string();
            
            // Apply repacking for normalization weights (Gemma specific)
            // Note: In a real implementation, this would modify the tensor data
            // by adding 1.0 to each element as per the original Gemma paper
            if !t.name.starts_with("v.") && t.name.ends_with("_norm.weight") {
                // Mark for repacking - actual repacking would happen during write
                converted.needs_repack = true;
            }
            
            out.push(converted);
        }
        
        Ok(out)
    }

    /// Repack normalization weights by adding 1.0 (Gemma-specific)
    pub fn repack_norm_weights(&self, name: &str, data: &[f32], shape: &[u64]) -> Result<Vec<f32>, NeuralGemmaError> {
        // Create ndarray from data
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let array = ArrayD::from_shape_vec(IxDyn(&shape_usize), data.to_vec())
            .map_err(|e| NeuralGemmaError::TensorRepackError(e.to_string()))?;
        
        // Add ones (Gemma pre-normalization bias)
        let ones = ArrayD::from_elem(IxDyn(&shape_usize), 1.0f32);
        let result = array + ones;
        
        // Convert back to Vec<f32>
        Ok(result.iter().copied().collect())
    }

    /// Estimate model size in bytes
    pub fn estimate_size(&self) -> u64 {
        let vocab_size = self.tokenizer.as_ref().map(|t| t.vocab_size()).unwrap_or(256000) as u64;
        let hidden_size = self.params.hidden_size as u64;
        let intermediate_size = self.params.intermediate_size as u64;
        let num_layers = self.params.num_hidden_layers as u64;
        let num_heads = self.params.num_attention_heads as u64;
        let num_kv_heads = self.params.num_key_value_heads as u64;
        let head_dim = self.params.head_dim as u64;

        // Embeddings
        let token_embd = vocab_size * hidden_size * 4; // f32

        // Attention weights (GQA - Grouped Query Attention)
        let q_proj = hidden_size * (num_heads * head_dim) * 4;
        let k_proj = hidden_size * (num_kv_heads * head_dim) * 4;
        let v_proj = hidden_size * (num_kv_heads * head_dim) * 4;
        let o_proj = (num_heads * head_dim) * hidden_size * 4;
        let attn_per_layer = q_proj + k_proj + v_proj + o_proj;

        // FFN (SwiGLU)
        let gate_proj = hidden_size * intermediate_size * 4;
        let up_proj = hidden_size * intermediate_size * 4;
        let down_proj = intermediate_size * hidden_size * 4;
        let ffn_per_layer = gate_proj + up_proj + down_proj;

        // RMS Norms
        let num_norms = if self.params.is_gemma2() { 3 } else { 2 }; // Gemma2 has extra norms
        let norms = num_norms * hidden_size * 4;

        let per_layer = attn_per_layer + ffn_per_layer + norms;
        let layers = num_layers * per_layer;

        // Output norm
        let output_norm = hidden_size * 4;

        token_embd + layers + output_norm
    }

    /// Get spatial metadata
    pub fn spatial_metadata(&self) -> SpatialTensorMetadata {
        self.params.spatial_context.to_metadata()
    }

    /// Get attention configuration summary
    pub fn attention_summary(&self) -> String {
        format!(
            "Gemma{} - {} layers, {} heads ({} KV), head_dim={}, sliding_window={}",
            if self.params.is_gemma2() { "2" } else { "" },
            self.params.num_hidden_layers,
            self.params.num_attention_heads,
            self.params.num_key_value_heads,
            self.params.head_dim,
            self.params.sliding_window.map_or("none".to_string(), |w| w.to_string())
        )
    }
}

/// Value types for Gemma metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NeuralGemmaValue {
    String(Arc<str>),
    U32(u32),
    F32(f32),
    Bool(bool),
}

/// Gemma converter error types
#[derive(Debug, Clone, PartialEq)]
pub enum NeuralGemmaError {
    ConfigReadError(String),
    ConfigParseError(String),
    TensorConversionError(String),
    TensorRepackError(String),
    IoError(String),
}

impl std::fmt::Display for NeuralGemmaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConfigReadError(e) => write!(f, "Config read error: {}", e),
            Self::ConfigParseError(e) => write!(f, "Config parse error: {}", e),
            Self::TensorConversionError(e) => write!(f, "Tensor conversion error: {}", e),
            Self::TensorRepackError(e) => write!(f, "Tensor repack error: {}", e),
            Self::IoError(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for NeuralGemmaError {}

impl From<io::Error> for NeuralGemmaError {
    fn from(e: io::Error) -> Self {
        NeuralGemmaError::IoError(e.to_string())
    }
}

/// Type aliases
pub type GemmaConverter = NeuralGemmaConverter;
pub type GemmaParams = NeuralGemmaModelParams;
pub type GemmaError = NeuralGemmaError;
pub type GemmaContext = Gemma2SpatialContext;
pub type GemmaArchitecture = NeuralGemmaArchitecture;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_context() {
        let ctx = GemmaContext::gemma_27b();
        assert_eq!(ctx.hidden_width, 6144);
        assert_eq!(ctx.transformer_height, 46);
        
        let metadata = ctx.to_metadata();
        assert_eq!(metadata.width, 6144);
    }

    #[test]
    fn test_architecture() {
        assert_eq!(NeuralGemmaArchitecture::Gemma.as_str(), "gemma");
        assert_eq!(NeuralGemmaArchitecture::Gemma2.as_str(), "gemma2");
        assert_eq!(NeuralGemmaArchitecture::from_str("gemma2"), NeuralGemmaArchitecture::Gemma2);
    }

    #[test]
    fn test_model_params() {
        let params = NeuralGemmaModelParams::gemma2();
        assert_eq!(params.hidden_size, 3584);
        assert_eq!(params.num_hidden_layers, 42);
        assert!(params.is_gemma2());
        assert!(params.has_softcapping());
        assert_eq!(params.effective_sliding_window(), 4096);
    }

    #[test]
    fn test_special_tokens() {
        let tokens = NeuralGemmaSpecialTokens::gemma2_default();
        assert_eq!(tokens.eot_token_id, 107);
        assert_eq!(tokens.middle_token_id, 68);
    }

    #[test]
    fn test_tensor_name_conversion() {
        let converter = NeuralGemmaConverter::new(NeuralGemmaModelParams::gemma2());
        
        assert_eq!(
            converter.convert_tensor_name("model.layers.0.self_attn.q_proj.weight").as_ref(),
            "blk.0.attn_q.weight"
        );
        assert_eq!(
            converter.convert_tensor_name("model.norm.weight").as_ref(),
            "output_norm.weight"
        );
    }

    #[test]
    fn test_replacements_gemma2() {
        let converter = NeuralGemmaConverter::new(NeuralGemmaModelParams::gemma2());
        let replacements = converter.get_replacements();
        
        // Check Gemma2-specific replacements
        assert!(replacements.iter().any(|(k, v)| k.as_ref() == "post_attention_layernorm" && v.as_ref() == "post_attention_norm"));
        assert!(replacements.iter().any(|(k, v)| k.as_ref() == "pre_feedforward_layernorm" && v.as_ref() == "ffn_norm"));
    }

    #[test]
    fn test_replacements_gemma1() {
        let converter = NeuralGemmaConverter::new(NeuralGemmaModelParams::gemma_2b());
        let replacements = converter.get_replacements();
        
        // Gemma1 uses simpler norm naming
        assert!(replacements.iter().any(|(k, v)| k.as_ref() == "post_attention_layernorm" && v.as_ref() == "ffn_norm"));
    }

    #[test]
    fn test_estimate_size() {
        let params = NeuralGemmaModelParams::gemma2();
        let converter = NeuralGemmaConverter::new(params);
        let size = converter.estimate_size();
        
        // Gemma2 9B should be ~9-10GB
        assert!(size > 8_000_000_000);
        assert!(size < 15_000_000_000);
    }

    #[test]
    fn test_attention_summary() {
        let converter = NeuralGemmaConverter::new(NeuralGemmaModelParams::gemma2());
        let summary = converter.attention_summary();
        
        assert!(summary.contains("Gemma2"));
        assert!(summary.contains("42 layers"));
        assert!(summary.contains("sliding_window=4096"));
    }

    #[test]
    fn test_spatial_coordinate() {
        let params = NeuralGemmaModelParams::gemma_27b();
        let coord = params.spatial_coordinate();
        assert_eq!(coord.0, 6144);
        assert_eq!(coord.1, 46);
        assert_eq!(coord.2, 4096);
    }

    #[test]
    fn test_repack_norm_weights() {
        let converter = NeuralGemmaConverter::new(NeuralGemmaModelParams::gemma2());
        let data = vec![0.0f32, 1.0, 2.0, 3.0];
        let shape = vec![4u64];
        
        let result = converter.repack_norm_weights("test_norm.weight", &data, &shape).unwrap();
        
        // Each element should have 1.0 added
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_error_display() {
        let err = NeuralGemmaError::TensorConversionError("invalid shape".to_string());
        assert_eq!(err.to_string(), "Tensor conversion error: invalid shape");
    }
}