
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::convert::pronax_converter_core::{
    ConversionCoordinate, NeuralMetadataKV, NeuralModelConverter, NeuralSourceTensor,
};
use crate::fs::ggml::pronax_ggml_format::NeuralGgmlTensor;
use crate::tokenizer::pronax_vocabulary::{NeuralConversionTokenizer, SpecialTokenType};

/// 3D Spatial coordinate for DeepSeekOCR (multi-modal)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeepSeekOcrCoordinate {
    pub modality_sequence: u64,
    pub fusion_tier: u16,
    pub vision_depth: u8,
    pub recognition_score: f32,
}

impl DeepSeekOcrCoordinate {
    pub const fn new(seq: u64, tier: u16, depth: u8, score: f32) -> Self {
        Self {
            modality_sequence: seq,
            fusion_tier: tier,
            vision_depth: depth,
            recognition_score: score,
        }
    }

    pub const fn language() -> Self {
        Self::new(0, 800, 8, 0.98)
    }

    pub const fn vision() -> Self {
        Self::new(0, 900, 10, 0.99)
    }

    pub const fn sam() -> Self {
        Self::new(0, 850, 9, 0.985)
    }

    pub const fn experts() -> Self {
        Self::new(0, 700, 7, 0.97)
    }

    pub const fn projector() -> Self {
        Self::new(0, 750, 8, 0.975)
    }

    /// Calculate multi-modal importance
    pub fn importance_score(&self) -> u64 {
        let seq_factor = 1000u64.saturating_sub(self.modality_sequence);
        let tier_boost = self.fusion_tier as u64 * 100;
        let depth_norm = self.vision_depth as u64 * 10;
        let recognition_boost = (self.recognition_score * 1000.0) as u64;

        seq_factor + tier_boost + depth_norm + recognition_boost
    }
}

/// DeepSeekOCR language configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekLanguageConfig {
    #[serde(rename = "max_position_embeddings")]
    pub max_position_embeddings: u32,

    #[serde(rename = "hidden_size")]
    pub hidden_size: u32,

    #[serde(rename = "num_hidden_layers")]
    pub num_hidden_layers: u32,

    #[serde(rename = "intermediate_size")]
    pub intermediate_size: u32,

    #[serde(rename = "num_attention_heads")]
    pub num_attention_heads: u32,

    #[serde(rename = "num_key_value_heads")]
    pub num_key_value_heads: u32,

    // MoE-specific fields
    #[serde(rename = "n_routed_experts")]
    pub num_routed_experts: u32,

    #[serde(rename = "n_shared_experts")]
    pub num_shared_experts: u32,

    #[serde(rename = "num_experts_per_tok")]
    pub num_experts_per_token: u32,

    #[serde(rename = "first_k_dense_replace")]
    pub first_k_dense_replace: u32,
}

/// CLIP vision configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipVisionConfig {
    pub heads: u32,

    #[serde(rename = "image_size")]
    pub image_size: u32,

    pub layers: u32,

    #[serde(rename = "patch_size")]
    pub patch_size: u32,

    pub width: u32,
}

/// SAM (Segment Anything Model) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamVisionConfig {
    #[serde(rename = "global_attn_indexes")]
    pub global_attention_indexes: Vec<i32>,

    pub heads: u32,
    pub layers: u32,
    pub width: u32,
}

/// Vision configuration container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekVisionConfig {
    #[serde(rename = "image_size")]
    pub image_size: u32,

    pub width: DeepSeekVisionWidth,
}

/// Vision width configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekVisionWidth {
    #[serde(rename = "clip-l-14-224")]
    pub vision: ClipVisionConfig,

    #[serde(rename = "sam_vit_b")]
    pub sam: SamVisionConfig,
}

/// DeepSeekOCR model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralDeepSeekOcrParameters {
    // Base parameters
    pub architectures: Vec<String>,
    pub vocab_size: u32,
    pub model_type: Option<String>,

    // Language model (MoE)
    #[serde(rename = "language_config")]
    pub language_config: DeepSeekLanguageConfig,

    // Vision encoders
    #[serde(rename = "vision_config")]
    pub vision_config: DeepSeekVisionConfig,
}

impl NeuralDeepSeekOcrParameters {
    /// Create with default values
    pub fn new() -> Self {
        Self {
            architectures: vec!["DeepseekOCRForCausalLM".to_string()],
            vocab_size: 32000,
            model_type: Some("deepseekocr".to_string()),
            language_config: DeepSeekLanguageConfig {
                max_position_embeddings: 4096,
                hidden_size: 4096,
                num_hidden_layers: 32,
                intermediate_size: 11008,
                num_attention_heads: 32,
                num_key_value_heads: 32,
                num_routed_experts: 64,
                num_shared_experts: 2,
                num_experts_per_token: 6,
                first_k_dense_replace: 1,
            },
            vision_config: DeepSeekVisionConfig {
                image_size: 224,
                width: DeepSeekVisionWidth {
                    vision: ClipVisionConfig {
                        heads: 16,
                        image_size: 224,
                        layers: 24,
                        patch_size: 14,
                        width: 1024,
                    },
                    sam: SamVisionConfig {
                        global_attention_indexes: vec![2, 5, 8, 11],
                        heads: 12,
                        layers: 12,
                        width: 768,
                    },
                },
            },
        }
    }

    /// Check if using MoE
    pub fn uses_moe(&self) -> bool {
        self.language_config.num_routed_experts > 0
    }

    /// Check if using GQA
    pub fn uses_gqa(&self) -> bool {
        self.language_config.num_key_value_heads < self.language_config.num_attention_heads
    }

    /// Get GQA factor
    pub fn gqa_factor(&self) -> u32 {
        if self.uses_gqa() {
            self.language_config.num_attention_heads / self.language_config.num_key_value_heads
        } else {
            1
        }
    }

    /// Calculate number of vision patches
    pub fn vision_patches(&self) -> u32 {
        let image_size = self.vision_config.width.vision.image_size;
        let patch_size = self.vision_config.width.vision.patch_size;
        (image_size / patch_size) * (image_size / patch_size)
    }

    /// Estimate total parameters
    pub fn estimate_parameters(&self) -> u64 {
        // Language model (MoE)
        let vocab_params = self.vocab_size as u64 * self.language_config.hidden_size as u64;
        let attention_params = self.language_config.num_hidden_layers as u64
            * self.language_config.hidden_size as u64
            * self.language_config.hidden_size as u64
            * 4;

        // MoE FFN (sparse)
        let expert_params = self.language_config.num_hidden_layers as u64
            * self.language_config.num_routed_experts as u64
            * self.language_config.hidden_size as u64
            * self.language_config.intermediate_size as u64
            * 3;

        // Shared experts
        let shared_expert_params = self.language_config.num_hidden_layers as u64
            * self.language_config.num_shared_experts as u64
            * self.language_config.hidden_size as u64
            * self.language_config.intermediate_size as u64
            * 3;

        // Vision encoder (CLIP)
        let vision_params = self.vision_config.width.vision.layers as u64
            * self.vision_config.width.vision.width as u64
            * self.vision_config.width.vision.width as u64
            * 4;

        // SAM encoder
        let sam_params = self.vision_config.width.sam.layers as u64
            * self.vision_config.width.sam.width as u64
            * self.vision_config.width.sam.width as u64
            * 4;

        vocab_params + attention_params + expert_params + shared_expert_params + vision_params + sam_params
    }

    /// Get expert merge configuration
    pub fn expert_merges(&self) -> Vec<(String, String)> {
        let mut merges = Vec::new();

        for layer in 0..self.language_config.num_hidden_layers {
            merges.push((
                format!("blk.{}.mlp.experts.*.gate_proj.weight", layer),
                format!("blk.{}.ffn_gate_exps.weight", layer),
            ));
            merges.push((
                format!("blk.{}.mlp.experts.*.up_proj.weight", layer),
                format!("blk.{}.ffn_up_exps.weight", layer),
            ));
            merges.push((
                format!("blk.{}.mlp.experts.*.down_proj.weight", layer),
                format!("blk.{}.ffn_down_exps.weight", layer),
            ));
        }

        merges
    }
}

impl Default for NeuralDeepSeekOcrParameters {
    fn default() -> Self {
        Self::new()
    }
}

/// DeepSeekOCR model converter
#[derive(Debug, Clone)]
pub struct NeuralDeepSeekOcrConverter {
    params: NeuralDeepSeekOcrParameters,
    coordinate: DeepSeekOcrCoordinate,
    tensor_replacements: HashMap<String, String>,
}

impl NeuralDeepSeekOcrConverter {
    /// Create new converter
    pub fn new(params: NeuralDeepSeekOcrParameters) -> Self {
        let mut converter = Self {
            params,
            coordinate: DeepSeekOcrCoordinate::language(),
            tensor_replacements: HashMap::new(),
        };

        converter.initialize_replacements();
        converter
    }

    /// Initialize tensor name replacements
    fn initialize_replacements(&mut self) {
        // Language model embeddings
        self.tensor_replacements
            .insert("model.embed_tokens".to_string(), "token_embd".to_string());
        self.tensor_replacements
            .insert("model.layers".to_string(), "blk".to_string());

        // Normalization
        self.tensor_replacements
            .insert("input_layernorm".to_string(), "attn_norm".to_string());
        self.tensor_replacements
            .insert("post_attention_layernorm".to_string(), "ffn_norm".to_string());

        // Attention projections
        self.tensor_replacements
            .insert("self_attn.q_proj".to_string(), "attn_q".to_string());
        self.tensor_replacements
            .insert("self_attn.k_proj".to_string(), "attn_k".to_string());
        self.tensor_replacements
            .insert("self_attn.v_proj".to_string(), "attn_v".to_string());
        self.tensor_replacements
            .insert("self_attn.o_proj".to_string(), "attn_output".to_string());

        // FFN (dense)
        self.tensor_replacements
            .insert("mlp.gate_proj".to_string(), "ffn_gate".to_string());
        self.tensor_replacements
            .insert("mlp.up_proj".to_string(), "ffn_up".to_string());
        self.tensor_replacements
            .insert("mlp.down_proj".to_string(), "ffn_down".to_string());

        // MoE gate
        self.tensor_replacements
            .insert("mlp.gate".to_string(), "ffn_gate_inp".to_string());

        // Shared experts
        self.tensor_replacements.insert(
            "mlp.shared_experts.gate_proj".to_string(),
            "ffn_gate_shexp".to_string(),
        );
        self.tensor_replacements.insert(
            "mlp.shared_experts.up_proj".to_string(),
            "ffn_up_shexp".to_string(),
        );
        self.tensor_replacements.insert(
            "mlp.shared_experts.down_proj".to_string(),
            "ffn_down_shexp".to_string(),
        );

        // Output layers
        self.tensor_replacements
            .insert("model.norm".to_string(), "output_norm".to_string());
        self.tensor_replacements
            .insert("lm_head".to_string(), "output".to_string());

        // Vision model (CLIP)
        self.tensor_replacements
            .insert("model.vision_model".to_string(), "v".to_string());
        self.tensor_replacements.insert(
            "embeddings.patch_embedding".to_string(),
            "patch_embd".to_string(),
        );
        self.tensor_replacements.insert(
            "embeddings.class_embedding".to_string(),
            "class_embd".to_string(),
        );
        self.tensor_replacements.insert(
            "embeddings.position_embedding".to_string(),
            "position_embd".to_string(),
        );
        self.tensor_replacements
            .insert("transformer.layers".to_string(), "blk".to_string());

        // Multi-modal projector
        self.tensor_replacements
            .insert("model.projector".to_string(), "mm".to_string());
        self.tensor_replacements.insert(
            "model.image_newline".to_string(),
            "mm.image_newline".to_string(),
        );
        // Note: "view_seperator" is misspelled in upstream, keeping for compatibility
        self.tensor_replacements.insert(
            "model.view_seperator".to_string(),
            "mm.view_seperator".to_string(),
        );

        // SAM model
        self.tensor_replacements.insert(
            "model.sam_model.patch_embed.proj".to_string(),
            "s.patch_embd".to_string(),
        );
        self.tensor_replacements.insert(
            "model.sam_model.pos_embed".to_string(),
            "s.position_embd".to_string(),
        );
        self.tensor_replacements
            .insert("model.sam_model.blocks".to_string(), "s.blk".to_string());
        self.tensor_replacements
            .insert("model.sam_model.neck".to_string(), "s.neck".to_string());
        self.tensor_replacements
            .insert("model.sam_model.net_".to_string(), "s.net_".to_string());
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

    /// Get parameter info
    pub fn parameter_info(&self) -> DeepSeekOcrParameterInfo {
        DeepSeekOcrParameterInfo {
            vocab_size: self.params.vocab_size,
            hidden_size: self.params.language_config.hidden_size,
            num_layers: self.params.language_config.num_hidden_layers,
            num_heads: self.params.language_config.num_attention_heads,
            num_kv_heads: self.params.language_config.num_key_value_heads,
            num_routed_experts: self.params.language_config.num_routed_experts,
            num_shared_experts: self.params.language_config.num_shared_experts,
            num_experts_per_token: self.params.language_config.num_experts_per_token,
            image_size: self.params.vision_config.image_size,
            vision_layers: self.params.vision_config.width.vision.layers,
            sam_layers: self.params.vision_config.width.sam.layers,
            estimated_params: self.params.estimate_parameters(),
            uses_moe: self.params.uses_moe(),
            uses_gqa: self.params.uses_gqa(),
            coordinate: self.coordinate,
        }
    }
}

/// DeepSeekOCR parameter info summary
#[derive(Debug, Clone)]
pub struct DeepSeekOcrParameterInfo {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub num_routed_experts: u32,
    pub num_shared_experts: u32,
    pub num_experts_per_token: u32,
    pub image_size: u32,
    pub vision_layers: u32,
    pub sam_layers: u32,
    pub estimated_params: u64,
    pub uses_moe: bool,
    pub uses_gqa: bool,
    pub coordinate: DeepSeekOcrCoordinate,
}

impl DeepSeekOcrParameterInfo {
    /// Format as human-readable summary
    pub fn format_summary(&self) -> String {
        let params_b = self.estimated_params as f64 / 1e9;

        format!(
            "DeepSeekOCR {}B ({} layers, {} hidden, {} routed experts, {} vision layers, {} SAM layers)",
            params_b,
            self.num_layers,
            self.hidden_size,
            self.num_routed_experts,
            self.vision_layers,
            self.sam_layers
        )
    }

    /// Check if large model (> 50B)
    pub fn is_large_model(&self) -> bool {
        self.estimated_params > 50_000_000_000
    }

    /// Get memory estimate (GB) for FP16
    pub fn memory_estimate_gb(&self) -> f64 {
        (self.estimated_params as f64 * 2.0) / (1024.0 * 1024.0 * 1024.0)
    }
}

impl NeuralModelConverter for NeuralDeepSeekOcrConverter {
    fn to_metadata_kv(&self, tokenizer: &NeuralConversionTokenizer) -> NeuralMetadataKV {
        let mut kv = NeuralMetadataKV::new();

        // Base metadata
        kv.insert("general.architecture", "deepseekocr".to_string());
        kv.insert("general.name", "deepseekocr".to_string());
        kv.insert("general.file_type", 1u32);
        kv.insert("general.quantization_version", 2u32);

        // Language model metadata
        let lang = &self.params.language_config;
        kv.insert("block_count", lang.num_hidden_layers);
        kv.insert("context_length", lang.max_position_embeddings);
        kv.insert("embedding_length", lang.hidden_size);
        kv.insert("feed_forward_length", lang.intermediate_size);
        kv.insert("attention.head_count", lang.num_attention_heads);
        kv.insert("attention.head_count_kv", lang.num_key_value_heads);

        // MoE metadata
        if self.params.uses_moe() {
            kv.insert("expert_count", lang.num_routed_experts);
            kv.insert("expert_used_count", lang.num_experts_per_token);
            kv.insert("shared_expert_count", lang.num_shared_experts);
            kv.insert("leading_dense_block_count", lang.first_k_dense_replace);
        }

        // GQA metadata
        if self.params.uses_gqa() {
            kv.insert("attention.gqa_factor", self.params.gqa_factor());
        }

        // Vision metadata (CLIP)
        let vision = &self.params.vision_config.width.vision;
        kv.insert("vision.block_count", vision.layers);
        kv.insert("vision.embedding_length", vision.width);
        kv.insert("vision.head_count", vision.heads);
        kv.insert("vision.image_size", vision.image_size);
        kv.insert("vision.patch_size", vision.patch_size);

        // SAM metadata
        let sam = &self.params.vision_config.width.sam;
        kv.insert("sam.block_count", sam.layers);
        kv.insert("sam.embedding_length", sam.width);
        kv.insert("sam.head_count", sam.heads);
        kv.insert("sam.global_attention_indexes", sam.global_attention_indexes.clone());

        // Multi-modal metadata
        kv.insert("multimodal.projector_type", "mlp".to_string());
        kv.insert("multimodal.image_token_index", 151655u32); // DeepSeekOCR specific

        // Add coordinate metadata
        kv.insert("pronax.coordinate.sequence", self.coordinate.modality_sequence);
        kv.insert("pronax.coordinate.tier", self.coordinate.fusion_tier);
        kv.insert("pronax.coordinate.depth", self.coordinate.vision_depth);
        kv.insert("pronax.coordinate.recognition", self.coordinate.recognition_score);

        // Tokenizer metadata
        let tokenizer_kv = tokenizer.to_kv();
        kv.merge(tokenizer_kv);

        kv.set_architecture("deepseekocr");
        kv
    }

    fn convert_tensors(&self, tensors: &[NeuralSourceTensor]) -> Vec<NeuralGgmlTensor> {
        let mut converted = Vec::with_capacity(tensors.len());

        // Apply expert merges first
        let expert_merges = self.params.expert_merges();
        // In actual implementation, merge tensors here

        for (idx, tensor) in tensors.iter().enumerate() {
            // Apply name replacements
            let new_name = self.replace_tensor_name(&tensor.name);

            // Determine coordinate based on tensor type
            let coordinate = if new_name.starts_with("v.") {
                DeepSeekOcrCoordinate::vision()
            } else if new_name.starts_with("s.") {
                DeepSeekOcrCoordinate::sam()
            } else if new_name.contains("exps") {
                DeepSeekOcrCoordinate::experts()
            } else if new_name.starts_with("mm.") {
                DeepSeekOcrCoordinate::projector()
            } else {
                DeepSeekOcrCoordinate::language()
            };

            // Create converted tensor
            let converted_tensor = NeuralGgmlTensor::new(
                new_name,
                tensor.data_type,
                tensor.shape.clone(),
                tensor.data.clone(),
            )
            .with_coordinate(crate::convert::pronax_converter_core::ConversionCoordinate::new(
                idx as u64,
                coordinate.fusion_tier,
                coordinate.vision_depth,
                coordinate.recognition_score,
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
            SpecialTokenType::Unk,
            SpecialTokenType::Pad,
            SpecialTokenType::Mask,
            // OCR-specific
            SpecialTokenType::Image,
            SpecialTokenType::User,
            SpecialTokenType::Assistant,
        ]
    }

    fn architecture(&self) -> &str {
        "deepseekocr"
    }

    fn coordinate(&self) -> ConversionCoordinate {
        ConversionCoordinate::new(
            self.coordinate.modality_sequence,
            self.coordinate.fusion_tier,
            self.coordinate.vision_depth,
            self.coordinate.recognition_score,
        )
    }
}

/// DeepSeekOCR-specific tokenizer handling
pub struct DeepSeekOcrTokenizerHandler;

impl DeepSeekOcrTokenizerHandler {
    /// Get special token IDs for DeepSeekOCR
    pub fn special_token_ids() -> HashMap<String, u32> {
        let mut tokens = HashMap::new();

        // Standard tokens
        tokens.insert("<s>".to_string(), 1);
        tokens.insert("</s>".to_string(), 2);
        tokens.insert("<unk>".to_string(), 0);
        tokens.insert("<pad>".to_string(), 32000);

        // Multi-modal tokens
        tokens.insert("<image>".to_string(), 151655);
        tokens.insert("<image_pad>".to_string(), 151656);
        tokens.insert("<image_newline>".to_string(), 151657);

        // Role tokens
        tokens.insert("<|User|>".to_string(), 151646);
        tokens.insert("<|Assistant|>".to_string(), 151647);

        tokens
    }

    /// Get image token ID
    pub fn image_token_id() -> u32 {
        151655
    }

    /// Get chat template for DeepSeekOCR
    pub fn chat_template() -> String {
        r#"{{bos_token}}{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|User|>' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<|Assistant|>' + message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|Assistant|>' }}{% endif %}"#.to_string()
    }
}

/// Factory function to create DeepSeekOCR converter
pub fn create_deepseekocr_converter(params: NeuralDeepSeekOcrParameters) -> NeuralDeepSeekOcrConverter {
    NeuralDeepSeekOcrConverter::new(params)
}

/// Convenience function with default parameters
pub fn create_default_deepseekocr_converter() -> NeuralDeepSeekOcrConverter {
    NeuralDeepSeekOcrConverter::new(NeuralDeepSeekOcrParameters::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseekocr_coordinate() {
        let coord = DeepSeekOcrCoordinate::vision();
        assert_eq!(coord.fusion_tier, 900);
        assert_eq!(coord.recognition_score, 0.99);

        let score = coord.importance_score();
        assert!(score > 0);
    }

    #[test]
    fn test_deepseekocr_parameters() {
        let params = NeuralDeepSeekOcrParameters::new();

        assert_eq!(params.language_config.hidden_size, 4096);
        assert_eq!(params.language_config.num_routed_experts, 64);
        assert!(params.uses_moe());

        // Vision config
        assert_eq!(params.vision_config.width.vision.layers, 24);
        assert_eq!(params.vision_config.width.sam.layers, 12);
    }

    #[test]
    fn test_parameter_info() {
        let params = NeuralDeepSeekOcrParameters::new();
        let converter = NeuralDeepSeekOcrConverter::new(params);
        let info = converter.parameter_info();

        assert_eq!(info.num_routed_experts, 64);
        assert!(info.uses_moe);
        assert!(info.estimated_params > 10_000_000_000); // > 10B
    }

    #[test]
    fn test_tensor_replacement() {
        let params = NeuralDeepSeekOcrParameters::new();
        let converter = NeuralDeepSeekOcrConverter::new(params);

        let replaced = converter.replace_tensor_name("model.vision_model.embeddings.patch_embedding");
        assert!(replaced.contains("v."));
        assert!(replaced.contains("patch_embd"));

        let sam_replaced = converter.replace_tensor_name("model.sam_model.patch_embed.proj");
        assert!(sam_replaced.contains("s.patch_embd"));
    }

    #[test]
    fn test_expert_merges() {
        let params = NeuralDeepSeekOcrParameters::new();
        let merges = params.expert_merges();

        // Should have 3 merges per layer
        let expected_count = params.language_config.num_hidden_layers as usize * 3;
        assert_eq!(merges.len(), expected_count);

        // Check first merge
        let (from, to) = &merges[0];
        assert!(from.contains("gate_proj"));
        assert!(to.contains("ffn_gate_exps"));
    }

    #[test]
    fn test_vision_patches() {
        let params = NeuralDeepSeekOcrParameters::new();
        let patches = params.vision_patches();

        // 224 / 14 = 16, so 16 * 16 = 256 patches
        assert_eq!(patches, 256);
    }

    #[test]
    fn test_special_tokens() {
        let tokens = DeepSeekOcrTokenizerHandler::special_token_ids();

        assert!(tokens.contains_key("<image>"));
        assert_eq!(tokens["<image>"], 151655);
        assert!(tokens.contains_key("<|User|>"));
    }

    #[test]
    fn test_image_token_id() {
        assert_eq!(DeepSeekOcrTokenizerHandler::image_token_id(), 151655);
    }

    #[test]
    fn test_chat_template() {
        let template = DeepSeekOcrTokenizerHandler::chat_template();

        assert!(template.contains("User"));
        assert!(template.contains("Assistant"));
        assert!(template.contains("bos_token"));
    }

    #[test]
    fn test_converter_trait() {
        let params = NeuralDeepSeekOcrParameters::new();
        let converter = NeuralDeepSeekOcrConverter::new(params);

        assert_eq!(converter.architecture(), "deepseekocr");

        let special_types = converter.special_token_types();
        assert!(special_types.contains(&SpecialTokenType::Image));
    }

    #[test]
    fn test_memory_estimate() {
        let params = NeuralDeepSeekOcrParameters::new();
        let converter = NeuralDeepSeekOcrConverter::new(params);
        let info = converter.parameter_info();

        let memory_gb = info.memory_estimate_gb();
        assert!(memory_gb > 5.0); // > 5GB for FP16
    }
}