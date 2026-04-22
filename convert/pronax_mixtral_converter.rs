
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::convert::pronax_converter_core::{
    ConversionCoordinate, NeuralMetadataKV, NeuralModelConverter, NeuralSourceTensor,
};
use crate::fs::ggml::pronax_ggml_format::NeuralGgmlTensor;
use crate::tokenizer::pronax_vocabulary::{NeuralConversionTokenizer, SpecialTokenType};

/// 3D Spatial coordinate for Mixtral MoE conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MixtralMoeCoordinate {
    pub expert_id: u32,
    pub layer_depth: u16,
    pub routing_score: f32,
    pub sparsity_factor: f32,
}

impl MixtralMoeCoordinate {
    pub const fn new(expert: u32, depth: u16, score: f32, sparsity: f32) -> Self {
        Self {
            expert_id: expert,
            layer_depth: depth,
            routing_score: score,
            sparsity_factor: sparsity,
        }
    }

    /// Router/gate coordinate
    pub const fn router() -> Self {
        Self::new(0, 1000, 0.999, 0.95)
    }

    /// Expert fusion coordinate
    pub const fn expert_fusion(expert_id: u32, layer: u16) -> Self {
        Self::new(expert_id, layer, 0.997, 0.90)
    }

    /// Active expert coordinate
    pub const fn active_expert(expert_id: u32) -> Self {
        Self::new(expert_id, 980, 0.995, 0.85)
    }

    /// Inactive/sparse expert coordinate
    pub const fn sparse_expert(expert_id: u32) -> Self {
        Self::new(expert_id, 960, 0.500, 0.10)
    }

    /// Base attention coordinate
    pub const fn attention_base() -> Self {
        Self::new(0, 990, 0.998, 1.0)
    }

    /// Calculate expert importance score
    pub fn expert_importance(&self) -> u64 {
        let expert_boost = self.expert_id as u64 * 50;
        let depth_boost = self.layer_depth as u64 * 100;
        let routing_boost = (self.routing_score * 2000.0) as u64;
        let sparsity_penalty = ((1.0 - self.sparsity_factor) * 1000.0) as u64;

        expert_boost + depth_boost + routing_boost - sparsity_penalty
    }

    /// Check if this expert is likely active
    pub fn is_active(&self) -> bool {
        self.routing_score > 0.8 && self.sparsity_factor > 0.5
    }
}

/// MoE expert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralMixtralMoeConfig {
    #[serde(rename = "num_local_experts")]
    pub num_local_experts: u32,
    #[serde(rename = "num_experts_per_tok")]
    pub num_experts_per_tok: u32,
    #[serde(rename = "expert_capacity_factor")]
    pub expert_capacity_factor: f32,
    #[serde(rename = "router_aux_loss_coef")]
    pub router_aux_loss_coef: f32,
    #[serde(rename = "router_z_loss_coef")]
    pub router_z_loss_coef: f32,
}

impl NeuralMixtralMoeConfig {
    pub fn new() -> Self {
        Self {
            num_local_experts: 8,
            num_experts_per_tok: 2,
            expert_capacity_factor: 1.0,
            router_aux_loss_coef: 0.001,
            router_z_loss_coef: 0.001,
        }
    }

    /// Create 8x7B configuration
    pub fn mixtral_8x7b() -> Self {
        Self::new()
    }

    /// Create 8x22B configuration
    pub fn mixtral_8x22b() -> Self {
        Self {
            num_local_experts: 8,
            num_experts_per_tok: 2,
            expert_capacity_factor: 1.25,
            router_aux_loss_coef: 0.001,
            router_z_loss_coef: 0.001,
        }
    }

    /// Calculate sparsity ratio (experts activated / total)
    pub fn sparsity_ratio(&self) -> f32 {
        self.num_experts_per_tok as f32 / self.num_local_experts as f32
    }

    /// Check if dense (all experts active)
    pub fn is_dense(&self) -> bool {
        self.num_experts_per_tok >= self.num_local_experts
    }

    /// Calculate effective parameter count
    pub fn effective_parameters(&self, expert_size: u64) -> u64 {
        let active_experts = self.num_experts_per_tok as u64;
        active_experts * expert_size
    }

    /// Get total experts count
    pub fn total_experts(&self) -> u32 {
        self.num_local_experts
    }
}

impl Default for NeuralMixtralMoeConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Expert tensor paths for MoE layers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertTensorType {
    Gate,  // w1 - routing gate
    Up,    // w2 - up projection
    Down,  // w3 - down projection
}

impl ExpertTensorType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    pub fn weight_suffix(&self) -> &'static str {
        match self {
            Self::Gate => "w1",
            Self::Up => "w2",
            Self::Down => "w3",
        }
    }

    pub fn merged_name(&self) -> &'static str {
        match self {
            Self::Gate => "ffn_gate_exps",
            Self::Up => "ffn_up_exps",
            Self::Down => "ffn_down_exps",
        }
    }
}

/// Expert merge specification
#[derive(Debug, Clone)]
pub struct ExpertMergeSpec {
    pub layer_idx: u32,
    pub expert_tensor: ExpertTensorType,
    pub has_bias: bool,
}

impl ExpertMergeSpec {
    pub fn new(layer: u32, tensor_type: ExpertTensorType, bias: bool) -> Self {
        Self {
            layer_idx: layer,
            expert_tensor: tensor_type,
            has_bias: bias,
        }
    }

    /// Generate source pattern for matching expert tensors
    pub fn source_pattern(&self) -> String {
        format!("blk.{}.*.{}", self.layer_idx, self.expert_tensor.weight_suffix())
    }

    /// Generate target name for merged tensor
    pub fn target_name(&self) -> String {
        if self.has_bias {
            format!("blk.{}.{}_exps.bias", self.layer_idx, self.expert_tensor.as_str())
        } else {
            format!("blk.{}.{}_exps.weight", self.layer_idx, self.expert_tensor.as_str())
        }
    }
}

/// Expert tensor merger for MoE layers
pub struct MixtralExpertMerger;

impl MixtralExpertMerger {
    /// Generate all merge specifications for a model
    pub fn generate_merge_specs(num_layers: u32) -> Vec<ExpertMergeSpec> {
        let mut specs = Vec::with_capacity((num_layers * 6) as usize);

        for layer in 0..num_layers {
            // Weights
            specs.push(ExpertMergeSpec::new(layer, ExpertTensorType::Gate, false));
            specs.push(ExpertMergeSpec::new(layer, ExpertTensorType::Up, false));
            specs.push(ExpertMergeSpec::new(layer, ExpertTensorType::Down, false));
            // Biases
            specs.push(ExpertMergeSpec::new(layer, ExpertTensorType::Gate, true));
            specs.push(ExpertMergeSpec::new(layer, ExpertTensorType::Up, true));
            specs.push(ExpertMergeSpec::new(layer, ExpertTensorType::Down, true));
        }

        specs
    }

    /// Merge expert tensors into combined tensor
    pub fn merge_expert_tensors(
        tensors: &[NeuralSourceTensor],
        num_experts: u32,
    ) -> Result<NeuralGgmlTensor, String> {
        if tensors.is_empty() {
            return Err("No tensors to merge".to_string());
        }

        if tensors.len() != num_experts as usize {
            return Err(format!(
                "Expected {} expert tensors, found {}",
                num_experts,
                tensors.len()
            ));
        }

        // Calculate merged shape: [num_experts, ...individual_shape]
        let first_shape = &tensors[0].shape;
        let mut merged_shape = vec![num_experts as u64];
        merged_shape.extend_from_slice(first_shape);

        // Merge data
        let mut merged_data = Vec::new();
        for tensor in tensors {
            merged_data.extend_from_slice(&tensor.data);
        }

        // Create merged tensor with expert coordinate
        let merged_tensor = NeuralGgmlTensor::new(
            tensors[0].name.clone(),
            tensors[0].data_type,
            merged_shape,
            merged_data,
        )
        .with_coordinate(ConversionCoordinate::new(
            0,
            1000,
            24,
            0.997,
        ));

        Ok(merged_tensor)
    }

    /// Check if tensor name matches expert pattern
    pub fn is_expert_tensor(name: &str) -> bool {
        name.contains("w1") || name.contains("w2") || name.contains("w3")
    }

    /// Extract layer index from expert tensor name
    pub fn extract_layer_idx(name: &str) -> Option<u32> {
        // Pattern: blk.{layer}.*.w{1,2,3}...
        if !name.starts_with("blk.") {
            return None;
        }

        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() < 2 {
            return None;
        }

        parts[1].parse().ok()
    }

    /// Extract expert tensor type from name
    pub fn extract_tensor_type(name: &str) -> Option<ExpertTensorType> {
        if name.contains("w1") {
            Some(ExpertTensorType::Gate)
        } else if name.contains("w2") {
            Some(ExpertTensorType::Up)
        } else if name.contains("w3") {
            Some(ExpertTensorType::Down)
        } else {
            None
        }
    }

    /// Check if tensor has bias
    pub fn has_bias(name: &str) -> bool {
        name.ends_with(".bias")
    }
}

/// Router/gate configuration for MoE
#[derive(Debug, Clone)]
pub struct MoeRouterConfig {
    pub num_experts: u32,
    pub top_k: u32,
    pub capacity_factor: f32,
    pub use_noise: bool,
}

impl MoeRouterConfig {
    pub fn new(num_experts: u32, top_k: u32) -> Self {
        Self {
            num_experts,
            top_k,
            capacity_factor: 1.0,
            use_noise: true,
        }
    }

    /// Calculate router capacity per expert
    pub fn expert_capacity(&self, batch_size: u32, seq_len: u32) -> u32 {
        let total_tokens = batch_size * seq_len;
        let tokens_per_expert = total_tokens / self.num_experts;
        (tokens_per_expert as f32 * self.capacity_factor).ceil() as u32
    }
}

/// Mixtral base configuration (extends Llama)
#[derive(Debug, Clone, Deserialize)]
pub struct NeuralMixtralBaseConfig {
    #[serde(rename = "num_hidden_layers")]
    pub num_hidden_layers: u32,
    #[serde(rename = "hidden_size")]
    pub hidden_size: u32,
    #[serde(rename = "intermediate_size")]
    pub intermediate_size: u32,
    #[serde(rename = "num_attention_heads")]
    pub num_attention_heads: u32,
    #[serde(rename = "num_key_value_heads")]
    pub num_key_value_heads: u32,
    #[serde(rename = "rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(rename = "vocab_size")]
    pub vocab_size: u32,
    #[serde(rename = "rope_theta")]
    pub rope_theta: f32,
}

impl NeuralMixtralBaseConfig {
    pub fn new() -> Self {
        Self {
            num_hidden_layers: 32,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-5,
            vocab_size: 32000,
            rope_theta: 10000.0,
        }
    }

    /// Create 8x7B configuration
    pub fn mixtral_8x7b() -> Self {
        Self::new()
    }

    /// Create 8x22B configuration
    pub fn mixtral_8x22b() -> Self {
        Self {
            num_hidden_layers: 56,
            hidden_size: 6144,
            intermediate_size: 16384,
            num_attention_heads: 48,
            num_key_value_heads: 16,
            rms_norm_eps: 1e-5,
            vocab_size: 32000,
            rope_theta: 10000.0,
        }
    }

    pub fn uses_gqa(&self) -> bool {
        self.num_key_value_heads > 0 && self.num_key_value_heads != self.num_attention_heads
    }
}

impl Default for NeuralMixtralBaseConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Mixtral model parameters
#[derive(Debug, Clone, Deserialize)]
pub struct NeuralMixtralParameters {
    #[serde(flatten)]
    pub base: NeuralMixtralBaseConfig,
    #[serde(flatten)]
    pub moe: NeuralMixtralMoeConfig,
}

impl NeuralMixtralParameters {
    pub fn new() -> Self {
        Self {
            base: NeuralMixtralBaseConfig::new(),
            moe: NeuralMixtralMoeConfig::new(),
        }
    }

    /// Create 8x7B model
    pub fn mixtral_8x7b() -> Self {
        Self {
            base: NeuralMixtralBaseConfig::mixtral_8x7b(),
            moe: NeuralMixtralMoeConfig::mixtral_8x7b(),
        }
    }

    /// Create 8x22B model
    pub fn mixtral_8x22b() -> Self {
        Self {
            base: NeuralMixtralBaseConfig::mixtral_8x22b(),
            moe: NeuralMixtralMoeConfig::mixtral_8x22b(),
        }
    }

    /// Calculate total parameters (all experts)
    pub fn total_parameters(&self) -> u64 {
        let vocab = self.base.vocab_size as u64;
        let hidden = self.base.hidden_size as u64;
        let intermediate = self.base.intermediate_size as u64;
        let layers = self.base.num_hidden_layers as u64;
        let experts = self.moe.num_local_experts as u64;

        // Base attention parameters
        let head_dim = hidden / self.base.num_attention_heads as u64;
        let q_proj = hidden * (self.base.num_attention_heads as u64 * head_dim);
        let k_proj = hidden * (self.base.num_key_value_heads as u64 * head_dim);
        let v_proj = k_proj;
        let o_proj = self.base.num_attention_heads as u64 * head_dim * hidden;

        // MoE FFN parameters (per layer)
        let expert_ffn = hidden * intermediate + intermediate * hidden; // gate + down (or up + down)
        let moe_ffn = experts * expert_ffn;

        // Router
        let router = hidden * experts;

        // Per layer total
        let per_layer = q_proj + k_proj + v_proj + o_proj + moe_ffn + router;

        // Embeddings and output
        let token_embd = vocab * hidden;
        let output = vocab * hidden;
        let norms = layers * hidden * 2; // 2 norms per layer

        token_embd + (layers * per_layer) + norms + output
    }

    /// Calculate active parameters per token
    pub fn active_parameters_per_token(&self) -> u64 {
        let total = self.total_parameters();
        let expert_ratio = self.moe.sparsity_ratio();
        (total as f64 * expert_ratio as f64) as u64
    }
}

impl Default for NeuralMixtralParameters {
    fn default() -> Self {
        Self::new()
    }
}

/// Mixtral converter
#[derive(Debug, Clone)]
pub struct NeuralMixtralConverter {
    params: NeuralMixtralParameters,
    coordinate: MixtralMoeCoordinate,
    tensor_replacements: HashMap<String, String>,
    merge_specs: Vec<ExpertMergeSpec>,
}

impl NeuralMixtralConverter {
    /// Create new converter
    pub fn new(params: NeuralMixtralParameters) -> Self {
        let merge_specs = MixtralExpertMerger::generate_merge_specs(params.base.num_hidden_layers);

        let mut converter = Self {
            params,
            coordinate: MixtralMoeCoordinate::router(),
            tensor_replacements: HashMap::new(),
            merge_specs,
        };

        converter.initialize_replacements();
        converter
    }

    /// Initialize tensor name replacements
    fn initialize_replacements(&mut self) {
        // Base model prefix
        self.tensor_replacements.insert("model.layers".to_string(), "blk".to_string());
        self.tensor_replacements.insert("model.".to_string(), "".to_string());

        // MoE router
        self.tensor_replacements.insert(
            "block_sparse_moe.gate".to_string(),
            "ffn_gate_inp".to_string(),
        );
        self.tensor_replacements.insert(
            "block_sparse_moe.experts.".to_string(),
            "".to_string(),
        );

        // Expert naming patterns (will be processed during merge)
        self.tensor_replacements.insert("w1.weight".to_string(), "ffn_gate_exps.weight".to_string());
        self.tensor_replacements.insert("w2.weight".to_string(), "ffn_up_exps.weight".to_string());
        self.tensor_replacements.insert("w3.weight".to_string(), "ffn_down_exps.weight".to_string());
        self.tensor_replacements.insert("w1.bias".to_string(), "ffn_gate_exps.bias".to_string());
        self.tensor_replacements.insert("w2.bias".to_string(), "ffn_up_exps.bias".to_string());
        self.tensor_replacements.insert("w3.bias".to_string(), "ffn_down_exps.bias".to_string());

        // Standard attention
        self.tensor_replacements.insert("self_attn.q_proj".to_string(), "attn_q".to_string());
        self.tensor_replacements.insert("self_attn.k_proj".to_string(), "attn_k".to_string());
        self.tensor_replacements.insert("self_attn.v_proj".to_string(), "attn_v".to_string());
        self.tensor_replacements.insert("self_attn.o_proj".to_string(), "attn_output".to_string());

        // Norms
        self.tensor_replacements.insert("input_layernorm".to_string(), "attn_norm".to_string());
        self.tensor_replacements.insert(
            "post_attention_layernorm".to_string(),
            "ffn_norm".to_string(),
        );

        // Embeddings
        self.tensor_replacements.insert("embed_tokens".to_string(), "token_embd".to_string());
        self.tensor_replacements.insert("norm".to_string(), "output_norm".to_string());

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

    /// Get all tensor replacements
    pub fn replacement_pairs(&self) -> Vec<(String, String)> {
        self.tensor_replacements
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Get merge specifications
    pub fn merge_specifications(&self) -> &[ExpertMergeSpec] {
        &self.merge_specs
    }

    /// Get coordinate for tensor
    fn coordinate_for_tensor(&self, name: &str) -> MixtralMoeCoordinate {
        if name.contains("ffn_gate_inp") || name.contains("router") {
            MixtralMoeCoordinate::router()
        } else if name.contains("exps") {
            // Extract expert info if available
            let layer = MixtralExpertMerger::extract_layer_idx(name).unwrap_or(0);
            MixtralMoeCoordinate::expert_fusion(0, layer as u16)
        } else if name.contains("attn") {
            MixtralMoeCoordinate::attention_base()
        } else {
            self.coordinate
        }
    }

    /// Get parameter info
    pub fn parameter_info(&self) -> MixtralParameterInfo {
        let base = &self.params.base;
        let moe = &self.params.moe;

        MixtralParameterInfo {
            num_layers: base.num_hidden_layers,
            hidden_size: base.hidden_size,
            intermediate_size: base.intermediate_size,
            num_attention_heads: base.num_attention_heads,
            num_kv_heads: base.num_key_value_heads,
            num_experts: moe.num_local_experts,
            experts_per_token: moe.num_experts_per_tok,
            sparsity_ratio: moe.sparsity_ratio(),
            uses_gqa: base.uses_gqa(),
            total_parameters: self.params.total_parameters(),
            active_parameters: self.params.active_parameters_per_token(),
            coordinate: self.coordinate,
        }
    }

    /// Generate model summary
    pub fn model_summary(&self) -> String {
        let info = self.parameter_info();
        let total_b = info.total_parameters as f64 / 1e9;
        let active_b = info.active_parameters as f64 / 1e9;

        format!(
            "Mixtral-MoE {}x{}B ({} active), {} layers, {} experts, {}x{} sparsity, {:.1}B total, {:.1}B active",
            info.num_experts,
            total_b / info.num_experts as f64,
            info.experts_per_token,
            info.num_layers,
            info.num_experts,
            info.num_experts,
            info.experts_per_token,
            total_b,
            active_b
        )
    }

    /// Group expert tensors by layer and type for merging
    pub fn group_expert_tensors(
        &self,
        tensors: &[NeuralSourceTensor],
    ) -> HashMap<(u32, ExpertTensorType, bool), Vec<NeuralSourceTensor>> {
        let mut groups: HashMap<(u32, ExpertTensorType, bool), Vec<NeuralSourceTensor>> = HashMap::new();

        for tensor in tensors {
            if let Some(layer) = MixtralExpertMerger::extract_layer_idx(&tensor.name) {
                if let Some(tensor_type) = MixtralExpertMerger::extract_tensor_type(&tensor.name) {
                    let has_bias = MixtralExpertMerger::has_bias(&tensor.name);
                    let key = (layer, tensor_type, has_bias);
                    groups.entry(key).or_default().push(tensor.clone());
                }
            }
        }

        groups
    }
}

/// Parameter info for Mixtral
#[derive(Debug, Clone)]
pub struct MixtralParameterInfo {
    pub num_layers: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub num_experts: u32,
    pub experts_per_token: u32,
    pub sparsity_ratio: f32,
    pub uses_gqa: bool,
    pub total_parameters: u64,
    pub active_parameters: u64,
    pub coordinate: MixtralMoeCoordinate,
}

impl NeuralModelConverter for NeuralMixtralConverter {
    fn to_metadata_kv(&self, tokenizer: &NeuralConversionTokenizer) -> NeuralMetadataKV {
        let mut kv = NeuralMetadataKV::new();
        let base = &self.params.base;
        let moe = &self.params.moe;

        // Base architecture
        kv.insert("general.architecture", "mixtral".to_string());
        kv.insert("general.type", "moe".to_string());

        // Model dimensions
        kv.insert("mixtral.block_count", base.num_hidden_layers);
        kv.insert("mixtral.embedding_length", base.hidden_size);
        kv.insert("mixtral.feed_forward_length", base.intermediate_size);
        kv.insert("mixtral.vocab_size", base.vocab_size);

        // Attention
        kv.insert("mixtral.attention.head_count", base.num_attention_heads);
        kv.insert("mixtral.attention.head_count_kv", base.num_key_value_heads);
        kv.insert("mixtral.attention.layer_norm_rms_epsilon", base.rms_norm_eps);

        // RoPE
        kv.insert("mixtral.rope.freq_base", base.rope_theta);

        // MoE configuration
        if moe.num_local_experts > 0 {
            kv.insert("mixtral.expert_count", moe.num_local_experts);
        }
        if moe.num_experts_per_tok > 0 {
            kv.insert("mixtral.expert_used_count", moe.num_experts_per_tok);
        }

        // GQA indicator
        if base.uses_gqa() {
            kv.insert("mixtral.attention.use_gqa", true);
        }

        // Coordinate metadata
        kv.insert("pronax.coordinate.expert_id", self.coordinate.expert_id);
        kv.insert("pronax.coordinate.layer_depth", self.coordinate.layer_depth);
        kv.insert("pronax.coordinate.routing_score", self.coordinate.routing_score);
        kv.insert("pronax.coordinate.sparsity", self.coordinate.sparsity_factor);

        // Tokenizer metadata
        let tokenizer_kv = tokenizer.to_kv();
        kv.merge(tokenizer_kv);

        kv.set_architecture("mixtral");
        kv
    }

    fn convert_tensors(&self, tensors: &[NeuralSourceTensor]) -> Vec<NeuralGgmlTensor> {
        let mut converted = Vec::new();

        // Group expert tensors for merging
        let expert_groups = self.group_expert_tensors(tensors);

        // Merge expert tensors
        for ((layer, tensor_type, has_bias), group) in expert_groups {
            if group.len() == self.params.moe.num_local_experts as usize {
                // All experts present - merge them
                let merged_name = format!(
                    "blk.{}.{}_exps.{}",
                    layer,
                    tensor_type.as_str(),
                    if has_bias { "bias" } else { "weight" }
                );

                let mut merged_data = Vec::new();
                let mut shape = vec![self.params.moe.num_local_experts as u64];
                if !group.is_empty() {
                    shape.extend_from_slice(&group[0].shape);
                    for tensor in &group {
                        merged_data.extend_from_slice(&tensor.data);
                    }
                }

                let coord = MixtralMoeCoordinate::expert_fusion(0, layer as u16);
                let merged_tensor = NeuralGgmlTensor::new(
                    merged_name,
                    group[0].data_type,
                    shape,
                    merged_data,
                )
                .with_coordinate(ConversionCoordinate::new(
                    layer as u64,
                    coord.layer_depth,
                    24,
                    coord.routing_score,
                ));

                converted.push(merged_tensor);
            } else {
                // Not all experts - add individually
                for tensor in group {
                    let name = self.replace_tensor_name(&tensor.name);
                    let coord = self.coordinate_for_tensor(&name);

                    let converted_tensor = NeuralGgmlTensor::new(
                        name,
                        tensor.data_type,
                        tensor.shape.clone(),
                        tensor.data.clone(),
                    )
                    .with_coordinate(ConversionCoordinate::new(
                        layer as u64,
                        coord.layer_depth,
                        24,
                        coord.routing_score,
                    ));

                    converted.push(converted_tensor);
                }
            }
        }

        // Process non-expert tensors
        for (idx, tensor) in tensors.iter().enumerate() {
            if !MixtralExpertMerger::is_expert_tensor(&tensor.name) {
                let name = self.replace_tensor_name(&tensor.name);
                let coord = self.coordinate_for_tensor(&name);

                let converted_tensor = NeuralGgmlTensor::new(
                    name,
                    tensor.data_type,
                    tensor.shape.clone(),
                    tensor.data.clone(),
                )
                .with_coordinate(ConversionCoordinate::new(
                    idx as u64,
                    coord.layer_depth,
                    24,
                    coord.routing_score,
                ));

                converted.push(converted_tensor);
            }
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
        "mixtral"
    }

    fn coordinate(&self) -> ConversionCoordinate {
        ConversionCoordinate::new(
            self.coordinate.expert_id as u64,
            self.coordinate.layer_depth,
            24,
            self.coordinate.routing_score,
        )
    }
}

/// Factory functions
pub fn create_mixtral_converter(params: NeuralMixtralParameters) -> NeuralMixtralConverter {
    NeuralMixtralConverter::new(params)
}

pub fn create_mixtral_8x7b_converter() -> NeuralMixtralConverter {
    NeuralMixtralConverter::new(NeuralMixtralParameters::mixtral_8x7b())
}

pub fn create_mixtral_8x22b_converter() -> NeuralMixtralConverter {
    NeuralMixtralConverter::new(NeuralMixtralParameters::mixtral_8x22b())
}

/// Calculate MoE routing overhead
pub fn calculate_routing_overhead(
    batch_size: u32,
    seq_len: u32,
    num_experts: u32,
    top_k: u32,
) -> u64 {
    // Softmax over experts + top-k selection
    let logits_size = batch_size as u64 * seq_len as u64 * num_experts as u64 * 4;
    let probs_size = logits_size;
    let topk_indices = batch_size as u64 * seq_len as u64 * top_k as u64 * 4;
    let topk_values = topk_indices;

    logits_size + probs_size + topk_indices + topk_values
}

/// Calculate expert capacity requirements
pub fn calculate_expert_capacity(
    batch_size: u32,
    seq_len: u32,
    num_experts: u32,
    top_k: u32,
    capacity_factor: f32,
) -> u32 {
    let total_tokens = batch_size * seq_len * top_k;
    let base_capacity = total_tokens / num_experts;
    (base_capacity as f32 * capacity_factor).ceil() as u32
}

/// Calculate memory savings from sparsity
pub fn calculate_sparse_memory_savings(
    dense_params: u64,
    num_experts: u32,
    experts_per_token: u32,
) -> f32 {
    let active_ratio = experts_per_token as f32 / num_experts as f32;
    1.0 - active_ratio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_coordinate() {
        let coord = MixtralMoeCoordinate::router();
        assert_eq!(coord.layer_depth, 1000);
        assert_eq!(coord.routing_score, 0.999);

        let importance = coord.expert_importance();
        assert!(importance > 0);
    }

    #[test]
    fn test_expert_coordinate() {
        let coord = MixtralMoeCoordinate::expert_fusion(3, 10);
        assert_eq!(coord.expert_id, 3);
        assert_eq!(coord.layer_depth, 10);
        assert!(coord.is_active());
    }

    #[test]
    fn test_sparse_expert() {
        let coord = MixtralMoeCoordinate::sparse_expert(5);
        assert!(!coord.is_active());
    }

    #[test]
    fn test_moe_config() {
        let config = NeuralMixtralMoeConfig::new();
        assert_eq!(config.num_local_experts, 8);
        assert_eq!(config.num_experts_per_tok, 2);
        assert!(!config.is_dense());

        let ratio = config.sparsity_ratio();
        assert_eq!(ratio, 0.25); // 2/8
    }

    #[test]
    fn test_moe_8x22b() {
        let config = NeuralMixtralMoeConfig::mixtral_8x22b();
        assert_eq!(config.num_local_experts, 8);
        assert!(config.expert_capacity_factor > 1.0);
    }

    #[test]
    fn test_expert_tensor_type() {
        assert_eq!(ExpertTensorType::Gate.weight_suffix(), "w1");
        assert_eq!(ExpertTensorType::Up.weight_suffix(), "w2");
        assert_eq!(ExpertTensorType::Down.weight_suffix(), "w3");
    }

    #[test]
    fn test_merge_spec() {
        let spec = ExpertMergeSpec::new(5, ExpertTensorType::Gate, false);
        assert_eq!(spec.layer_idx, 5);
        assert!(!spec.has_bias);
        assert!(spec.target_name().contains("ffn_gate_exps"));
    }

    #[test]
    fn test_expert_merger_patterns() {
        assert!(MixtralExpertMerger::is_expert_tensor("blk.0.3.w1.weight"));
        assert!(!MixtralExpertMerger::is_expert_tensor("blk.0.attn_q.weight"));

        let layer = MixtralExpertMerger::extract_layer_idx("blk.5.2.w2.weight");
        assert_eq!(layer, Some(5));

        let tensor_type = MixtralExpertMerger::extract_tensor_type("blk.0.1.w3.bias");
        assert_eq!(tensor_type, Some(ExpertTensorType::Down));
    }

    #[test]
    fn test_generate_merge_specs() {
        let specs = MixtralExpertMerger::generate_merge_specs(2);
        assert_eq!(specs.len(), 12); // 2 layers * 6 specs each
    }

    #[test]
    fn test_router_config() {
        let router = MoeRouterConfig::new(8, 2);
        assert_eq!(router.num_experts, 8);
        assert_eq!(router.top_k, 2);

        let capacity = router.expert_capacity(4, 1024);
        assert!(capacity > 0);
    }

    #[test]
    fn test_base_config() {
        let config = NeuralMixtralBaseConfig::new();
        assert!(config.uses_gqa());
        assert_eq!(config.num_key_value_heads, 8);
    }

    #[test]
    fn test_8x22b_config() {
        let config = NeuralMixtralBaseConfig::mixtral_8x22b();
        assert_eq!(config.num_hidden_layers, 56);
        assert_eq!(config.hidden_size, 6144);
    }

    #[test]
    fn test_parameters() {
        let params = NeuralMixtralParameters::mixtral_8x7b();
        let total = params.total_parameters();
        assert!(total > 40_000_000_000); // > 40B

        let active = params.active_parameters_per_token();
        assert!(active < total);
    }

    #[test]
    fn test_converter_creation() {
        let params = NeuralMixtralParameters::new();
        let converter = NeuralMixtralConverter::new(params);

        assert_eq!(converter.architecture(), "mixtral");
        assert_eq!(converter.coordinate().tier, 1000);
    }

    #[test]
    fn test_tensor_replacement() {
        let params = NeuralMixtralParameters::new();
        let converter = NeuralMixtralConverter::new(params);

        let replaced = converter.replace_tensor_name("model.layers.0.self_attn.q_proj.weight");
        assert!(replaced.contains("blk"));
        assert!(!replaced.contains("model.layers"));

        let router_replaced = converter.replace_tensor_name("block_sparse_moe.gate.weight");
        assert!(router_replaced.contains("ffn_gate_inp"));
    }

    #[test]
    fn test_parameter_info() {
        let converter = create_mixtral_8x7b_converter();
        let info = converter.parameter_info();

        assert_eq!(info.num_experts, 8);
        assert_eq!(info.experts_per_token, 2);
        assert!(info.uses_gqa);
        assert!(info.total_parameters > info.active_parameters);
    }

    #[test]
    fn test_model_summary() {
        let converter = create_mixtral_8x7b_converter();
        let summary = converter.model_summary();

        assert!(summary.contains("Mixtral-MoE"));
        assert!(summary.contains("8x"));
        assert!(summary.contains("experts"));
    }

    #[test]
    fn test_routing_overhead() {
        let overhead = calculate_routing_overhead(2, 512, 8, 2);
        assert!(overhead > 0);
    }

    #[test]
    fn test_expert_capacity() {
        let capacity = calculate_expert_capacity(4, 1024, 8, 2, 1.0);
        assert!(capacity > 0);
        assert_eq!(capacity, 1024); // 4*1024*2 / 8 = 1024
    }

    #[test]
    fn test_memory_savings() {
        let savings = calculate_sparse_memory_savings(47_000_000_000, 8, 2);
        assert!((savings - 0.75).abs() < 0.01); // 1 - 2/8 = 0.75
    }

    #[test]
    fn test_factory_functions() {
        let converter_7b = create_mixtral_8x7b_converter();
        let info_7b = converter_7b.parameter_info();
        assert_eq!(info_7b.hidden_size, 4096);

        let converter_22b = create_mixtral_8x22b_converter();
        let info_22b = converter_22b.parameter_info();
        assert_eq!(info_22b.hidden_size, 6144);
        assert_eq!(info_22b.num_layers, 56);
    }
}