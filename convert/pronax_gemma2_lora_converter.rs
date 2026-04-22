
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::convert::pronax_converter_core::{
    ConversionCoordinate, NeuralAdapterConverter, NeuralMetadataKV, NeuralSourceTensor,
};
use crate::fs::ggml::pronax_ggml_format::NeuralGgmlTensor;
use crate::tokenizer::pronax_vocabulary::NeuralConversionTokenizer;

/// 3D Spatial coordinate for Gemma2 LoRA conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Gemma2LoraCoordinate {
    pub adapter_sequence: u64,
    pub rank_tier: u16,
    pub projection_depth: u8,
    pub adaptation_score: f32,
}

impl Gemma2LoraCoordinate {
    pub const fn new(seq: u64, tier: u16, depth: u8, score: f32) -> Self {
        Self {
            adapter_sequence: seq,
            rank_tier: tier,
            projection_depth: depth,
            adaptation_score: score,
        }
    }

    pub const fn lora_a() -> Self {
        Self::new(0, 600, 6, 0.96)
    }

    pub const fn lora_b() -> Self {
        Self::new(0, 700, 7, 0.97)
    }

    pub const fn attention() -> Self {
        Self::new(0, 800, 8, 0.98)
    }

    pub const fn feed_forward() -> Self {
        Self::new(0, 750, 7, 0.975)
    }

    /// Calculate LoRA-specific importance
    pub fn importance_score(&self) -> u64 {
        let seq_factor = 1000u64.saturating_sub(self.adapter_sequence);
        let tier_boost = self.rank_tier as u64 * 100;
        let depth_norm = self.projection_depth as u64 * 10;
        let adaptation_boost = (self.adaptation_score * 1000.0) as u64;

        seq_factor + tier_boost + depth_norm + adaptation_boost
    }
}

/// LoRA adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralLoraAdapterConfig {
    pub lora_rank: u32,
    pub lora_alpha: f32,
    pub lora_dropout: f32,
    pub target_modules: Vec<String>,
    pub bias: String,
}

impl NeuralLoraAdapterConfig {
    /// Create default LoRA config
    pub fn new() -> Self {
        Self {
            lora_rank: 8,
            lora_alpha: 16.0,
            lora_dropout: 0.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            bias: "none".to_string(),
        }
    }

    /// Calculate scaling factor (alpha / rank)
    pub fn scaling_factor(&self) -> f32 {
        self.lora_alpha / self.lora_rank as f32
    }

    /// Check if module is targeted
    pub fn targets_module(&self, module_name: &str) -> bool {
        self.target_modules.iter().any(|m| module_name.contains(m))
    }
}

impl Default for NeuralLoraAdapterConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Gemma2 LoRA adapter parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralGemma2LoraParameters {
    // Base adapter parameters
    pub adapter_name: String,
    pub base_model_name: String,
    pub adapter_version: String,

    // LoRA configuration
    pub lora_config: NeuralLoraAdapterConfig,

    // Gemma2-specific
    pub architecture: String,
}

impl NeuralGemma2LoraParameters {
    /// Create with default values
    pub fn new() -> Self {
        Self {
            adapter_name: "gemma2-lora-adapter".to_string(),
            base_model_name: "google/gemma-2-9b-it".to_string(),
            adapter_version: "1.0".to_string(),
            lora_config: NeuralLoraAdapterConfig::new(),
            architecture: "gemma2".to_string(),
        }
    }

    /// Create for specific base model
    pub fn for_model(model_name: &str) -> Self {
        let mut params = Self::new();
        params.base_model_name = model_name.to_string();
        params
    }
}

impl Default for NeuralGemma2LoraParameters {
    fn default() -> Self {
        Self::new()
    }
}

/// LoRA tensor shape validator
#[derive(Debug, Clone)]
pub struct LoraShapeValidator;

impl LoraShapeValidator {
    /// Check if tensor shape needs transposition
    /// LoRA-A should have shape [in_dim, rank] where in_dim > rank
    /// LoRA-B should have shape [rank, out_dim] where rank < out_dim
    pub fn needs_transpose(name: &str, shape: &[u64]) -> bool {
        if shape.len() < 2 {
            return false;
        }

        let dim0 = shape[0];
        let dim1 = shape[1];

        if Self::is_lora_a(name) {
            // LoRA-A: shape should be [in_dim, rank], if [rank, in_dim] then transpose
            dim0 < dim1
        } else if Self::is_lora_b(name) {
            // LoRA-B: shape should be [rank, out_dim], if [out_dim, rank] then transpose
            dim0 > dim1
        } else {
            false
        }
    }

    /// Check if tensor is LoRA-A matrix
    fn is_lora_a(name: &str) -> bool {
        name.ends_with("weight.lora_a") || name.ends_with("lora_a") || name.contains("lora_A")
    }

    /// Check if tensor is LoRA-B matrix
    fn is_lora_b(name: &str) -> bool {
        name.ends_with("weight.lora_b") || name.ends_with("lora_b") || name.contains("lora_B")
    }

    /// Get expected shape after transpose
    pub fn transposed_shape(shape: &[u64]) -> Vec<u64> {
        if shape.len() < 2 {
            return shape.to_vec();
        }
        let mut result = shape.to_vec();
        result.swap(0, 1);
        result
    }
}

/// Tensor repacker for LoRA matrices
pub struct LoraTensorRepacker;

impl LoraTensorRepacker {
    /// Repack tensor data by transposing 2D matrix
    /// Converts row-major to column-major or vice versa
    pub fn repack_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; data.len()];

        for i in 0..rows {
            for j in 0..cols {
                // Row-major to column-major: result[j][i] = data[i][j]
                result[j * rows + i] = data[i * cols + j];
            }
        }

        result
    }

    /// Repack bytes directly
    pub fn repack_bytes(data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
        let float_data = Self::bytes_to_f32_slice(data);
        let repacked = Self::repack_2d(&float_data, rows, cols);
        Self::f32_slice_to_bytes(&repacked)
    }

    /// Helper: Convert bytes to f32 slice
    fn bytes_to_f32_slice(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(arr)
            })
            .collect()
    }

    /// Helper: Convert f32 slice to bytes
    fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|&f| f.to_le_bytes().to_vec()).collect()
    }
}

/// Gemma2 LoRA adapter converter
#[derive(Debug, Clone)]
pub struct NeuralGemma2LoraConverter {
    params: NeuralGemma2LoraParameters,
    coordinate: Gemma2LoraCoordinate,
    tensor_replacements: HashMap<String, String>,
}

impl NeuralGemma2LoraConverter {
    /// Create new converter
    pub fn new(params: NeuralGemma2LoraParameters) -> Self {
        let mut converter = Self {
            params,
            coordinate: Gemma2LoraCoordinate::lora_a(),
            tensor_replacements: HashMap::new(),
        };

        converter.initialize_replacements();
        converter
    }

    /// Initialize tensor name replacements
    fn initialize_replacements(&mut self) {
        // Remove base model prefix
        self.tensor_replacements
            .insert("base_model.model.".to_string(), "".to_string());

        // Layer mappings
        self.tensor_replacements
            .insert("model.layers".to_string(), "blk".to_string());

        // Attention projections
        self.tensor_replacements
            .insert("self_attn.q_proj".to_string(), "attn_q".to_string());
        self.tensor_replacements
            .insert("self_attn.k_proj".to_string(), "attn_k".to_string());
        self.tensor_replacements
            .insert("self_attn.v_proj".to_string(), "attn_v".to_string());
        self.tensor_replacements
            .insert("self_attn.o_proj".to_string(), "attn_output".to_string());

        // Feed-forward network
        self.tensor_replacements
            .insert("mlp.gate_proj".to_string(), "ffn_gate".to_string());
        self.tensor_replacements
            .insert("mlp.down_proj".to_string(), "ffn_down".to_string());
        self.tensor_replacements
            .insert("mlp.up_proj".to_string(), "ffn_up".to_string());

        // LoRA naming conventions
        self.tensor_replacements
            .insert("lora_A.weight".to_string(), "weight.lora_a".to_string());
        self.tensor_replacements
            .insert("lora_B.weight".to_string(), "weight.lora_b".to_string());
        self.tensor_replacements
            .insert("lora_a".to_string(), "weight.lora_a".to_string());
        self.tensor_replacements
            .insert("lora_b".to_string(), "weight.lora_b".to_string());
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

    /// Validate and process LoRA tensor
    pub fn process_lora_tensor(&self, tensor: &NeuralSourceTensor) -> (String, Vec<u8>, Vec<u64>) {
        let new_name = self.replace_tensor_name(&tensor.name);

        // Check if needs transpose
        let needs_transpose = LoraShapeValidator::needs_transpose(&new_name, &tensor.shape);

        let (processed_data, final_shape) = if needs_transpose && tensor.shape.len() >= 2 {
            let rows = tensor.shape[0] as usize;
            let cols = tensor.shape[1] as usize;
            let repacked = LoraTensorRepacker::repack_bytes(&tensor.data, rows, cols);
            let new_shape = LoraShapeValidator::transposed_shape(&tensor.shape);
            (repacked, new_shape)
        } else {
            (tensor.data.clone(), tensor.shape.clone())
        };

        (new_name, processed_data, final_shape)
    }

    /// Get parameter info
    pub fn parameter_info(&self) -> Gemma2LoraParameterInfo {
        Gemma2LoraParameterInfo {
            adapter_name: self.params.adapter_name.clone(),
            base_model: self.params.base_model_name.clone(),
            version: self.params.adapter_version.clone(),
            lora_rank: self.params.lora_config.lora_rank,
            lora_alpha: self.params.lora_config.lora_alpha,
            scaling_factor: self.params.lora_config.scaling_factor(),
            target_modules: self.params.lora_config.target_modules.clone(),
            architecture: self.params.architecture.clone(),
            coordinate: self.coordinate,
        }
    }
}

/// Gemma2 LoRA parameter info
#[derive(Debug, Clone)]
pub struct Gemma2LoraParameterInfo {
    pub adapter_name: String,
    pub base_model: String,
    pub version: String,
    pub lora_rank: u32,
    pub lora_alpha: f32,
    pub scaling_factor: f32,
    pub target_modules: Vec<String>,
    pub architecture: String,
    pub coordinate: Gemma2LoraCoordinate,
}

impl Gemma2LoraParameterInfo {
    /// Format as human-readable summary
    pub fn format_summary(&self) -> String {
        format!(
            "Gemma2 LoRA Adapter '{}' (rank={}, alpha={}, scaling={:.2}, {} modules)",
            self.adapter_name,
            self.lora_rank,
            self.lora_alpha,
            self.scaling_factor,
            self.target_modules.len()
        )
    }

    /// Check if valid configuration
    pub fn is_valid(&self) -> bool {
        self.lora_rank > 0 && self.lora_alpha > 0.0 && !self.target_modules.is_empty()
    }
}

impl NeuralAdapterConverter for NeuralGemma2LoraConverter {
    fn to_metadata_kv(&self, _tokenizer: &NeuralConversionTokenizer) -> NeuralMetadataKV {
        let mut kv = NeuralMetadataKV::new();

        // Base metadata
        kv.insert("general.architecture", "gemma2".to_string());
        kv.insert("general.name", self.params.adapter_name.clone());
        kv.insert("adapter.type", "lora".to_string());
        kv.insert("adapter.base_model", self.params.base_model_name.clone());
        kv.insert("adapter.version", self.params.adapter_version.clone());

        // LoRA-specific metadata
        kv.insert("lora.rank", self.params.lora_config.lora_rank);
        kv.insert("lora.alpha", self.params.lora_config.lora_alpha);
        kv.insert("lora.dropout", self.params.lora_config.lora_dropout);
        kv.insert("lora.scaling_factor", self.params.lora_config.scaling_factor());
        kv.insert("lora.target_modules", self.params.lora_config.target_modules.clone());
        kv.insert("lora.bias", self.params.lora_config.bias.clone());

        // Add coordinate metadata
        kv.insert("pronax.coordinate.sequence", self.coordinate.adapter_sequence);
        kv.insert("pronax.coordinate.tier", self.coordinate.rank_tier);
        kv.insert("pronax.coordinate.depth", self.coordinate.projection_depth);
        kv.insert("pronax.coordinate.adaptation", self.coordinate.adaptation_score);

        kv.set_architecture("gemma2");
        kv
    }

    fn convert_tensors(&self, tensors: &[NeuralSourceTensor]) -> Vec<NeuralGgmlTensor> {
        let mut converted = Vec::with_capacity(tensors.len());

        for (idx, tensor) in tensors.iter().enumerate() {
            let (new_name, processed_data, final_shape) = self.process_lora_tensor(tensor);

            // Determine coordinate based on tensor type
            let coordinate = if new_name.contains("lora_a") {
                Gemma2LoraCoordinate::lora_a()
            } else if new_name.contains("lora_b") {
                Gemma2LoraCoordinate::lora_b()
            } else if new_name.contains("attn") {
                Gemma2LoraCoordinate::attention()
            } else if new_name.contains("ffn") {
                Gemma2LoraCoordinate::feed_forward()
            } else {
                Gemma2LoraCoordinate::new(idx as u64, 500, 5, 0.95)
            };

            // Create converted tensor
            let converted_tensor = NeuralGgmlTensor::new(
                new_name,
                tensor.data_type,
                final_shape,
                processed_data,
            )
            .with_coordinate(crate::convert::pronax_converter_core::ConversionCoordinate::new(
                idx as u64,
                coordinate.rank_tier,
                coordinate.projection_depth,
                coordinate.adaptation_score,
            ));

            converted.push(converted_tensor);
        }

        converted
    }

    fn name_replacements(&self) -> Vec<(String, String)> {
        self.replacement_pairs()
    }

    fn architecture(&self) -> &str {
        "gemma2"
    }

    fn base_model(&self) -> &str {
        &self.params.base_model_name
    }

    fn coordinate(&self) -> ConversionCoordinate {
        ConversionCoordinate::new(
            self.coordinate.adapter_sequence,
            self.coordinate.rank_tier,
            self.coordinate.projection_depth,
            self.coordinate.adaptation_score,
        )
    }
}

/// LoRA utility functions
pub struct LoraUtility;

impl LoraUtility {
    /// Merge LoRA weights with base model (W = W + alpha/rank * B * A)
    pub fn merge_lora(
        base_weight: &[f32],
        lora_a: &[f32],
        lora_b: &[f32],
        alpha: f32,
        rank: u32,
    ) -> Vec<f32> {
        let scaling = alpha / rank as f32;

        // For simplicity, assume 2D matrices
        // In practice, this would need proper matrix multiplication
        let mut merged = base_weight.to_vec();

        // Add scaled LoRA contribution
        for i in 0..merged.len().min(lora_a.len()).min(lora_b.len()) {
            merged[i] += scaling * lora_a[i] * lora_b[i];
        }

        merged
    }

    /// Estimate adapter parameter count
    pub fn estimate_adapter_params(
        base_hidden_size: u32,
        num_layers: u32,
        rank: u32,
        num_target_modules: u32,
    ) -> u64 {
        // Each LoRA adapter has A [hidden_size, rank] and B [rank, hidden_size]
        let params_per_module = 2u64 * base_hidden_size as u64 * rank as u64;
        let total_modules = num_layers as u64 * num_target_modules as u64;

        params_per_module * total_modules
    }

    /// Get default target modules for Gemma2
    pub fn default_target_modules() -> Vec<String> {
        vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
            "gate_proj".to_string(),
            "up_proj".to_string(),
            "down_proj".to_string(),
        ]
    }
}

/// Factory function to create Gemma2 LoRA converter
pub fn create_gemma2_lora_converter(params: NeuralGemma2LoraParameters) -> NeuralGemma2LoraConverter {
    NeuralGemma2LoraConverter::new(params)
}

/// Convenience function with default parameters
pub fn create_default_gemma2_lora_converter() -> NeuralGemma2LoraConverter {
    NeuralGemma2LoraConverter::new(NeuralGemma2LoraParameters::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma2_lora_coordinate() {
        let coord = Gemma2LoraCoordinate::lora_a();
        assert_eq!(coord.rank_tier, 600);
        assert_eq!(coord.adaptation_score, 0.96);

        let score = coord.importance_score();
        assert!(score > 0);
    }

    #[test]
    fn test_lora_config() {
        let config = NeuralLoraAdapterConfig::new();

        assert_eq!(config.lora_rank, 8);
        assert_eq!(config.lora_alpha, 16.0);
        assert_eq!(config.scaling_factor(), 2.0);
        assert!(config.targets_module("q_proj"));
        assert!(!config.targets_module("lm_head"));
    }

    #[test]
    fn test_gemma2_lora_parameters() {
        let params = NeuralGemma2LoraParameters::new();

        assert_eq!(params.architecture, "gemma2");
        assert_eq!(params.lora_config.lora_rank, 8);
        assert!(params.base_model_name.contains("gemma"));
    }

    #[test]
    fn test_shape_validator() {
        // LoRA-A with wrong shape [rank, in_dim] should need transpose
        let shape_a = vec![8u64, 4096]; // [rank, in_dim] - needs transpose
        assert!(LoraShapeValidator::needs_transpose("attn_q.weight.lora_a", &shape_a));

        // LoRA-A with correct shape [in_dim, rank]
        let shape_a_correct = vec![4096u64, 8];
        assert!(!LoraShapeValidator::needs_transpose("attn_q.weight.lora_a", &shape_a_correct));

        // LoRA-B with wrong shape [out_dim, rank] should need transpose
        let shape_b = vec![4096u64, 8]; // [out_dim, rank] - needs transpose
        assert!(LoraShapeValidator::needs_transpose("attn_q.weight.lora_b", &shape_b));

        // LoRA-B with correct shape [rank, out_dim]
        let shape_b_correct = vec![8u64, 4096];
        assert!(!LoraShapeValidator::needs_transpose("attn_q.weight.lora_b", &shape_b_correct));
    }

    #[test]
    fn test_tensor_repacker() {
        // Test 2x3 matrix transpose
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 row-major
        let repacked = LoraTensorRepacker::repack_2d(&data, 2, 3);

        // After transpose (column-major): [1, 4, 2, 5, 3, 6]
        assert_eq!(repacked, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_tensor_replacement() {
        let params = NeuralGemma2LoraParameters::new();
        let converter = NeuralGemma2LoraConverter::new(params);

        let replaced = converter.replace_tensor_name("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight");
        assert!(replaced.contains("blk"));
        assert!(replaced.contains("attn_q"));
        assert!(replaced.contains("lora_a"));
        assert!(!replaced.contains("base_model"));
    }

    #[test]
    fn test_parameter_info() {
        let params = NeuralGemma2LoraParameters::new();
        let converter = NeuralGemma2LoraConverter::new(params);
        let info = converter.parameter_info();

        assert_eq!(info.lora_rank, 8);
        assert_eq!(info.lora_alpha, 16.0);
        assert_eq!(info.scaling_factor, 2.0);
        assert!(info.is_valid());
    }

    #[test]
    fn test_lora_utility_estimate() {
        let params = LoraUtility::estimate_adapter_params(4096, 32, 8, 7);

        // 2 * 4096 * 8 * 32 * 7 = ~14.7M parameters
        assert!(params > 10_000_000);
        assert!(params < 20_000_000);
    }

    #[test]
    fn test_lora_merge() {
        let base = vec![1.0f32, 2.0, 3.0, 4.0];
        let lora_a = vec![0.1f32, 0.2, 0.3, 0.4];
        let lora_b = vec![1.0f32, 1.0, 1.0, 1.0];

        let merged = LoraUtility::merge_lora(&base, &lora_a, &lora_b, 16.0, 8);
        let scaling = 2.0;

        // merged[i] = base[i] + scaling * lora_a[i] * lora_b[i]
        assert_eq!(merged[0], 1.0 + scaling * 0.1 * 1.0);
        assert_eq!(merged[1], 2.0 + scaling * 0.2 * 1.0);
    }

    #[test]
    fn test_adapter_converter_trait() {
        let params = NeuralGemma2LoraParameters::new();
        let converter = NeuralGemma2LoraConverter::new(params);

        assert_eq!(converter.architecture(), "gemma2");
        assert!(converter.base_model().contains("gemma"));
    }
}