
use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tracing::{debug, error, info, warn};

use crate::fs::ggml::{
    pronax_ggml_format::NeuralGgmlTensor, NeuralGgmlFormat, NeuralGgmlMetadata,
};
use crate::tokenizer::pronax_vocabulary::{
    NeuralSpecialToken, NeuralVocabulary, SpecialTokenType, TokenType,
};

/// 3D Spatial coordinate for model conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversionCoordinate {
    pub tensor_sequence: u64,
    pub architecture_tier: u16,
    pub precision_depth: u8,
    pub quality_score: f32,
}

impl ConversionCoordinate {
    pub const fn new(seq: u64, tier: u16, depth: u8, score: f32) -> Self {
        Self {
            tensor_sequence: seq,
            architecture_tier: tier,
            precision_depth: depth,
            quality_score: score,
        }
    }

    pub const fn standard() -> Self {
        Self::new(0, 500, 5, 0.95)
    }

    pub const fn high_precision() -> Self {
        Self::new(0, 800, 8, 0.98)
    }

    pub const fn adapter() -> Self {
        Self::new(0, 600, 6, 0.94)
    }

    /// Calculate conversion priority
    pub fn priority_score(&self) -> u64 {
        let seq_factor = 1000u64.saturating_sub(self.tensor_sequence);
        let tier_boost = self.architecture_tier as u64 * 100;
        let depth_norm = self.precision_depth as u64 * 10;
        let quality_boost = (self.quality_score * 1000.0) as u64;
        
        seq_factor + tier_boost + depth_norm + quality_boost
    }
}

/// Converter error types
#[derive(Error, Debug, Clone)]
pub enum ConverterError {
    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),
    
    #[error("Config parse error: {0}")]
    ConfigParseError(String),
    
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
    
    #[error("Tensor parse error: {0}")]
    TensorParseError(String),
    
    #[error("File I/O error: {0}")]
    FileError(String),
    
    #[error("Architecture not set for base model")]
    ArchitectureNotSet,
    
    #[error("Vocabulary size mismatch: expected {expected}, got {actual}")]
    VocabularySizeMismatch { expected: usize, actual: usize },
    
    #[error("Missing required file: {0}")]
    MissingFile(String),
    
    #[error("Conversion failed: {0}")]
    ConversionFailed(String),
}

/// Model parameters from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralModelParameters {
    pub architectures: Vec<String>,
    pub vocab_size: u32,
    pub model_type: Option<String>,
    #[serde(rename = "text_config")]
    pub text_model: Option<NeuralTextModelConfig>,
}

/// Text model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralTextModelConfig {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub model_type: Option<String>,
}

/// Adapter parameters (LoRA)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralAdapterParameters {
    #[serde(rename = "lora_alpha")]
    pub alpha: u32,
    #[serde(rename = "lora_layers")]
    pub layers: u32,
    #[serde(rename = "lora_parameters")]
    pub params: NeuralLoraParameters,
}

/// LoRA parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralLoraParameters {
    pub rank: u32,
    pub alpha: f32,
    pub scale: f32,
}

/// Key-Value metadata store
#[derive(Debug, Clone, Default)]
pub struct NeuralMetadataKV {
    inner: HashMap<String, Value>,
    architecture: String,
}

impl NeuralMetadataKV {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
            architecture: "unknown".to_string(),
        }
    }

    pub fn with_coordinate(mut self, coordinate: ConversionCoordinate) -> Self {
        self.insert("pronax.coordinate.sequence", coordinate.tensor_sequence);
        self.insert("pronax.coordinate.tier", coordinate.architecture_tier);
        self.insert("pronax.coordinate.depth", coordinate.precision_depth);
        self.insert("pronax.coordinate.score", coordinate.quality_score);
        self
    }

    /// Get string value
    pub fn get_string(&self, key: &str, default: &str) -> String {
        let full_key = if !key.starts_with("tokenizer.") && !key.starts_with("general.") && !key.starts_with("pronax.") {
            format!("{}.{}", self.architecture, key)
        } else {
            key.to_string()
        };

        self.inner
            .get(&full_key)
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| default.to_string())
    }

    /// Get uint value
    pub fn get_uint(&self, key: &str, default: u32) -> u32 {
        let full_key = if !key.starts_with("tokenizer.") && !key.starts_with("general.") && !key.starts_with("pronax.") {
            format!("{}.{}", self.architecture, key)
        } else {
            key.to_string()
        };

        self.inner
            .get(&full_key)
            .and_then(|v| v.as_u64().map(|n| n as u32))
            .unwrap_or(default)
    }

    /// Get float value
    pub fn get_float(&self, key: &str, default: f32) -> f32 {
        let full_key = if !key.starts_with("tokenizer.") && !key.starts_with("general.") && !key.starts_with("pronax.") {
            format!("{}.{}", self.architecture, key)
        } else {
            key.to_string()
        };

        self.inner
            .get(&full_key)
            .and_then(|v| v.as_f64().map(|n| n as f32))
            .unwrap_or(default)
    }

    /// Get bool value
    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        let full_key = if !key.starts_with("tokenizer.") && !key.starts_with("general.") && !key.starts_with("pronax.") {
            format!("{}.{}", self.architecture, key)
        } else {
            key.to_string()
        };

        self.inner
            .get(&full_key)
            .and_then(|v| v.as_bool())
            .unwrap_or(default)
    }

    /// Get string array
    pub fn get_strings(&self, key: &str, default: Vec<String>) -> Vec<String> {
        let full_key = if !key.starts_with("tokenizer.") && !key.starts_with("general.") && !key.starts_with("pronax.") {
            format!("{}.{}", self.architecture, key)
        } else {
            key.to_string()
        };

        self.inner
            .get(&full_key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or(default)
    }

    /// Get uint array
    pub fn get_uints(&self, key: &str, default: Vec<u32>) -> Vec<u32> {
        let full_key = if !key.starts_with("tokenizer.") && !key.starts_with("general.") && !key.starts_with("pronax.") {
            format!("{}.{}", self.architecture, key)
        } else {
            key.to_string()
        };

        self.inner
            .get(&full_key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u32))
                    .collect()
            })
            .unwrap_or(default)
    }

    /// Get architecture
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Set architecture
    pub fn set_architecture(&mut self, arch: impl Into<String>) {
        self.architecture = arch.into();
    }

    /// Insert value
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Value>) {
        self.inner.insert(key.into(), value.into());
    }

    /// Get raw value
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.inner.get(key)
    }

    /// Get all keys
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.inner.keys()
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Merge with another KV store
    pub fn merge(&mut self, other: NeuralMetadataKV) {
        for (k, v) in other.inner {
            self.inner.insert(k, v);
        }
    }

    /// Convert to GGUF metadata format
    pub fn to_gguf_metadata(&self) -> NeuralGgmlMetadata {
        let mut metadata = NeuralGgmlMetadata::new();
        
        for (key, value) in &self.inner {
            match value {
                Value::String(s) => metadata.insert_string(key.clone(), s.clone()),
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        metadata.insert_i64(key.clone(), i);
                    } else if let Some(f) = n.as_f64() {
                        metadata.insert_f64(key.clone(), f);
                    }
                }
                Value::Bool(b) => metadata.insert_bool(key.clone(), *b),
                Value::Array(arr) => {
                    if let Some(first) = arr.first() {
                        match first {
                            Value::String(_) => {
                                let strings: Vec<String> = arr.iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect();
                                metadata.insert_array_string(key.clone(), strings);
                            }
                            Value::Number(_) => {
                                let nums: Vec<f32> = arr.iter()
                                    .filter_map(|v| v.as_f64().map(|n| n as f32))
                                    .collect();
                                metadata.insert_array_f32(key.clone(), nums);
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        
        metadata
    }
}

impl IntoIterator for NeuralMetadataKV {
    type Item = (String, Value);
    type IntoIter = std::collections::hash_map::IntoIter<String, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

/// Tokenizer data for conversion
#[derive(Debug, Clone)]
pub struct NeuralConversionTokenizer {
    pub vocabulary: NeuralVocabulary,
    pub merges: Vec<String>,
    pub template: String,
    pub pre_tokenizer: String,
    pub special_tokens: Vec<NeuralSpecialToken>,
    pub coordinate: ConversionCoordinate,
}

impl NeuralConversionTokenizer {
    pub fn new(vocabulary: NeuralVocabulary) -> Self {
        Self {
            vocabulary,
            merges: Vec::new(),
            template: String::new(),
            pre_tokenizer: String::new(),
            special_tokens: Vec::new(),
            coordinate: ConversionCoordinate::standard(),
        }
    }

    pub fn with_coordinate(mut self, coordinate: ConversionCoordinate) -> Self {
        self.coordinate = coordinate;
        self
    }

    /// Generate KV metadata for tokenizer
    pub fn to_kv(&self) -> NeuralMetadataKV {
        let mut kv = NeuralMetadataKV::new();
        
        kv.insert("general.file_type", 1u32);
        kv.insert("general.quantization_version", 2u32);
        kv.insert("tokenizer.ggml.pre", self.pre_tokenizer.clone());
        kv.insert("tokenizer.ggml.model", "bpe");
        kv.insert("tokenizer.ggml.tokens", 
            self.vocabulary.tokens.iter().map(|t| t.text.clone()).collect::<Vec<_>>());
        kv.insert("tokenizer.ggml.scores", 
            self.vocabulary.tokens.iter().map(|t| t.score).collect::<Vec<_>>());
        kv.insert("tokenizer.ggml.token_type", 
            self.vocabulary.tokens.iter().map(|t| t.token_type as u32).collect::<Vec<_>>());
        
        if !self.merges.is_empty() {
            kv.insert("tokenizer.ggml.merges", self.merges.clone());
        }
        
        if !self.template.is_empty() {
            kv.insert("tokenizer.chat_template", self.template.clone());
        }
        
        // Add special tokens
        for token in &self.special_tokens {
            let key = format!("tokenizer.ggml.{}_token", token.token_type.as_str());
            kv.insert(&key, token.text.clone());
            
            let id_key = format!("tokenizer.ggml.{}_token_id", token.token_type.as_str());
            kv.insert(&id_key, token.id as u32);
        }
        
        kv.set_architecture("llama");
        kv
    }
}

/// Model converter trait
pub trait NeuralModelConverter: Send + Sync {
    /// Convert to KV metadata
    fn to_metadata_kv(&self, tokenizer: &NeuralConversionTokenizer) -> NeuralMetadataKV;
    
    /// Convert tensors
    fn convert_tensors(&self, tensors: &[NeuralSourceTensor]) -> Vec<NeuralGgmlTensor>;
    
    /// Get tensor name replacements
    fn name_replacements(&self) -> Vec<(String, String)>;
    
    /// Get special token types
    fn special_token_types(&self) -> Vec<SpecialTokenType> {
        vec![
            SpecialTokenType::Bos,
            SpecialTokenType::Eos,
            SpecialTokenType::Unk,
            SpecialTokenType::Sep,
            SpecialTokenType::Pad,
            SpecialTokenType::Cls,
            SpecialTokenType::Mask,
        ]
    }
    
    /// Get architecture name
    fn architecture(&self) -> &str;
    
    /// Get coordinate
    fn coordinate(&self) -> ConversionCoordinate {
        ConversionCoordinate::standard()
    }
}

/// Adapter converter trait
pub trait NeuralAdapterConverter: Send + Sync {
    /// Convert to KV metadata
    fn to_metadata_kv(&self, base_metadata: &NeuralMetadataKV) -> NeuralMetadataKV;
    
    /// Convert tensors
    fn convert_tensors(&self, tensors: &[NeuralSourceTensor]) -> Vec<NeuralGgmlTensor>;
    
    /// Get tensor name replacements
    fn name_replacements(&self) -> Vec<(String, String)>;
    
    /// Get adapter type
    fn adapter_type(&self) -> &str;
}

/// Source tensor from input model
#[derive(Debug, Clone)]
pub struct NeuralSourceTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data_type: TensorDataType,
    pub data: Vec<u8>,
    pub coordinate: ConversionCoordinate,
}

impl NeuralSourceTensor {
    pub fn new(name: impl Into<String>, shape: Vec<usize>, data_type: TensorDataType) -> Self {
        Self {
            name: name.into(),
            shape,
            data_type,
            data: Vec::new(),
            coordinate: ConversionCoordinate::standard(),
        }
    }

    pub fn with_data(mut self, data: Vec<u8>) -> Self {
        self.data = data;
        self
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn num_bytes(&self) -> usize {
        self.num_elements() * self.data_type.size_bytes()
    }
}

/// Tensor data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDataType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    I8,
    I16,
    I32,
}

impl TensorDataType {
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            _ => 2, // Quantized types average
        }
    }
}

/// Architecture registry
pub struct NeuralArchitectureRegistry {
    converters: HashMap<String, Box<dyn Fn() -> Box<dyn NeuralModelConverter> + Send + Sync>>,
    coordinate: ConversionCoordinate,
}

impl NeuralArchitectureRegistry {
    pub fn new() -> Self {
        Self {
            converters: HashMap::new(),
            coordinate: ConversionCoordinate::standard(),
        }
    }

    pub fn register<C: NeuralModelConverter + 'static>(
        &mut self,
        architecture: impl Into<String>,
        factory: impl Fn() -> C + Send + Sync + 'static,
    ) {
        self.converters.insert(
            architecture.into(),
            Box::new(move || Box::new(factory()) as Box<dyn NeuralModelConverter>),
        );
    }

    pub fn get(&self, architecture: &str) -> Option<Box<dyn NeuralModelConverter>> {
        self.converters.get(architecture).map(|f| f())
    }

    pub fn contains(&self, architecture: &str) -> bool {
        self.converters.contains_key(architecture)
    }

    pub fn architectures(&self) -> impl Iterator<Item = &String> {
        self.converters.keys()
    }
}

impl Default for NeuralArchitectureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Model conversion engine
pub struct NeuralModelConversionEngine {
    registry: NeuralArchitectureRegistry,
    coordinate_counter: AtomicU64,
}

impl NeuralModelConversionEngine {
    pub fn new() -> Self {
        Self {
            registry: NeuralArchitectureRegistry::new(),
            coordinate_counter: AtomicU64::new(0),
        }
    }

    /// Get next coordinate
    fn next_coordinate(&self) -> ConversionCoordinate {
        let seq = self.coordinate_counter.fetch_add(1, Ordering::SeqCst);
        ConversionCoordinate::new(seq, 500, 5, 0.95)
    }

    /// Load model metadata from directory
    pub fn load_model_metadata(
        &self,
        model_path: impl AsRef<Path>,
    ) -> Result<(Box<dyn NeuralModelConverter>, NeuralConversionTokenizer), ConverterError> {
        let config_path = model_path.as_ref().join("config.json");
        
        if !config_path.exists() {
            return Err(ConverterError::MissingFile("config.json".to_string()));
        }

        // Read and parse config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| ConverterError::FileError(e.to_string()))?;

        let params: NeuralModelParameters = serde_json::from_str(&config_str)
            .map_err(|e| ConverterError::ConfigParseError(e.to_string()))?;

        if params.architectures.is_empty() {
            return Err(ConverterError::ConfigParseError("No architecture specified".to_string()));
        }

        // Get converter for architecture
        let arch = &params.architectures[0];
        let converter = self.registry.get(arch)
            .ok_or_else(|| ConverterError::UnsupportedArchitecture(arch.clone()))?;

        // Parse tokenizer
        let tokenizer = self.parse_tokenizer(model_path.as_ref(), &*converter)?;

        info!("Loaded model metadata for architecture: {}", arch);
        Ok((converter, tokenizer))
    }

    /// Parse tokenizer
    fn parse_tokenizer(
        &self,
        model_path: &Path,
        converter: &dyn NeuralModelConverter,
    ) -> Result<NeuralConversionTokenizer, ConverterError> {
        // Try tokenizer.json first, then tokenizer.model
        let tokenizer_json = model_path.join("tokenizer.json");
        let tokenizer_model = model_path.join("tokenizer.model");

        let vocab = if tokenizer_json.exists() {
            // Parse tokenizer.json
            let content = std::fs::read_to_string(&tokenizer_json)
                .map_err(|e| ConverterError::TokenizerError(e.to_string()))?;
            
            let value: Value = serde_json::from_str(&content)
                .map_err(|e| ConverterError::TokenizerError(e.to_string()))?;

            self.extract_vocabulary_from_tokenizer_json(&value)?
        } else if tokenizer_model.exists() {
            // Parse tokenizer.model (SentencePiece)
            NeuralVocabulary::from_sentencepiece_file(&tokenizer_model)
                .map_err(|e| ConverterError::TokenizerError(e.to_string()))?
        } else {
            return Err(ConverterError::MissingFile("tokenizer.json or tokenizer.model".to_string()));
        };

        let mut tokenizer = NeuralConversionTokenizer::new(vocab)
            .with_coordinate(self.next_coordinate());

        // Parse special tokens
        tokenizer.special_tokens = self.parse_special_tokens(model_path, converter)?;

        // Try to load chat template
        let template_path = model_path.join("tokenizer_config.json");
        if template_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&template_path) {
                if let Ok(value) = serde_json::from_str::<Value>(&content) {
                    if let Some(template) = value.get("chat_template").and_then(|v| v.as_str()) {
                        tokenizer.template = template.to_string();
                    }
                }
            }
        }

        Ok(tokenizer)
    }

    /// Extract vocabulary from tokenizer.json
    fn extract_vocabulary_from_tokenizer_json(
        &self,
        value: &Value,
    ) -> Result<NeuralVocabulary, ConverterError> {
        let model = value.get("model")
            .ok_or_else(|| ConverterError::TokenizerError("Missing model field".to_string()))?;

        let vocab = model.get("vocab")
            .ok_or_else(|| ConverterError::TokenizerError("Missing vocab field".to_string()))?;

        let vocab_obj = vocab.as_object()
            .ok_or_else(|| ConverterError::TokenizerError("Vocab is not an object".to_string()))?;

        let mut vocabulary = NeuralVocabulary::new();

        for (token, id_val) in vocab_obj {
            let id = id_val.as_u64()
                .ok_or_else(|| ConverterError::TokenizerError(format!("Invalid ID for token: {}", token)))? as usize;
            
            vocabulary.add_token_with_id(
                token.clone(),
                id,
                0.0, // score
                TokenType::Normal,
            );
        }

        Ok(vocabulary)
    }

    /// Parse special tokens
    fn parse_special_tokens(
        &self,
        model_path: &Path,
        converter: &dyn NeuralModelConverter,
    ) -> Result<Vec<NeuralSpecialToken>, ConverterError> {
        let mut tokens = Vec::new();
        let types = converter.special_token_types();

        let special_tokens_map = model_path.join("special_tokens_map.json");
        if special_tokens_map.exists() {
            let content = std::fs::read_to_string(&special_tokens_map)
                .map_err(|e| ConverterError::TokenizerError(e.to_string()))?;

            let value: Value = serde_json::from_str(&content)
                .map_err(|e| ConverterError::TokenizerError(e.to_string()))?;

            for token_type in types {
                if let Some(token_val) = value.get(token_type.as_str()) {
                    if let Some(content) = token_val.get("content").and_then(|v| v.as_str()) {
                        tokens.push(NeuralSpecialToken::new(
                            content.to_string(),
                            token_type,
                            tokens.len(), // ID
                            true,
                        ));
                    }
                }
            }
        }

        Ok(tokens)
    }

    /// Convert model
    pub fn convert_model(
        &self,
        model_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
    ) -> Result<(), ConverterError> {
        let (converter, tokenizer) = self.load_model_metadata(&model_path)?;

        // Generate metadata
        let metadata_kv = converter.to_metadata_kv(&tokenizer);
        let metadata = metadata_kv.to_gguf_metadata();

        // Parse tensors
        let source_tensors = self.parse_tensors(model_path.as_ref(), &*converter)?;
        let tensors = converter.convert_tensors(&source_tensors);

        // Write GGUF file
        let mut output_file = File::create(&output_path)
            .map_err(|e| ConverterError::FileError(e.to_string()))?;

        let format = NeuralGgmlFormat::new(metadata, tensors);
        format.write_to(&mut output_file)
            .map_err(|e| ConverterError::ConversionFailed(e.to_string()))?;

        info!("Model converted successfully: {:?}", output_path.as_ref());
        Ok(())
    }

    /// Parse tensors from model directory
    fn parse_tensors(
        &self,
        model_path: &Path,
        converter: &dyn NeuralModelConverter,
    ) -> Result<Vec<NeuralSourceTensor>, ConverterError> {
        let mut tensors = Vec::new();

        // Look for safetensors files
        for entry in std::fs::read_dir(model_path)
            .map_err(|e| ConverterError::FileError(e.to_string()))? {
            
            let entry = entry.map_err(|e| ConverterError::FileError(e.to_string()))?;
            let path = entry.path();
            
            if let Some(ext) = path.extension() {
                if ext == "safetensors" {
                    let file_tensors = self.parse_safetensors_file(&path, converter)?;
                    tensors.extend(file_tensors);
                }
            }
        }

        if tensors.is_empty() {
            return Err(ConverterError::MissingFile("No .safetensors files found".to_string()));
        }

        Ok(tensors)
    }

    /// Parse safetensors file
    fn parse_safetensors_file(
        &self,
        path: &Path,
        _converter: &dyn NeuralModelConverter,
    ) -> Result<Vec<NeuralSourceTensor>, ConverterError> {
        // Simplified implementation - would use actual safetensors parsing
        // For now, return placeholder
        info!("Parsing safetensors: {:?}", path);
        
        // In real implementation, parse the safetensors format
        // and return the tensors
        
        Ok(Vec::new())
    }
}

impl Default for NeuralModelConversionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Sanitize JSON with non-finite values
pub fn sanitize_nonfinite_json(input: &str) -> String {
    let re = Regex::new(r":\s*NaN|:\s*Infinity|:\s*-Infinity").unwrap();
    re.replace_all(input, ": null").to_string()
}

/// Convenience function to convert model
pub fn convert_model(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<(), ConverterError> {
    let engine = NeuralModelConversionEngine::new();
    engine.convert_model(input_path, output_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_coordinate() {
        let coord = ConversionCoordinate::high_precision();
        assert_eq!(coord.architecture_tier, 800);
        assert_eq!(coord.quality_score, 0.98);
        
        let score = coord.priority_score();
        assert!(score > 0);
    }

    #[test]
    fn test_metadata_kv() {
        let mut kv = NeuralMetadataKV::new();
        
        kv.set_architecture("llama");
        kv.insert("general.name", "test-model");
        kv.insert("llama.hidden_size", 4096u32);
        
        assert_eq!(kv.architecture(), "llama");
        assert_eq!(kv.get_string("general.name", ""), "test-model");
        assert_eq!(kv.get_uint("hidden_size", 0), 4096);
    }

    #[test]
    fn test_model_parameters() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32000,
            "model_type": "llama",
            "text_config": {
                "vocab_size": 32000,
                "hidden_size": 4096
            }
        }"#;

        let params: NeuralModelParameters = serde_json::from_str(json).unwrap();
        
        assert_eq!(params.architectures, vec!["LlamaForCausalLM"]);
        assert_eq!(params.vocab_size, 32000);
        assert!(params.text_model.is_some());
    }

    #[test]
    fn test_tensor_data_type() {
        assert_eq!(TensorDataType::F32.size_bytes(), 4);
        assert_eq!(TensorDataType::F16.size_bytes(), 2);
        assert_eq!(TensorDataType::I8.size_bytes(), 1);
    }

    #[test]
    fn test_source_tensor() {
        let tensor = NeuralSourceTensor::new(
            "test",
            vec![1, 2, 3],
            TensorDataType::F32,
        );
        
        assert_eq!(tensor.num_elements(), 6);
        assert_eq!(tensor.num_bytes(), 24);
    }

    #[test]
    fn test_sanitize_json() {
        let input = r#"{"value": NaN, "inf": Infinity, "neg": -Infinity}"#;
        let sanitized = sanitize_nonfinite_json(input);
        
        assert!(sanitized.contains("null"));
        assert!(!sanitized.contains("NaN"));
    }

    #[test]
    fn test_adapter_parameters() {
        let json = r#"{
            "lora_alpha": 16,
            "lora_layers": 32,
            "lora_parameters": {
                "rank": 8,
                "alpha": 16.0,
                "scale": 1.0
            }
        }"#;

        let params: NeuralAdapterParameters = serde_json::from_str(json).unwrap();
        
        assert_eq!(params.alpha, 16);
        assert_eq!(params.params.rank, 8);
    }
}