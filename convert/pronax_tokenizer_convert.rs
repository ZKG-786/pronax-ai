use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Sha256, Digest};
use tracing::{debug, warn};

use crate::convert::pronax_converter_core::{ConverterError, ConversionCoordinate};
use crate::tokenizer::pronax_vocabulary::{
    NeuralVocabulary, NeuralVocabEntry, TokenSemanticType, VocabCoordinate, SpecialTokenType,
};
use crate::tokenizer::pronax_tokenizer_trait::{
    NeuralTokenizationContext, NeuralTokenCategory, NeuralSpecialToken,
};

/// Neural tokenizer conversion result with 3D spatial metadata
#[derive(Debug, Clone)]
pub struct NeuralTokenizerBundle {
    /// Core vocabulary
    pub vocabulary: NeuralVocabulary,
    /// Special token definitions
    pub special_tokens: Vec<NeuralSpecialTokenDef>,
    /// BPE merge rules
    pub merge_rules: Vec<NeuralBpeMerge>,
    /// Pre-tokenizer type detected
    pub pretokenizer_type: PreTokenizerType,
    /// Chat template string
    pub chat_template: Option<String>,
    /// 3D spatial coordinate for this tokenizer
    pub spatial_coordinate: ConversionCoordinate,
    /// Token type mapping
    pub token_type_map: HashMap<i32, NeuralTokenCategory>,
}

impl NeuralTokenizerBundle {
    /// Create new empty bundle
    pub fn new() -> Self {
        Self {
            vocabulary: NeuralVocabulary::new(),
            special_tokens: Vec::new(),
            merge_rules: Vec::new(),
            pretokenizer_type: PreTokenizerType::Default,
            chat_template: None,
            spatial_coordinate: ConversionCoordinate::standard(),
            token_type_map: HashMap::new(),
        }
    }

    /// Load and parse tokenizer from directory
    pub fn from_directory<P: AsRef<Path>>(dir: P) -> Result<Self, ConverterError> {
        let path = dir.as_ref();
        
        // Try different vocabulary formats in order
        let mut bundle = if path.join("tokenizer.model").exists() {
            debug!("Found SentencePiece tokenizer.model");
            Self::parse_sentencepiece(path)?
        } else if path.join("tokenizer.json").exists() {
            debug!("Found HuggingFace tokenizer.json");
            Self::parse_huggingface(path)?
        } else {
            return Err(ConverterError::MissingFile(
                "No tokenizer.model or tokenizer.json found".to_string()
            ));
        };

        // Enhance with tokenizer_config.json if available
        if path.join("tokenizer_config.json").exists() {
            debug!("Loading tokenizer_config.json");
            bundle.enhance_from_config(path)?;
        }

        // Enhance with generation_config.json if available
        if path.join("generation_config.json").exists() {
            debug!("Loading generation_config.json");
            bundle.enhance_from_generation_config(path)?;
        }

        Ok(bundle)
    }

    /// Parse SentencePiece format tokenizer
    fn parse_sentencepiece<P: AsRef<Path>>(dir: P) -> Result<Self, ConverterError> {
        let path = dir.as_ref().join("tokenizer.model");
        let mut file = File::open(&path)
            .map_err(|e| ConverterError::FileError(format!("Cannot open tokenizer.model: {}", e)))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| ConverterError::FileError(format!("Cannot read tokenizer.model: {}", e)))?;

        // Parse SentencePiece protobuf format
        let sp_model = Self::decode_sentencepiece_proto(&buffer)?;
        
        let mut bundle = Self::new();
        bundle.pretokenizer_type = PreTokenizerType::SentencePiece;

        // Convert pieces to vocabulary entries
        for (idx, piece) in sp_model.pieces.iter().enumerate() {
            let id = idx as i32;
            let semantic_type = match piece.piece_type {
                1 => TokenSemanticType::Normal,
                2 => TokenSemanticType::Unknown,
                3 => TokenSemanticType::Control,
                4 => TokenSemanticType::UserDefined,
                5 => TokenSemanticType::Byte,
                _ => TokenSemanticType::Normal,
            };

            let coordinate = VocabCoordinate::new(
                idx as u64,
                1000 - (idx as u32).min(1000), // Frequency tier (lower index = higher frequency)
                50,
                0.9,
            );

            let entry = NeuralVocabEntry::new(
                &piece.content,
                id,
                semantic_type,
                coordinate,
                piece.score,
            );

            bundle.vocabulary.add_token(
                entry.lexeme.clone(),
                id,
                NeuralTokenCategory::from_u8(semantic_type as u8).unwrap_or(NeuralTokenCategory::Normal),
            );
            bundle.token_type_map.insert(id, NeuralTokenCategory::from_u8(semantic_type as u8).unwrap_or(NeuralTokenCategory::Normal));
        }

        debug!("Loaded {} SentencePiece tokens", bundle.vocabulary.len());
        Ok(bundle)
    }

    /// Decode SentencePiece protobuf (simplified)
    fn decode_sentencepiece_proto(data: &[u8]) -> Result<SentencePieceModel, ConverterError> {
        // Simple protobuf parser for SentencePiece
        // In production, use prost or similar
        let mut model = SentencePieceModel::default();
        
        // Parse pieces from protobuf
        // This is a simplified implementation
        if data.len() < 8 {
            return Err(ConverterError::TokenizerError("Invalid SentencePiece model".to_string()));
        }

        // Try to extract pieces by scanning
        let mut offset = 0;
        while offset < data.len() - 4 {
            // Look for piece entries (simplified parsing)
            if data[offset] == 0x0a { // Field 1 (pieces) wire type 2 (length-delimited)
                if let Some(piece) = Self::parse_piece(data, &mut offset) {
                    model.pieces.push(piece);
                } else {
                    offset += 1;
                }
            } else {
                offset += 1;
            }
        }

        if model.pieces.is_empty() {
            // Fallback: assume text format (one token per line)
            let text = String::from_utf8_lossy(data);
            for (idx, line) in text.lines().enumerate() {
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() >= 2 {
                    if let Ok(score) = parts[1].parse::<f32>() {
                        model.pieces.push(SentencePiece {
                            content: parts[0].to_string(),
                            score,
                            piece_type: 1,
                        });
                    }
                }
            }
        }

        Ok(model)
    }

    fn parse_piece(data: &[u8], offset: &mut usize) -> Option<SentencePiece> {
        if *offset + 2 > data.len() {
            return None;
        }

        let len = data[*offset + 1] as usize;
        if *offset + 2 + len > data.len() {
            return None;
        }

        // Extract piece data (simplified)
        let piece_data = &data[*offset + 2..*offset + 2 + len];
        
        // Parse piece content (field 1)
        let mut piece = SentencePiece::default();
        let mut p = 0;
        while p < piece_data.len() {
            if piece_data[p] == 0x0a && p + 1 < piece_data.len() {
                let str_len = piece_data[p + 1] as usize;
                if p + 2 + str_len <= piece_data.len() {
                    piece.content = String::from_utf8_lossy(&piece_data[p + 2..p + 2 + str_len]).to_string();
                    p += 2 + str_len;
                } else {
                    break;
                }
            } else if piece_data[p] == 0x15 && p + 4 < piece_data.len() {
                // Float field (score)
                let bytes: [u8; 4] = [
                    piece_data[p + 1], piece_data[p + 2], piece_data[p + 3], piece_data[p + 4]
                ];
                piece.score = f32::from_le_bytes(bytes);
                p += 5;
            } else {
                p += 1;
            }
        }

        if !piece.content.is_empty() {
            *offset += 2 + len;
            Some(piece)
        } else {
            None
        }
    }

    /// Parse HuggingFace tokenizer.json format
    fn parse_huggingface<P: AsRef<Path>>(dir: P) -> Result<Self, ConverterError> {
        let path = dir.as_ref().join("tokenizer.json");
        let mut file = File::open(&path)
            .map_err(|e| ConverterError::FileError(format!("Cannot open tokenizer.json: {}", e)))?;

        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| ConverterError::FileError(format!("Cannot read tokenizer.json: {}", e)))?;

        let tokenizer: HuggingFaceTokenizer = serde_json::from_str(&content)
            .map_err(|e| ConverterError::TokenizerError(format!("JSON parse error: {}", e)))?;

        let mut bundle = Self::new();
        
        // Detect pre-tokenizer type from SHA256 of regex patterns
        bundle.pretokenizer_type = Self::detect_pretokenizer(&tokenizer);

        // Parse vocabulary
        let mut tokens_map: HashMap<i32, TokenEntry> = HashMap::new();
        
        for (token_str, id) in &tokenizer.model.vocab {
            tokens_map.insert(*id, TokenEntry {
                content: token_str.clone(),
                id: *id,
                special: false,
                user_defined: false,
            });
        }

        // Mark added tokens
        for added in &tokenizer.added_tokens {
            if let Some(entry) = tokens_map.get_mut(&added.id) {
                entry.special = added.special;
                entry.user_defined = true;
            }
        }

        // Build vocabulary with 3D coordinates
        let mut sorted_ids: Vec<i32> = tokens_map.keys().copied().collect();
        sorted_ids.sort();

        for (idx, id) in sorted_ids.iter().enumerate() {
            let entry = &tokens_map[id];
            let semantic_type = if entry.special {
                TokenSemanticType::Control
            } else if entry.user_defined {
                TokenSemanticType::UserDefined
            } else {
                TokenSemanticType::Normal
            };

            let coordinate = VocabCoordinate::new(
                *id as u64,
                (1000 - idx.min(1000)) as u32,
                50,
                if entry.special { 1.0 } else { 0.9 },
            );

            let vocab_entry = NeuralVocabEntry::new(
                &entry.content,
                *id,
                semantic_type,
                coordinate,
                *id as f32,
            );

            let category = match semantic_type {
                TokenSemanticType::Control => NeuralTokenCategory::Control,
                TokenSemanticType::UserDefined => NeuralTokenCategory::UserDefined,
                TokenSemanticType::Normal => NeuralTokenCategory::Normal,
                _ => NeuralTokenCategory::Unknown,
            };

            bundle.vocabulary.add_token(vocab_entry.lexeme, *id, category);
            bundle.token_type_map.insert(*id, category);
        }

        // Parse merges
        bundle.merge_rules = Self::parse_merges(&tokenizer.model.merges)?;

        debug!(
            "Loaded {} HF tokens with {} merges",
            bundle.vocabulary.len(),
            bundle.merge_rules.len()
        );

        Ok(bundle)
    }

    /// Detect pre-tokenizer type using SHA256 checksum
    fn detect_pretokenizer(tokenizer: &HuggingFaceTokenizer) -> PreTokenizerType {
        let mut hasher = Sha256::new();
        
        for pre_tokenizer in &tokenizer.pre_tokenizer.pre_tokenizers {
            if pre_tokenizer.type_ == "Split" && !pre_tokenizer.pattern.regex.is_empty() {
                hasher.update(pre_tokenizer.pattern.regex.as_bytes());
            }
        }

        let digest = format!("{:x}", hasher.finalize());
        
        match digest.as_str() {
            "d98f9631be1e9607a9848c26c1f9eac1aa9fc21ac6ba82a2fc0741af9780a48f" => {
                debug!("Detected llama-bpe pre-tokenizer");
                PreTokenizerType::LlamaBpe
            }
            "03df5c5863ad70781dcfdef491ead25140f895fe8010964be0daefe27be32b02" => {
                debug!("Detected deepseek-llm pre-tokenizer");
                PreTokenizerType::DeepSeekLlama
            }
            "21cde974d587f0d54dc8d56b183cc1e6239600172035c68fbd6d4b9f8da0576e" => {
                debug!("Detected deepseek-coder pre-tokenizer");
                PreTokenizerType::DeepSeekCoder
            }
            "1ff7f41064896984db5d1bb6ff64fa4bc29007d08c1b439e505b7392777a319e" => {
                debug!("Detected qwen2 pre-tokenizer");
                PreTokenizerType::Qwen2
            }
            "00431aed57e696b747435f734d1e3b9b1bfd931a121fb5cac7129e97c181e9ba" => {
                debug!("Detected qwen35 pre-tokenizer");
                PreTokenizerType::Qwen35
            }
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" => {
                // Empty pre-tokenizer
                PreTokenizerType::Default
            }
            _ => {
                warn!("Unknown pre-tokenizer, using default: {}", &digest[..16]);
                PreTokenizerType::Default
            }
        }
    }

    /// Parse merge rules from JSON
    fn parse_merges(merges: &Value) -> Result<Vec<NeuralBpeMerge>, ConverterError> {
        let mut result = Vec::new();

        // Try array of strings first: ["a b", "c d"]
        if let Some(arr) = merges.as_array() {
            for item in arr {
                if let Some(s) = item.as_str() {
                    let parts: Vec<&str> = s.split_whitespace().collect();
                    if parts.len() == 2 {
                        result.push(NeuralBpeMerge {
                            first: parts[0].to_string(),
                            second: parts[1].to_string(),
                            priority: result.len() as u32,
                        });
                    }
                } else if let Some(pair) = item.as_array() {
                    // Array format: [["a", "b"], ["c", "d"]]
                    if pair.len() == 2 {
                        if let (Some(a), Some(b)) = (pair[0].as_str(), pair[1].as_str()) {
                            result.push(NeuralBpeMerge {
                                first: a.to_string(),
                                second: b.to_string(),
                                priority: result.len() as u32,
                            });
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Enhance from tokenizer_config.json
    fn enhance_from_config<P: AsRef<Path>>(&mut self, dir: P) -> Result<(), ConverterError> {
        let path = dir.as_ref().join("tokenizer_config.json");
        let mut file = File::open(&path)
            .map_err(|e| ConverterError::FileError(format!("Cannot open tokenizer_config.json: {}", e)))?;

        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| ConverterError::FileError(format!("Cannot read tokenizer_config.json: {}", e)))?;

        let config: HashMap<String, Value> = serde_json::from_str(&content)
            .map_err(|e| ConverterError::ConfigParseError(format!("JSON parse error: {}", e)))?;

        // Parse chat template
        if let Some(template) = config.get("chat_template") {
            // Try string first
            if let Some(s) = template.as_str() {
                self.chat_template = Some(s.to_string());
            } else if let Some(arr) = template.as_array() {
                // Array of template objects
                for item in arr {
                    if let Some(obj) = item.as_object() {
                        if let (Some(name), Some(tpl)) = (
                            obj.get("name").and_then(|v| v.as_str()),
                            obj.get("template").and_then(|v| v.as_str()),
                        ) {
                            if name == "default" {
                                self.chat_template = Some(tpl.to_string());
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Parse special tokens
        let special_types = vec!["bos", "eos", "unk", "pad", "cls", "sep", "mask"];
        for stype in special_types {
            let add_token_key = format!("add_{}_token", stype);
            let token_key = format!("{}_token", stype);

            let add_token = config.get(&add_token_key)
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let content = if let Some(val) = config.get(&token_key) {
                if let Some(s) = val.as_str() {
                    Some(s.to_string())
                } else if let Some(obj) = val.as_object() {
                    obj.get("content").and_then(|v| v.as_str()).map(|s| s.to_string())
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(content) = content {
                // Find ID from vocabulary
                let id = self.vocabulary.encode(&content).unwrap_or(-1);
                
                let special_type = match stype {
                    "bos" => SpecialTokenType::SequenceStart,
                    "eos" => SpecialTokenType::SequenceEnd,
                    "unk" => SpecialTokenType::Unknown,
                    "pad" => SpecialTokenType::Padding,
                    _ => SpecialTokenType::Custom { tier: 500, depth: 100 },
                };

                self.special_tokens.push(NeuralSpecialTokenDef {
                    token_type: special_type,
                    id,
                    content,
                    add_token,
                    aliases: Vec::new(),
                });
            }
        }

        Ok(())
    }

    /// Enhance from generation_config.json
    fn enhance_from_generation_config<P: AsRef<Path>>(&mut self, dir: P) -> Result<(), ConverterError> {
        let path = dir.as_ref().join("generation_config.json");
        let mut file = File::open(&path)
            .map_err(|e| ConverterError::FileError(format!("Cannot open generation_config.json: {}", e)))?;

        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| ConverterError::FileError(format!("Cannot read generation_config.json: {}", e)))?;

        let config: HashMap<String, Value> = serde_json::from_str(&content)
            .map_err(|e| ConverterError::ConfigParseError(format!("JSON parse error: {}", e)))?;

        // Update special token IDs
        let special_types = vec!["bos", "eos", "unk", "pad"];
        for stype in special_types {
            let key = format!("{}_token_id", stype);
            
            if let Some(val) = config.get(&key) {
                // Handle both single ID and array of IDs
                let ids: Vec<i32> = if let Some(arr) = val.as_array() {
                    arr.iter().filter_map(|v| v.as_i64().map(|n| n as i32)).collect()
                } else if let Some(n) = val.as_i64() {
                    vec![n as i32]
                } else {
                    continue;
                };

                // Find and update special token
                for special in &mut self.special_tokens {
                    if special.token_type.as_str().to_lowercase() == stype {
                        if !ids.is_empty() {
                            special.id = ids[0];
                            special.aliases = ids[1..].to_vec();
                        }
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get 3D tokenization context
    pub fn to_tokenization_context(&self) -> NeuralTokenizationContext {
        let vocab_size = self.vocabulary.len();
        let depth = ((vocab_size as f32).log2() as u32).max(64).min(512);
        
        NeuralTokenizationContext::new(
            1024,                    // context_width
            1024,                    // context_height
            depth,                   // context_depth based on vocab size
            1.0,                     // guidance_scale
        ).with_max_length(4096)
    }
}

impl Default for NeuralTokenizerBundle {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-tokenizer type detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizerType {
    Default,
    SentencePiece,
    LlamaBpe,
    DeepSeekLlama,
    DeepSeekCoder,
    Qwen2,
    Qwen35,
}

impl PreTokenizerType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::SentencePiece => "sentencepiece",
            Self::LlamaBpe => "llama-bpe",
            Self::DeepSeekLlama => "deepseek-llm",
            Self::DeepSeekCoder => "deepseek-coder",
            Self::Qwen2 => "qwen2",
            Self::Qwen35 => "qwen35",
        }
    }
}

/// Special token definition
#[derive(Debug, Clone)]
pub struct NeuralSpecialTokenDef {
    pub token_type: SpecialTokenType,
    pub id: i32,
    pub content: String,
    pub add_token: bool,
    pub aliases: Vec<i32>,
}

impl NeuralSpecialTokenDef {
    /// Get key for GGML format
    pub fn to_ggml_key(&self) -> String {
        match self.token_type {
            SpecialTokenType::SequenceStart => "bos".to_string(),
            SpecialTokenType::SequenceEnd => "eos".to_string(),
            SpecialTokenType::Unknown => "unknown".to_string(),
            SpecialTokenType::Padding => "padding".to_string(),
            SpecialTokenType::Custom { .. } => format!("special_{}", self.id),
        }
    }
}

/// BPE merge rule
#[derive(Debug, Clone)]
pub struct NeuralBpeMerge {
    pub first: String,
    pub second: String,
    pub priority: u32,
}

// Internal structs for parsing

#[derive(Default)]
struct SentencePieceModel {
    pieces: Vec<SentencePiece>,
}

#[derive(Default)]
struct SentencePiece {
    content: String,
    score: f32,
    piece_type: i32,
}

#[derive(Deserialize)]
struct HuggingFaceTokenizer {
    #[serde(rename = "added_tokens")]
    added_tokens: Vec<HfAddedToken>,
    model: HfModel,
    #[serde(rename = "pre_tokenizer")]
    pre_tokenizer: HfPreTokenizer,
}

#[derive(Deserialize)]
struct HfAddedToken {
    id: i32,
    content: String,
    special: bool,
}

#[derive(Deserialize)]
struct HfModel {
    #[serde(rename = "type")]
    type_: String,
    vocab: HashMap<String, i32>,
    merges: Value,
}

#[derive(Deserialize, Default)]
struct HfPreTokenizer {
    #[serde(rename = "pre_tokenizers")]
    pre_tokenizers: Vec<HfPreTokenizerItem>,
}

#[derive(Deserialize, Default)]
struct HfPreTokenizerItem {
    #[serde(rename = "type")]
    type_: String,
    pattern: HfPattern,
}

#[derive(Deserialize, Default)]
struct HfPattern {
    #[serde(rename = "Regex")]
    regex: String,
}

struct TokenEntry {
    content: String,
    id: i32,
    special: bool,
    user_defined: bool,
}

/// 3D tokenizer converter utility functions
pub mod tokenizer_utils {
    use super::*;

    /// Convert special token type string to enum
    pub fn parse_special_type(s: &str) -> Option<SpecialTokenType> {
        match s.to_lowercase().as_str() {
            "bos" | "beginningofsequence" | "start" => Some(SpecialTokenType::SequenceStart),
            "eos" | "endofsequence" | "end" => Some(SpecialTokenType::SequenceEnd),
            "unk" | "unknown" => Some(SpecialTokenType::Unknown),
            "pad" | "padding" => Some(SpecialTokenType::Padding),
            _ => Some(SpecialTokenType::Custom { tier: 500, depth: 50 }),
        }
    }

    /// Calculate 3D spatial coordinate for token position
    pub fn calculate_token_coordinate(
        token_id: i32,
        vocab_size: usize,
        seq_position: usize,
    ) -> VocabCoordinate {
        let id_norm = token_id as f64 / vocab_size as f64;
        let seq_norm = seq_position as f64 / 1000.0; // Assume max 1k context

        VocabCoordinate::new(
            token_id as u64,
            ((1.0 - id_norm) * 1000.0) as u32, // High frequency = high tier
            (seq_norm * 100.0) as u16,
            (0.5 + id_norm * 0.5) as f32,
        )
    }

    /// Merge rules to string format for serialization
    pub fn serialize_merges(merges: &[NeuralBpeMerge]) -> Vec<String> {
        merges.iter()
            .map(|m| format!("{} {}", m.first, m.second))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_tokenizer_bundle_new() {
        let bundle = NeuralTokenizerBundle::new();
        assert_eq!(bundle.vocabulary.len(), 0);
        assert!(bundle.merge_rules.is_empty());
    }

    #[test]
    fn test_detect_pretokenizer_llama() {
        // Simulate llama-bpe pre-tokenizer detection
        let tokenizer = HuggingFaceTokenizer {
            added_tokens: vec![],
            model: HfModel {
                type_: "BPE".to_string(),
                vocab: HashMap::new(),
                merges: Value::Array(vec![]),
            },
            pre_tokenizer: HfPreTokenizer {
                pre_tokenizers: vec![HfPreTokenizerItem {
                    type_: "Split".to_string(),
                    pattern: HfPattern {
                        regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(!?)$".to_string(),
                    },
                }],
            },
        };

        let ptype = NeuralTokenizerBundle::detect_pretokenizer(&tokenizer);
        assert_eq!(ptype, PreTokenizerType::LlamaBpe);
    }

    #[test]
    fn test_parse_merges_string_format() {
        let json = serde_json::json!(["a b", "c d", "e f"]);
        let merges = NeuralTokenizerBundle::parse_merges(&json).unwrap();
        assert_eq!(merges.len(), 3);
        assert_eq!(merges[0].first, "a");
        assert_eq!(merges[0].second, "b");
    }

    #[test]
    fn test_parse_merges_array_format() {
        let json = serde_json::json!([["a", "b"], ["c", "d"]]);
        let merges = NeuralTokenizerBundle::parse_merges(&json).unwrap();
        assert_eq!(merges.len(), 2);
        assert_eq!(merges[0].first, "a");
        assert_eq!(merges[0].second, "b");
    }

    #[test]
    fn test_tokenizer_utils() {
        assert_eq!(
            tokenizer_utils::parse_special_type("bos"),
            Some(SpecialTokenType::SequenceStart)
        );
        assert_eq!(
            tokenizer_utils::parse_special_type("eos"),
            Some(SpecialTokenType::SequenceEnd)
        );

        let coord = tokenizer_utils::calculate_token_coordinate(100, 50000, 50);
        assert_eq!(coord.linear_idx, 100);
        assert!(coord.importance > 0.0);
    }

    #[test]
    fn test_special_token_ggml_key() {
        let special = NeuralSpecialTokenDef {
            token_type: SpecialTokenType::SequenceStart,
            id: 1,
            content: "<s>".to_string(),
            add_token: true,
            aliases: vec![],
        };
        assert_eq!(special.to_ggml_key(), "bos");
    }

    #[test]
    fn test_pretokenizer_types() {
        assert_eq!(PreTokenizerType::LlamaBpe.as_str(), "llama-bpe");
        assert_eq!(PreTokenizerType::DeepSeekCoder.as_str(), "deepseek-coder");
    }
}
