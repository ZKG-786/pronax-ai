use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::convert::pronax_converter_core::{ConverterError, ConversionCoordinate};
use crate::tokenizer::pronax_vocabulary::{
    NeuralVocabulary, NeuralVocabEntry, TokenSemanticType, VocabCoordinate, SpecialTokenType,
};
use crate::tokenizer::pronax_tokenizer_trait::{NeuralTokenCategory, NeuralTokenizationContext};

/// 3D SentencePiece tokenizer converter with spatial metadata
pub struct SentencePieceConverter;

impl SentencePieceConverter {
    /// Parse SentencePiece model with 3D spatial awareness
    pub fn parse_model<P: AsRef<Path>>(dir: P) -> Result<NeuralTokenizerArchive, ConverterError> {
        let path = dir.as_ref();
        debug!("Parsing SentencePiece model from {:?}", path);

        // Load additional special tokens first
        let extra_tokens = Self::load_special_token_map(path)?;

        // Parse the main tokenizer.model protobuf
        let model_path = path.join("tokenizer.model");
        let mut file = File::open(&model_path)
            .map_err(|e| ConverterError::FileError(format!("Cannot open tokenizer.model: {}", e)))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| ConverterError::FileError(format!("Cannot read tokenizer.model: {}", e)))?;

        // Decode protobuf
        let sp_model = Self::decode_protobuf(&buffer)?;

        // Build 3D vocabulary
        let mut archive = NeuralTokenizerArchive::new();
        archive.spatial_origin = ConversionCoordinate::high_precision();

        for (idx, segment) in sp_model.segments.iter().enumerate() {
            let id = idx as i32;
            let token_type = Self::classify_segment(segment, &extra_tokens);
            
            // 3D spatial coordinate based on position and type
            let coordinate = Self::compute_spatial_coord(idx, segment.score, &token_type);
            
            let entry = NeuralVocabEntry::new(
                &segment.lexeme,
                id,
                token_type,
                coordinate,
                segment.score,
            );

            let neural_category = match token_type {
                TokenSemanticType::Control => NeuralTokenCategory::Control,
                TokenSemanticType::UserDefined => NeuralTokenCategory::UserDefined,
                TokenSemanticType::Unknown => NeuralTokenCategory::Unknown,
                TokenSemanticType::Byte => NeuralTokenCategory::Unused,
                _ => NeuralTokenCategory::Normal,
            };

            archive.vocabulary.add_token(entry.lexeme, id, neural_category);
            archive.token_metadata.insert(id, TokenSpatialMeta {
                original_type: segment.piece_type,
                confidence: segment.score,
                coordinate,
                is_special: matches!(token_type, TokenSemanticType::Control | TokenSemanticType::UserDefined),
            });
        }

        // Load added_tokens.json if present
        Self::integrate_added_tokens(path, &mut archive)?;

        debug!(
            "SentencePiece model loaded: {} tokens, {} special",
            archive.vocabulary.len(),
            archive.token_metadata.values().filter(|m| m.is_special).count()
        );

        Ok(archive)
    }

    /// Decode SentencePiece protobuf format
    fn decode_protobuf(data: &[u8]) -> Result<SentencePieceArchive, ConverterError> {
        let mut archive = SentencePieceArchive::default();
        let mut offset = 0;

        while offset < data.len() {
            if data[offset] == 0 {
                offset += 1;
                continue;
            }

            // Parse field tag
            let (field_num, wire_type, bytes_read) = Self::parse_tag(&data[offset..])?;
            offset += bytes_read;

            match (field_num, wire_type) {
                (1, 2) => { // Pieces (length-delimited)
                    let len = Self::parse_varint(&data[offset..])?;
                    offset += Self::varint_len(len);
                    if offset + len <= data.len() {
                        if let Some(segment) = Self::parse_segment(&data[offset..offset + len]) {
                            archive.segments.push(segment);
                        }
                        offset += len;
                    }
                }
                (2, 0) => { // Trainer spec (varint) - skip
                    let (_, len) = Self::parse_varint_full(&data[offset..])?;
                    offset += len;
                }
                (3, 0) => { // Self-test samples (varint) - skip
                    let (_, len) = Self::parse_varint_full(&data[offset..])?;
                    offset += len;
                }
                (4, 2) => { // Normalizer spec (length-delimited) - skip
                    let len = Self::parse_varint(&data[offset..])?;
                    offset += Self::varint_len(len) + len;
                }
                _ => {
                    // Skip unknown field
                    offset += 1;
                }
            }
        }

        // Sort by score descending for proper vocabulary ordering
        archive.segments.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Re-assign IDs based on sorted order
        for (idx, seg) in archive.segments.iter_mut().enumerate() {
            seg.id = idx as i32;
        }

        Ok(archive)
    }

    /// Parse protobuf field tag
    fn parse_tag(data: &[u8]) -> Result<(u32, u8, usize), ConverterError> {
        if data.is_empty() {
            return Err(ConverterError::TokenizerError("Empty protobuf data".to_string()));
        }
        let (tag, len) = Self::parse_varint_full(data)?;
        let field_num = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;
        Ok((field_num, wire_type, len))
    }

    /// Parse variable-length integer
    fn parse_varint(data: &[u8]) -> Result<usize, ConverterError> {
        let (val, _) = Self::parse_varint_full(data)?;
        Ok(val as usize)
    }

    fn parse_varint_full(data: &[u8]) -> Result<(u64, usize), ConverterError> {
        let mut result: u64 = 0;
        let mut shift = 0;
        let mut pos = 0;

        loop {
            if pos >= data.len() || pos >= 10 {
                return Err(ConverterError::TokenizerError("Invalid varint".to_string()));
            }
            let byte = data[pos];
            result |= ((byte & 0x7F) as u64) << shift;
            pos += 1;
            if (byte & 0x80) == 0 {
                break;
            }
            shift += 7;
        }

        Ok((result, pos))
    }

    fn varint_len(val: usize) -> usize {
        let mut len = 1;
        let mut v = val >> 7;
        while v > 0 {
            len += 1;
            v >>= 7;
        }
        len
    }

    /// Parse individual SentencePiece segment
    fn parse_segment(data: &[u8]) -> Option<SentencePieceSegment> {
        let mut segment = SentencePieceSegment::default();
        let mut offset = 0;

        while offset < data.len() {
            if data[offset] == 0 {
                offset += 1;
                continue;
            }

            let (field_num, wire_type, tag_len) = Self::parse_tag(&data[offset..]).ok()?;
            offset += tag_len;

            match (field_num, wire_type) {
                (1, 2) => { // Piece (string)
                    let len = Self::parse_varint(&data[offset..]).ok()?;
                    offset += Self::varint_len(len);
                    if offset + len <= data.len() {
                        segment.lexeme = String::from_utf8_lossy(&data[offset..offset + len]).to_string();
                        offset += len;
                    }
                }
                (2, 5) => { // Score (float, fixed32)
                    if offset + 4 <= data.len() {
                        let bytes: [u8; 4] = [
                            data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
                        ];
                        segment.score = f32::from_le_bytes(bytes);
                        offset += 4;
                    }
                }
                (3, 0) => { // Type (varint)
                    let (t, len) = Self::parse_varint_full(&data[offset..]).ok()?;
                    segment.piece_type = t as i32;
                    offset += len;
                }
                _ => break,
            }
        }

        if !segment.lexeme.is_empty() {
            Some(segment)
        } else {
            None
        }
    }

    /// Classify segment type with special token handling
    fn classify_segment(segment: &SentencePieceSegment, extra_tokens: &[SpecialTokenDef]) -> TokenSemanticType {
        // Check protobuf type first
        match segment.piece_type {
            0 => TokenSemanticType::Unknown,
            1 => {
                // Normal token - check for special overrides
                let gemma3_specials = [
                    "<end_of_turn>", "<start_of_turn>",
                    "<start_function_declaration>", "<end_function_declaration>",
                    "<start_function_call>", "<end_function_call>",
                    "<start_function_response>", "<end_function_response>",
                    "<escape>",
                ];
                
                if gemma3_specials.contains(&segment.lexeme.as_str()) {
                    return TokenSemanticType::Control;
                }

                // Check additional special tokens
                for special in extra_tokens {
                    if special.content == segment.lexeme {
                        return TokenSemanticType::Control;
                    }
                }

                TokenSemanticType::Normal
            }
            2 => TokenSemanticType::Control,
            3 => TokenSemanticType::UserDefined,
            4 => TokenSemanticType::Byte,
            5 => TokenSemanticType::Unused,
            6 => TokenSemanticType::Control, // whitespace
            _ => TokenSemanticType::Unknown,
        }
    }

    /// Compute 3D spatial coordinate for token
    fn compute_spatial_coord(idx: usize, score: f32, token_type: &TokenSemanticType) -> VocabCoordinate {
        let importance = match token_type {
            TokenSemanticType::Control => 1.0,
            TokenSemanticType::UserDefined => 0.95,
            TokenSemanticType::Normal => 0.5 + (score.abs().min(1.0) * 0.5),
            _ => 0.3,
        };

        VocabCoordinate::new(
            idx as u64,
            ((1.0 - (idx as f32 / 100000.0).min(1.0)) * 1000.0) as u32,
            (score.abs() * 100.0) as u16,
            importance,
        )
    }

    /// Load special_tokens_map.json
    fn load_special_token_map<P: AsRef<Path>>(dir: P) -> Result<Vec<SpecialTokenDef>, ConverterError> {
        let path = dir.as_ref().join("special_tokens_map.json");
        
        let mut file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(ConverterError::FileError(format!("Cannot open special_tokens_map.json: {}", e))),
        };

        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| ConverterError::FileError(format!("Cannot read special_tokens_map.json: {}", e)))?;

        let wrapper: SpecialTokenWrapper = serde_json::from_str(&content)
            .map_err(|e| ConverterError::ConfigParseError(format!("JSON parse error: {}", e)))?;

        let tokens = match wrapper.additional_special_tokens {
            serde_json::Value::Array(arr) => {
                arr.iter()
                    .filter_map(|v| {
                        if let Some(s) = v.as_str() {
                            Some(SpecialTokenDef {
                                content: s.to_string(),
                                lstrip: false,
                                normalized: true,
                                rstrip: false,
                                single_word: false,
                            })
                        } else if let Some(obj) = v.as_object() {
                            Some(SpecialTokenDef {
                                content: obj.get("content")?.as_str()?.to_string(),
                                lstrip: obj.get("lstrip")?.as_bool().unwrap_or(false),
                                normalized: obj.get("normalized")?.as_bool().unwrap_or(true),
                                rstrip: obj.get("rstrip")?.as_bool().unwrap_or(false),
                                single_word: obj.get("single_word")?.as_bool().unwrap_or(false),
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            }
            _ => Vec::new(),
        };

        debug!("Loaded {} additional special tokens", tokens.len());
        Ok(tokens)
    }

    /// Integrate added_tokens.json into vocabulary
    fn integrate_added_tokens<P: AsRef<Path>>(dir: P, archive: &mut NeuralTokenizerArchive) -> Result<(), ConverterError> {
        let path = dir.as_ref().join("added_tokens.json");
        
        let mut file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(ConverterError::FileError(format!("Cannot open added_tokens.json: {}", e))),
        };

        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| ConverterError::FileError(format!("Cannot read added_tokens.json: {}", e)))?;

        let added: HashMap<String, i32> = serde_json::from_str(&content)
            .map_err(|e| ConverterError::ConfigParseError(format!("JSON parse error: {}", e)))?;

        // Sort by ID
        let mut entries: Vec<(i32, String)> = added.into_iter().map(|(k, v)| (v, k)).collect();
        entries.sort_by_key(|(id, _)| *id);

        for (id, content) in entries {
            let expected_pos = archive.vocabulary.len() as i32;
            
            if id < expected_pos {
                // Check for duplicates
                if let Some(existing) = archive.vocabulary.decode(id) {
                    if existing == content {
                        warn!("Duplicate token '{}' at ID {}", content, id);
                        continue;
                    }
                    return Err(ConverterError::TokenizerError(
                        format!("Token mismatch: '{}' != '{}' at pos [{}]", content, existing, id)
                    ));
                }
            } else if id != expected_pos {
                return Err(ConverterError::TokenizerError(
                    format!("Invalid token ID: [{}] at pos [{}]", id, expected_pos)
                ));
            }

            // Add as user-defined token
            let coord = VocabCoordinate::new(id as u64, 100, 50, 0.95);
            archive.vocabulary.add_token(
                content.clone(),
                id,
                NeuralTokenCategory::UserDefined,
            );
            archive.token_metadata.insert(id, TokenSpatialMeta {
                original_type: 3, // UserDefined
                confidence: -1000.0,
                coordinate: coord,
                is_special: true,
            });
        }

        Ok(())
    }
}

/// Neural tokenizer archive with 3D spatial metadata
#[derive(Debug, Clone)]
pub struct NeuralTokenizerArchive {
    pub vocabulary: NeuralVocabulary,
    pub token_metadata: HashMap<i32, TokenSpatialMeta>,
    pub spatial_origin: ConversionCoordinate,
    pub model_format: SentencePieceFormat,
}

impl NeuralTokenizerArchive {
    pub fn new() -> Self {
        Self {
            vocabulary: NeuralVocabulary::new(),
            token_metadata: HashMap::new(),
            spatial_origin: ConversionCoordinate::standard(),
            model_format: SentencePieceFormat::Standard,
        }
    }

    /// Convert to 3D tokenization context
    pub fn to_context(&self) -> NeuralTokenizationContext {
        let vocab_size = self.vocabulary.len();
        let depth = ((vocab_size as f32).log2() as u32).max(64).min(512);
        
        NeuralTokenizationContext::new(
            2048,  // width
            2048,  // height
            depth, // depth based on vocab
            self.spatial_origin.quality_score,
        ).with_max_length(8192)
    }

    /// Get token spatial metadata
    pub fn get_spatial_meta(&self, id: i32) -> Option<&TokenSpatialMeta> {
        self.token_metadata.get(&id)
    }

    /// Get all special tokens
    pub fn special_tokens(&self) -> Vec<(i32, &TokenSpatialMeta)> {
        self.token_metadata
            .iter()
            .filter(|(_, m)| m.is_special)
            .map(|(id, m)| (*id, m))
            .collect()
    }
}

impl Default for NeuralTokenizerArchive {
    fn default() -> Self {
        Self::new()
    }
}

/// Spatial metadata for individual token
#[derive(Debug, Clone, Copy)]
pub struct TokenSpatialMeta {
    pub original_type: i32,
    pub confidence: f32,
    pub coordinate: VocabCoordinate,
    pub is_special: bool,
}

/// SentencePiece protobuf segment
#[derive(Debug, Default, Clone)]
struct SentencePieceSegment {
    pub id: i32,
    pub lexeme: String,
    pub score: f32,
    pub piece_type: i32,
}

/// SentencePiece model archive
#[derive(Debug, Default)]
struct SentencePieceArchive {
    pub segments: Vec<SentencePieceSegment>,
}

/// SentencePiece format variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentencePieceFormat {
    Standard,
    Llama,
    Gemma3,
    Custom,
}

/// Special token definition
#[derive(Debug, Clone, Deserialize)]
struct SpecialTokenDef {
    pub content: String,
    #[serde(default)]
    pub lstrip: bool,
    #[serde(default = "default_normalized")]
    pub normalized: bool,
    #[serde(default)]
    pub rstrip: bool,
    #[serde(default)]
    pub single_word: bool,
}

fn default_normalized() -> bool {
    true
}

/// Wrapper for special_tokens_map.json
#[derive(Debug, Deserialize)]
struct SpecialTokenWrapper {
    #[serde(rename = "additional_special_tokens")]
    pub additional_special_tokens: serde_json::Value,
}

/// 3D SPM converter utilities
pub mod spm_utils {
    use super::*;

    /// Detect SentencePiece format variant based on tokens
    pub fn detect_format(archive: &NeuralTokenizerArchive) -> SentencePieceFormat {
        let has_gemma3_tokens = archive.token_metadata.values().any(|m| {
            matches!(m.original_type, 2) // Control tokens
        });

        if has_gemma3_tokens {
            // Check for gemma3 specific tokens
            let has_function_tokens = archive.vocabulary.encode("<start_of_turn>").is_some();
            if has_function_tokens {
                return SentencePieceFormat::Gemma3;
            }
        }

        SentencePieceFormat::Standard
    }

    /// Compute vocabulary statistics with 3D spatial analysis
    pub fn analyze_spatial_distribution(archive: &NeuralTokenizerArchive) -> SpatialStats {
        let mut total_importance = 0.0;
        let mut special_count = 0;
        let mut normal_count = 0;
        let mut depth_sum = 0u64;

        for (id, meta) in &archive.token_metadata {
            total_importance += meta.coordinate.importance;
            depth_sum += meta.coordinate.semantic_depth as u64;
            
            if meta.is_special {
                special_count += 1;
            } else {
                normal_count += 1;
            }
        }

        let total = archive.token_metadata.len() as f32;
        SpatialStats {
            avg_importance: if total > 0.0 { total_importance / total } else { 0.0 },
            special_ratio: if total > 0.0 { special_count as f32 / total } else { 0.0 },
            avg_depth: if total > 0.0 { (depth_sum as f32 / total) as u16 } else { 0 },
            spatial_variance: 0.0, // Would need more calculation
        }
    }

    /// Export to GGML-compatible format
    pub fn to_ggml_compatible(archive: &NeuralTokenizerArchive) -> GgmlVocabExport {
        let mut tokens = Vec::new();
        let mut scores = Vec::new();
        let mut types = Vec::new();

        // Collect in ID order
        let mut ids: Vec<i32> = archive.token_metadata.keys().copied().collect();
        ids.sort();

        for id in ids {
            if let Some(lexeme) = archive.vocabulary.decode(id) {
                tokens.push(lexeme.to_string());
                
                if let Some(meta) = archive.token_metadata.get(&id) {
                    scores.push(meta.confidence);
                    types.push(meta.original_type);
                } else {
                    scores.push(0.0);
                    types.push(1); // Normal
                }
            }
        }

        GgmlVocabExport {
            model_name: "llama".to_string(), // SPM defaults to llama format
            tokens,
            scores,
            types,
        }
    }
}

/// Spatial distribution statistics
#[derive(Debug, Clone, Copy)]
pub struct SpatialStats {
    pub avg_importance: f32,
    pub special_ratio: f32,
    pub avg_depth: u16,
    pub spatial_variance: f32,
}

/// GGML vocabulary export format
#[derive(Debug, Clone)]
pub struct GgmlVocabExport {
    pub model_name: String,
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub types: Vec<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_varint_parsing() {
        // Test varint encoding/decoding
        let test_cases: Vec<(u64, Vec<u8>)> = vec![
            (0, vec![0]),
            (1, vec![1]),
            (127, vec![0x7F]),
            (128, vec![0x80, 0x01]),
            (255, vec![0xFF, 0x01]),
            (16384, vec![0x80, 0x80, 0x01]),
        ];

        for (expected, bytes) in test_cases {
            let (val, len) = SentencePieceConverter::parse_varint_full(&bytes).unwrap();
            assert_eq!(val, expected);
            assert_eq!(len, bytes.len());
        }
    }

    #[test]
    fn test_spatial_coord_computation() {
        let coord = SentencePieceConverter::compute_spatial_coord(100, 0.5, &TokenSemanticType::Normal);
        assert_eq!(coord.linear_idx, 100);
        assert!(coord.importance > 0.0);
        assert!(coord.importance <= 1.0);

        let control_coord = SentencePieceConverter::compute_spatial_coord(5, 0.0, &TokenSemanticType::Control);
        assert_eq!(control_coord.importance, 1.0);
    }

    #[test]
    fn test_classify_segment() {
        let segment = SentencePieceSegment {
            id: 0,
            lexeme: "<end_of_turn>".to_string(),
            score: 0.0,
            piece_type: 1, // Normal but should be Control for gemma3
        };
        
        let token_type = SentencePieceConverter::classify_segment(&segment, &[]);
        assert_eq!(token_type, TokenSemanticType::Control);
    }

    #[test]
    fn test_neural_archive() {
        let mut archive = NeuralTokenizerArchive::new();
        
        archive.vocabulary.add_token("hello", 0, NeuralTokenCategory::Normal);
        archive.token_metadata.insert(0, TokenSpatialMeta {
            original_type: 1,
            confidence: 0.5,
            coordinate: VocabCoordinate::origin(),
            is_special: false,
        });

        assert_eq!(archive.vocabulary.len(), 1);
        assert!(archive.get_spatial_meta(0).is_some());
    }

    #[test]
    fn test_special_token_parsing() {
        let json = r#"{"additional_special_tokens": ["<tool>", {"content": "<think>", "lstrip": true}]}"#;
        let wrapper: SpecialTokenWrapper = serde_json::from_str(json).unwrap();
        
        match wrapper.additional_special_tokens {
            serde_json::Value::Array(arr) => {
                assert_eq!(arr.len(), 2);
            }
            _ => panic!("Expected array"),
        }
    }

    #[test]
    fn test_ggml_export() {
        let mut archive = NeuralTokenizerArchive::new();
        archive.vocabulary.add_token("test", 0, NeuralTokenCategory::Normal);
        archive.token_metadata.insert(0, TokenSpatialMeta {
            original_type: 1,
            confidence: 0.5,
            coordinate: VocabCoordinate::origin(),
            is_special: false,
        });

        let export = spm_utils::to_ggml_compatible(&archive);
        assert_eq!(export.tokens.len(), 1);
        assert_eq!(export.tokens[0], "test");
    }
}
