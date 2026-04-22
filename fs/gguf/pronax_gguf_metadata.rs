use std::any::Any;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// Neural metadata value with 3D spatial context and type safety
/// Completely redesigned from Go's `any` based Value to Rust's type-safe enum
#[derive(Debug, Clone, PartialEq)]
pub enum TitanMetadataValue {
    // Scalar types
    NeuralU8(u8),
    NeuralI8(i8),
    NeuralU16(u16),
    NeuralI16(i16),
    NeuralU32(u32),
    NeuralI32(i32),
    NeuralU64(u64),
    NeuralI64(i64),
    NeuralF32(f32),
    NeuralF64(f64),
    NeuralBool(bool),
    NeuralText(String),
    
    // 3D spatial-enhanced scalar types (unique to ProNax)
    SpatialU8(u8, SpatialTensorMetadata),
    SpatialI8(i8, SpatialTensorMetadata),
    SpatialU16(u16, SpatialTensorMetadata),
    SpatialI16(i16, SpatialTensorMetadata),
    SpatialU32(u32, SpatialTensorMetadata),
    SpatialI32(i32, SpatialTensorMetadata),
    SpatialU64(u64, SpatialTensorMetadata),
    SpatialI64(i64, SpatialTensorMetadata),
    SpatialF32(f32, SpatialTensorMetadata),
    SpatialF64(f64, SpatialTensorMetadata),
    SpatialBool(bool, SpatialTensorMetadata),
    SpatialText(String, SpatialTensorMetadata),
    
    // Array types
    ArrayU8(Vec<u8>),
    ArrayI8(Vec<i8>),
    ArrayU16(Vec<u16>),
    ArrayI16(Vec<i16>),
    ArrayU32(Vec<u32>),
    ArrayI32(Vec<i32>),
    ArrayU64(Vec<u64>),
    ArrayI64(Vec<i64>),
    ArrayF32(Vec<f32>),
    ArrayF64(Vec<f64>),
    ArrayBool(Vec<bool>),
    ArrayText(Vec<String>),
    
    // 3D spatial-enhanced array types (unique to ProNax)
    SpatialArrayU8(Vec<u8>, SpatialTensorMetadata),
    SpatialArrayI8(Vec<i8>, SpatialTensorMetadata),
    SpatialArrayU16(Vec<u16>, SpatialTensorMetadata),
    SpatialArrayI16(Vec<i16>, SpatialTensorMetadata),
    SpatialArrayU32(Vec<u32>, SpatialTensorMetadata),
    SpatialArrayI32(Vec<i32>, SpatialTensorMetadata),
    SpatialArrayU64(Vec<u64>, SpatialTensorMetadata),
    SpatialArrayI64(Vec<i64>, SpatialTensorMetadata),
    SpatialArrayF32(Vec<f32>, SpatialTensorMetadata),
    SpatialArrayF64(Vec<f64>, SpatialTensorMetadata),
    SpatialArrayBool(Vec<bool>, SpatialTensorMetadata),
    SpatialArrayText(Vec<String>, SpatialTensorMetadata),
    
    // Nested metadata (for complex structures)
    NestedMetadata(Box<TitanMetadataEntry>),
    NestedArray(Vec<TitanMetadataValue>),
    
    // Zero-copy reference type
    SharedValue(Arc<TitanMetadataValue>),
    
    // Null/empty value
    NeuralNull,
}

impl TitanMetadataValue {
    /// Create a value with automatic 3D spatial context detection
    pub fn with_spatial_context(self, width: u32, height: u32, depth: u32) -> Self {
        let spatial = SpatialTensorMetadata::new(width, height, depth);
        
        match self {
            Self::NeuralU8(v) => Self::SpatialU8(v, spatial),
            Self::NeuralI8(v) => Self::SpatialI8(v, spatial),
            Self::NeuralU16(v) => Self::SpatialU16(v, spatial),
            Self::NeuralI16(v) => Self::SpatialI16(v, spatial),
            Self::NeuralU32(v) => Self::SpatialU32(v, spatial),
            Self::NeuralI32(v) => Self::SpatialI32(v, spatial),
            Self::NeuralU64(v) => Self::SpatialU64(v, spatial),
            Self::NeuralI64(v) => Self::SpatialI64(v, spatial),
            Self::NeuralF32(v) => Self::SpatialF32(v, spatial),
            Self::NeuralF64(v) => Self::SpatialF64(v, spatial),
            Self::NeuralBool(v) => Self::SpatialBool(v, spatial),
            Self::NeuralText(v) => Self::SpatialText(v, spatial),
            Self::ArrayU8(v) => Self::SpatialArrayU8(v, spatial),
            Self::ArrayI8(v) => Self::SpatialArrayI8(v, spatial),
            Self::ArrayU16(v) => Self::SpatialArrayU16(v, spatial),
            Self::ArrayI16(v) => Self::SpatialArrayI16(v, spatial),
            Self::ArrayU32(v) => Self::SpatialArrayU32(v, spatial),
            Self::ArrayI32(v) => Self::SpatialArrayI32(v, spatial),
            Self::ArrayU64(v) => Self::SpatialArrayU64(v, spatial),
            Self::ArrayI64(v) => Self::SpatialArrayI64(v, spatial),
            Self::ArrayF32(v) => Self::SpatialArrayF32(v, spatial),
            Self::ArrayF64(v) => Self::SpatialArrayF64(v, spatial),
            Self::ArrayBool(v) => Self::SpatialArrayBool(v, spatial),
            Self::ArrayText(v) => Self::SpatialArrayText(v, spatial),
            _ => self, // Already spatial or complex type
        }
    }

    /// Get spatial metadata if available
    pub fn spatial_metadata(&self) -> Option<&SpatialTensorMetadata> {
        match self {
            Self::SpatialU8(_, s) |
            Self::SpatialI8(_, s) |
            Self::SpatialU16(_, s) |
            Self::SpatialI16(_, s) |
            Self::SpatialU32(_, s) |
            Self::SpatialI32(_, s) |
            Self::SpatialU64(_, s) |
            Self::SpatialI64(_, s) |
            Self::SpatialF32(_, s) |
            Self::SpatialF64(_, s) |
            Self::SpatialBool(_, s) |
            Self::SpatialText(_, s) |
            Self::SpatialArrayU8(_, s) |
            Self::SpatialArrayI8(_, s) |
            Self::SpatialArrayU16(_, s) |
            Self::SpatialArrayI16(_, s) |
            Self::SpatialArrayU32(_, s) |
            Self::SpatialArrayI32(_, s) |
            Self::SpatialArrayU64(_, s) |
            Self::SpatialArrayI64(_, s) |
            Self::SpatialArrayF32(_, s) |
            Self::SpatialArrayF64(_, s) |
            Self::SpatialArrayBool(_, s) |
            Self::SpatialArrayText(_, s) => Some(s),
            _ => None,
        }
    }

    /// Check if value has spatial context
    pub fn has_spatial_context(&self) -> bool {
        self.spatial_metadata().is_some()
    }

    /// Get as signed integer (converts various int types)
    pub fn as_signed_int(&self) -> Option<i64> {
        match self {
            Self::NeuralI8(v) => Some(*v as i64),
            Self::NeuralI16(v) => Some(*v as i64),
            Self::NeuralI32(v) => Some(*v as i64),
            Self::NeuralI64(v) => Some(*v),
            Self::NeuralU8(v) => Some(*v as i64),
            Self::NeuralU16(v) => Some(*v as i64),
            Self::NeuralU32(v) => Some(*v as i64),
            Self::NeuralU64(v) => Some(*v as i64),
            Self::NeuralBool(v) => Some(if *v { 1 } else { 0 }),
            Self::SpatialI8(v, _) |
            Self::SpatialU8(v, _) => Some(*v as i64),
            Self::SpatialI16(v, _) |
            Self::SpatialU16(v, _) => Some(*v as i64),
            Self::SpatialI32(v, _) |
            Self::SpatialU32(v, _) => Some(*v as i64),
            Self::SpatialI64(v, _) |
            Self::SpatialU64(v, _) => Some(*v as i64),
            Self::SpatialBool(v, _) => Some(if *v { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Get as unsigned integer
    pub fn as_unsigned_int(&self) -> Option<u64> {
        match self {
            Self::NeuralU8(v) => Some(*v as u64),
            Self::NeuralU16(v) => Some(*v as u64),
            Self::NeuralU32(v) => Some(*v as u64),
            Self::NeuralU64(v) => Some(*v),
            Self::NeuralI8(v) if *v >= 0 => Some(*v as u64),
            Self::NeuralI16(v) if *v >= 0 => Some(*v as u64),
            Self::NeuralI32(v) if *v >= 0 => Some(*v as u64),
            Self::NeuralI64(v) if *v >= 0 => Some(*v as u64),
            Self::NeuralBool(v) => Some(if *v { 1 } else { 0 }),
            Self::SpatialU8(v, _) |
            Self::SpatialI8(v, _) if *v >= 0 => Some(*v as u64),
            Self::SpatialU16(v, _) |
            Self::SpatialI16(v, _) if *v >= 0 => Some(*v as u64),
            Self::SpatialU32(v, _) |
            Self::SpatialI32(v, _) if *v >= 0 => Some(*v as u64),
            Self::SpatialU64(v, _) |
            Self::SpatialI64(v, _) if *v >= 0 => Some(*v as u64),
            Self::SpatialBool(v, _) => Some(if *v { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Get as float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::NeuralF32(v) => Some(*v as f64),
            Self::NeuralF64(v) => Some(*v),
            Self::SpatialF32(v, _) => Some(*v as f64),
            Self::SpatialF64(v, _) => Some(*v),
            _ => self.as_signed_int().map(|v| v as f64),
        }
    }

    /// Get as boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::NeuralBool(v) => Some(*v),
            Self::SpatialBool(v, _) => Some(*v),
            Self::NeuralI8(v) => Some(*v != 0),
            Self::NeuralI16(v) => Some(*v != 0),
            Self::NeuralI32(v) => Some(*v != 0),
            Self::NeuralI64(v) => Some(*v != 0),
            Self::NeuralU8(v) => Some(*v != 0),
            Self::NeuralU16(v) => Some(*v != 0),
            Self::NeuralU32(v) => Some(*v != 0),
            Self::NeuralU64(v) => Some(*v != 0),
            _ => None,
        }
    }

    /// Get as string (zero-copy if possible)
    pub fn as_text(&self) -> Option<Cow<'_, str>> {
        match self {
            Self::NeuralText(v) => Some(Cow::Borrowed(v)),
            Self::SpatialText(v, _) => Some(Cow::Borrowed(v)),
            _ => None,
        }
    }

    /// Get as string or convert to string representation
    pub fn to_text(&self) -> String {
        match self.as_text() {
            Some(s) => s.into_owned(),
            None => format!("{:?}", self),
        }
    }

    /// Get as slice of signed integers (zero-copy view)
    pub fn as_signed_int_slice(&self) -> Option<&[i64]> {
        match self {
            Self::ArrayI64(v) => Some(v),
            Self::SpatialArrayI64(v, _) => Some(v),
            _ => None,
        }
    }

    /// Get as slice of unsigned integers
    pub fn as_unsigned_int_slice(&self) -> Option<&[u64]> {
        match self {
            Self::ArrayU64(v) => Some(v),
            Self::SpatialArrayU64(v, _) => Some(v),
            _ => None,
        }
    }

    /// Get as slice of floats
    pub fn as_float_slice(&self) -> Option<&[f64]> {
        match self {
            Self::ArrayF64(v) => Some(v),
            Self::SpatialArrayF64(v, _) => Some(v),
            _ => None,
        }
    }

    /// Get as slice of booleans
    pub fn as_bool_slice(&self) -> Option<&[bool]> {
        match self {
            Self::ArrayBool(v) => Some(v),
            Self::SpatialArrayBool(v, _) => Some(v),
            _ => None,
        }
    }

    /// Get as slice of strings
    pub fn as_text_slice(&self) -> Option<&[String]> {
        match self {
            Self::ArrayText(v) => Some(v),
            Self::SpatialArrayText(v, _) => Some(v),
            _ => None,
        }
    }

    /// Convert to vector of signed integers (may allocate)
    pub fn to_signed_ints(&self) -> Vec<i64> {
        match self {
            Self::ArrayI8(v) => v.iter().map(|&x| x as i64).collect(),
            Self::ArrayI16(v) => v.iter().map(|&x| x as i64).collect(),
            Self::ArrayI32(v) => v.iter().map(|&x| x as i64).collect(),
            Self::ArrayI64(v) => v.clone(),
            Self::ArrayU8(v) => v.iter().map(|&x| x as i64).collect(),
            Self::ArrayU16(v) => v.iter().map(|&x| x as i64).collect(),
            Self::ArrayU32(v) => v.iter().map(|&x| x as i64).collect(),
            Self::ArrayU64(v) => v.iter().map(|&x| *x as i64).collect(),
            Self::SpatialArrayI8(v, _) => v.iter().map(|&x| x as i64).collect(),
            Self::SpatialArrayI16(v, _) => v.iter().map(|&x| x as i64).collect(),
            Self::SpatialArrayI32(v, _) => v.iter().map(|&x| x as i64).collect(),
            Self::SpatialArrayI64(v, _) => v.clone(),
            Self::SpatialArrayU8(v, _) => v.iter().map(|&x| x as i64).collect(),
            Self::SpatialArrayU16(v, _) => v.iter().map(|&x| x as i64).collect(),
            Self::SpatialArrayU32(v, _) => v.iter().map(|&x| x as i64).collect(),
            Self::SpatialArrayU64(v, _) => v.iter().map(|&x| *x as i64).collect(),
            _ => Vec::new(),
        }
    }

    /// Convert to vector of unsigned integers
    pub fn to_unsigned_ints(&self) -> Vec<u64> {
        match self {
            Self::ArrayU8(v) => v.iter().map(|&x| x as u64).collect(),
            Self::ArrayU16(v) => v.iter().map(|&x| x as u64).collect(),
            Self::ArrayU32(v) => v.iter().map(|&x| x as u64).collect(),
            Self::ArrayU64(v) => v.clone(),
            Self::ArrayI8(v) => v.iter().filter(|&&x| x >= 0).map(|&x| x as u64).collect(),
            Self::ArrayI16(v) => v.iter().filter(|&&x| x >= 0).map(|&x| x as u64).collect(),
            Self::ArrayI32(v) => v.iter().filter(|&&x| x >= 0).map(|&x| x as u64).collect(),
            Self::ArrayI64(v) => v.iter().filter(|&&x| *x >= 0).map(|&x| x as u64).collect(),
            Self::SpatialArrayU8(v, _) => v.iter().map(|&x| x as u64).collect(),
            Self::SpatialArrayU16(v, _) => v.iter().map(|&x| x as u64).collect(),
            Self::SpatialArrayU32(v, _) => v.iter().map(|&x| x as u64).collect(),
            Self::SpatialArrayU64(v, _) => v.clone(),
            Self::SpatialArrayI8(v, _) => v.iter().filter(|&&x| x >= 0).map(|&x| x as u64).collect(),
            Self::SpatialArrayI16(v, _) => v.iter().filter(|&&x| x >= 0).map(|&x| x as u64).collect(),
            Self::SpatialArrayI32(v, _) => v.iter().filter(|&&x| x >= 0).map(|&x| x as u64).collect(),
            Self::SpatialArrayI64(v, _) => v.iter().filter(|&&x| *x >= 0).map(|&x| x as u64).collect(),
            _ => Vec::new(),
        }
    }

    /// Convert to vector of floats
    pub fn to_floats(&self) -> Vec<f64> {
        match self {
            Self::ArrayF32(v) => v.iter().map(|&x| x as f64).collect(),
            Self::ArrayF64(v) => v.clone(),
            Self::SpatialArrayF32(v, _) => v.iter().map(|&x| x as f64).collect(),
            Self::SpatialArrayF64(v, _) => v.clone(),
            _ => self.to_signed_ints().iter().map(|&x| x as f64).collect(),
        }
    }

    /// Convert to vector of booleans
    pub fn to_bools(&self) -> Vec<bool> {
        match self {
            Self::ArrayBool(v) => v.clone(),
            Self::SpatialArrayBool(v, _) => v.clone(),
            Self::ArrayI8(v) => v.iter().map(|&x| x != 0).collect(),
            Self::ArrayI16(v) => v.iter().map(|&x| x != 0).collect(),
            Self::ArrayI32(v) => v.iter().map(|&x| x != 0).collect(),
            Self::ArrayI64(v) => v.iter().map(|&x| x != 0).collect(),
            Self::ArrayU8(v) => v.iter().map(|&x| x != 0).collect(),
            Self::ArrayU16(v) => v.iter().map(|&x| x != 0).collect(),
            Self::ArrayU32(v) => v.iter().map(|&x| x != 0).collect(),
            Self::ArrayU64(v) => v.iter().map(|&x| x != 0).collect(),
            _ => Vec::new(),
        }
    }

    /// Convert to vector of strings
    pub fn to_texts(&self) -> Vec<String> {
        match self {
            Self::ArrayText(v) => v.clone(),
            Self::SpatialArrayText(v, _) => v.clone(),
            _ => Vec::new(),
        }
    }

    /// Check if value is null/empty
    pub fn is_null(&self) -> bool {
        matches!(self, Self::NeuralNull)
    }

    /// Check if value is valid (not null and has data)
    pub fn is_valid(&self) -> bool {
        !self.is_null()
    }

    /// Get the type name as string
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::NeuralU8(_) => "u8",
            Self::NeuralI8(_) => "i8",
            Self::NeuralU16(_) => "u16",
            Self::NeuralI16(_) => "i16",
            Self::NeuralU32(_) => "u32",
            Self::NeuralI32(_) => "i32",
            Self::NeuralU64(_) => "u64",
            Self::NeuralI64(_) => "i64",
            Self::NeuralF32(_) => "f32",
            Self::NeuralF64(_) => "f64",
            Self::NeuralBool(_) => "bool",
            Self::NeuralText(_) => "text",
            Self::SpatialU8(_, _) => "u8:spatial",
            Self::SpatialI8(_, _) => "i8:spatial",
            Self::SpatialU16(_, _) => "u16:spatial",
            Self::SpatialI16(_, _) => "i16:spatial",
            Self::SpatialU32(_, _) => "u32:spatial",
            Self::SpatialI32(_, _) => "i32:spatial",
            Self::SpatialU64(_, _) => "u64:spatial",
            Self::SpatialI64(_, _) => "i64:spatial",
            Self::SpatialF32(_, _) => "f32:spatial",
            Self::SpatialF64(_, _) => "f64:spatial",
            Self::SpatialBool(_, _) => "bool:spatial",
            Self::SpatialText(_, _) => "text:spatial",
            Self::ArrayU8(_) => "[u8]",
            Self::ArrayI8(_) => "[i8]",
            Self::ArrayU16(_) => "[u16]",
            Self::ArrayI16(_) => "[i16]",
            Self::ArrayU32(_) => "[u32]",
            Self::ArrayI32(_) => "[i32]",
            Self::ArrayU64(_) => "[u64]",
            Self::ArrayI64(_) => "[i64]",
            Self::ArrayF32(_) => "[f32]",
            Self::ArrayF64(_) => "[f64]",
            Self::ArrayBool(_) => "[bool]",
            Self::ArrayText(_) => "[text]",
            Self::NestedMetadata(_) => "metadata",
            Self::NestedArray(_) => "[any]",
            Self::SharedValue(_) => "shared",
            Self::NeuralNull => "null",
            _ => "[spatial]",
        }
    }

    /// Create a zero-copy shared reference
    pub fn to_shared(self) -> Self {
        Self::SharedValue(Arc::new(self))
    }

    /// Get byte size estimate for memory planning
    pub fn estimated_byte_size(&self) -> usize {
        match self {
            Self::NeuralU8(_) | Self::NeuralI8(_) => 1,
            Self::NeuralU16(_) | Self::NeuralI16(_) => 2,
            Self::NeuralU32(_) | Self::NeuralI32(_) | Self::NeuralF32(_) => 4,
            Self::NeuralU64(_) | Self::NeuralI64(_) | Self::NeuralF64(_) => 8,
            Self::NeuralBool(_) => 1,
            Self::NeuralText(v) => v.len(),
            Self::ArrayU8(v) => v.len(),
            Self::ArrayI8(v) => v.len(),
            Self::ArrayU16(v) => v.len() * 2,
            Self::ArrayI16(v) => v.len() * 2,
            Self::ArrayU32(v) => v.len() * 4,
            Self::ArrayI32(v) => v.len() * 4,
            Self::ArrayF32(v) => v.len() * 4,
            Self::ArrayU64(v) => v.len() * 8,
            Self::ArrayI64(v) => v.len() * 8,
            Self::ArrayF64(v) => v.len() * 8,
            Self::ArrayBool(v) => v.len(),
            Self::ArrayText(v) => v.iter().map(|s| s.len()).sum(),
            Self::SharedValue(v) => v.estimated_byte_size(),
            Self::NestedMetadata(v) => v.value.estimated_byte_size() + v.key.len(),
            Self::NestedArray(v) => v.iter().map(|x| x.estimated_byte_size()).sum(),
            _ => std::mem::size_of::<Self>(), // Spatial variants (same base size)
        }
    }
}

impl fmt::Display for TitanMetadataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NeuralText(v) => write!(f, "{}", v),
            Self::SpatialText(v, s) => write!(f, "{} [{}x{}x{}]", v, s.width, s.height, s.depth),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Neural metadata entry with 3D spatial key and value
#[derive(Debug, Clone, PartialEq)]
pub struct TitanMetadataEntry {
    /// Key identifier
    pub key: String,
    /// Value with optional spatial context
    pub value: TitanMetadataValue,
    /// 3D spatial region this metadata applies to
    pub spatial_region: Option<SpatialTensorMetadata>,
    /// Timestamp for versioning
    pub timestamp: u64,
}

impl TitanMetadataEntry {
    /// Create new metadata entry
    pub fn new(key: String, value: TitanMetadataValue) -> Self {
        Self {
            spatial_region: value.spatial_metadata().cloned(),
            key,
            value,
            timestamp: 0,
        }
    }

    /// Create with explicit spatial region
    pub fn with_spatial_region(
        key: String,
        value: TitanMetadataValue,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Self {
        Self {
            key,
            value,
            spatial_region: Some(SpatialTensorMetadata::new(width, height, depth)),
            timestamp: 0,
        }
    }

    /// Check if entry has valid key and value
    pub fn is_valid(&self) -> bool {
        !self.key.is_empty() && self.value.is_valid()
    }

    /// Get spatial volume if available
    pub fn spatial_volume(&self) -> Option<u64> {
        self.spatial_region.as_ref().map(|s| s.volume())
    }
}

/// High-performance metadata registry with 3D spatial indexing
pub struct TitanMetadataRegistry {
    entries: HashMap<String, TitanMetadataEntry>,
    spatial_index: HashMap<(u32, u32, u32), Vec<String>>, // (width, height, depth) -> keys
}

impl TitanMetadataRegistry {
    /// Create new empty registry
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            spatial_index: HashMap::new(),
        }
    }

    /// Insert metadata entry
    pub fn insert(&mut self, entry: TitanMetadataEntry) {
        // Update spatial index if entry has spatial region
        if let Some(ref spatial) = entry.spatial_region {
            let key_list = self
                .spatial_index
                .entry((spatial.width, spatial.height, spatial.depth))
                .or_default();
            key_list.push(entry.key.clone());
        }

        self.entries.insert(entry.key.clone(), entry);
    }

    /// Get entry by key
    pub fn get(&self, key: &str) -> Option<&TitanMetadataEntry> {
        self.entries.get(key)
    }

    /// Get mutable entry by key
    pub fn get_mut(&mut self, key: &str) -> Option<&mut TitanMetadataEntry> {
        self.entries.get_mut(key)
    }

    /// Get value by key
    pub fn value(&self, key: &str) -> Option<&TitanMetadataValue> {
        self.entries.get(key).map(|e| &e.value)
    }

    /// Get signed integer value
    pub fn get_signed_int(&self, key: &str) -> Option<i64> {
        self.value(key).and_then(|v| v.as_signed_int())
    }

    /// Get unsigned integer value
    pub fn get_unsigned_int(&self, key: &str) -> Option<u64> {
        self.value(key).and_then(|v| v.as_unsigned_int())
    }

    /// Get float value
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.value(key).and_then(|v| v.as_float())
    }

    /// Get boolean value
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.value(key).and_then(|v| v.as_bool())
    }

    /// Get text value
    pub fn get_text(&self, key: &str) -> Option<Cow<'_, str>> {
        self.value(key).and_then(|v| v.as_text())
    }

    /// Get signed integer array
    pub fn get_signed_ints(&self, key: &str) -> Option<Vec<i64>> {
        self.value(key).map(|v| v.to_signed_ints())
    }

    /// Get unsigned integer array
    pub fn get_unsigned_ints(&self, key: &str) -> Option<Vec<u64>> {
        self.value(key).map(|v| v.to_unsigned_ints())
    }

    /// Get float array
    pub fn get_floats(&self, key: &str) -> Option<Vec<f64>> {
        self.value(key).map(|v| v.to_floats())
    }

    /// Get boolean array
    pub fn get_bools(&self, key: &str) -> Option<Vec<bool>> {
        self.value(key).map(|v| v.to_bools())
    }

    /// Get text array
    pub fn get_texts(&self, key: &str) -> Option<Vec<String>> {
        self.value(key).map(|v| v.to_texts())
    }

    /// Find entries by spatial region
    pub fn find_by_spatial(&self, width: u32, height: u32, depth: u32) -> Vec<&TitanMetadataEntry> {
        self.spatial_index
            .get(&(width, height, depth))
            .map(|keys| {
                keys.iter()
                    .filter_map(|k| self.entries.get(k))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find entries matching spatial requirements
    pub fn find_matching_spatial(
        &self,
        min_width: u32,
        min_height: u32,
        min_depth: u32,
    ) -> Vec<&TitanMetadataEntry> {
        self.entries
            .values()
            .filter(|e| {
                e.spatial_region
                    .as_ref()
                    .map(|s| {
                        s.width >= min_width && s.height >= min_height && s.depth >= min_depth
                    })
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Count of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate all entries
    pub fn iter(&self) -> impl Iterator<Item = &TitanMetadataEntry> {
        self.entries.values()
    }

    /// Get all keys
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    /// Remove entry by key
    pub fn remove(&mut self, key: &str) -> Option<TitanMetadataEntry> {
        let entry = self.entries.remove(key);
        
        // Update spatial index
        if let Some(ref spatial) = entry.as_ref().and_then(|e| e.spatial_region.as_ref()) {
            if let Some(keys) = self.spatial_index.get_mut(&(spatial.width, spatial.height, spatial.depth)) {
                keys.retain(|k| k != key);
            }
        }
        
        entry
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.spatial_index.clear();
    }

    /// Get total memory estimate for all metadata
    pub fn total_memory_estimate(&self) -> usize {
        self.entries
            .values()
            .map(|e| e.key.len() + e.value.estimated_byte_size())
            .sum()
    }
}

impl Default for TitanMetadataRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating metadata values with fluent API
pub struct TitanMetadataBuilder {
    value: Option<TitanMetadataValue>,
    spatial: Option<SpatialTensorMetadata>,
}

impl TitanMetadataBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            value: None,
            spatial: None,
        }
    }

    /// Set unsigned 8-bit value
    pub fn u8(mut self, v: u8) -> Self {
        self.value = Some(TitanMetadataValue::NeuralU8(v));
        self
    }

    /// Set signed 8-bit value
    pub fn i8(mut self, v: i8) -> Self {
        self.value = Some(TitanMetadataValue::NeuralI8(v));
        self
    }

    /// Set unsigned 16-bit value
    pub fn u16(mut self, v: u16) -> Self {
        self.value = Some(TitanMetadataValue::NeuralU16(v));
        self
    }

    /// Set signed 16-bit value
    pub fn i16(mut self, v: i16) -> Self {
        self.value = Some(TitanMetadataValue::NeuralI16(v));
        self
    }

    /// Set unsigned 32-bit value
    pub fn u32(mut self, v: u32) -> Self {
        self.value = Some(TitanMetadataValue::NeuralU32(v));
        self
    }

    /// Set signed 32-bit value
    pub fn i32(mut self, v: i32) -> Self {
        self.value = Some(TitanMetadataValue::NeuralI32(v));
        self
    }

    /// Set unsigned 64-bit value
    pub fn u64(mut self, v: u64) -> Self {
        self.value = Some(TitanMetadataValue::NeuralU64(v));
        self
    }

    /// Set signed 64-bit value
    pub fn i64(mut self, v: i64) -> Self {
        self.value = Some(TitanMetadataValue::NeuralI64(v));
        self
    }

    /// Set 32-bit float value
    pub fn f32(mut self, v: f32) -> Self {
        self.value = Some(TitanMetadataValue::NeuralF32(v));
        self
    }

    /// Set 64-bit float value
    pub fn f64(mut self, v: f64) -> Self {
        self.value = Some(TitanMetadataValue::NeuralF64(v));
        self
    }

    /// Set boolean value
    pub fn bool(mut self, v: bool) -> Self {
        self.value = Some(TitanMetadataValue::NeuralBool(v));
        self
    }

    /// Set text value
    pub fn text(mut self, v: impl Into<String>) -> Self {
        self.value = Some(TitanMetadataValue::NeuralText(v.into()));
        self
    }

    /// Set array value
    pub fn array<T: Into<TitanMetadataValue>>(mut self, items: Vec<T>) -> Self {
        // This would need more complex implementation for generic arrays
        // For now, placeholder
        self
    }

    /// Add 3D spatial context
    pub fn spatial(mut self, width: u32, height: u32, depth: u32) -> Self {
        self.spatial = Some(SpatialTensorMetadata::new(width, height, depth));
        self
    }

    /// Build final value
    pub fn build(self) -> Option<TitanMetadataValue> {
        match (self.value, self.spatial) {
            (Some(v), Some(s)) => {
                // Apply spatial context to value
                Some(v.with_spatial_context(s.width, s.height, s.depth))
            }
            (Some(v), None) => Some(v),
            _ => None,
        }
    }

    /// Build entry with key
    pub fn build_entry(self, key: impl Into<String>) -> Option<TitanMetadataEntry> {
        self.build().map(|v| TitanMetadataEntry::new(key.into(), v))
    }
}

impl Default for TitanMetadataBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Type aliases for convenience
pub type NeuralValue = TitanMetadataValue;
pub type NeuralEntry = TitanMetadataEntry;
pub type NeuralRegistry = TitanMetadataRegistry;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_conversions() {
        let val = NeuralValue::NeuralI64(42);
        assert_eq!(val.as_signed_int(), Some(42));
        assert_eq!(val.as_unsigned_int(), Some(42));
        assert_eq!(val.as_float(), Some(42.0));

        let val = NeuralValue::NeuralF64(3.14);
        assert_eq!(val.as_float(), Some(3.14));
        assert_eq!(val.as_signed_int(), Some(3));

        let val = NeuralValue::NeuralBool(true);
        assert_eq!(val.as_bool(), Some(true));
        assert_eq!(val.as_signed_int(), Some(1));
    }

    #[test]
    fn test_spatial_values() {
        let val = NeuralValue::NeuralF32(1.5)
            .with_spatial_context(256, 256, 128);
        
        assert!(val.has_spatial_context());
        assert!(matches!(val, NeuralValue::SpatialF32(_, _)));
        
        let spatial = val.spatial_metadata().unwrap();
        assert_eq!(spatial.width, 256);
        assert_eq!(spatial.height, 256);
        assert_eq!(spatial.depth, 128);
    }

    #[test]
    fn test_array_conversions() {
        let val = NeuralValue::ArrayI32(vec![1, 2, 3, 4, 5]);
        let ints = val.to_signed_ints();
        assert_eq!(ints, vec![1, 2, 3, 4, 5]);

        let val = NeuralValue::ArrayF64(vec![1.1, 2.2, 3.3]);
        let floats = val.to_floats();
        assert_eq!(floats, vec![1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_registry_operations() {
        let mut registry = NeuralRegistry::new();
        
        let entry = NeuralEntry::new(
            "model.layers".to_string(),
            NeuralValue::NeuralU32(32),
        );
        registry.insert(entry);
        
        assert_eq!(registry.get_unsigned_int("model.layers"), Some(32));
        assert_eq!(registry.len(), 1);
        
        let entry2 = NeuralEntry::with_spatial_region(
            "tensor.shape".to_string(),
            NeuralValue::ArrayU64(vec![256, 256, 128]),
            256, 256, 128,
        );
        registry.insert(entry2);
        
        let spatial_entries = registry.find_by_spatial(256, 256, 128);
        assert_eq!(spatial_entries.len(), 1);
    }

    #[test]
    fn test_builder_pattern() {
        let value = TitanMetadataBuilder::new()
            .f32(1.5)
            .spatial(512, 512, 256)
            .build();
        
        assert!(value.is_some());
        let val = value.unwrap();
        assert!(matches!(val, NeuralValue::SpatialF32(_, _)));
    }

    #[test]
    fn test_entry_validity() {
        let valid = NeuralEntry::new("key".to_string(), NeuralValue::NeuralU32(42));
        assert!(valid.is_valid());
        
        let invalid_key = NeuralEntry::new("".to_string(), NeuralValue::NeuralU32(42));
        assert!(!invalid_key.is_valid());
        
        let invalid_value = NeuralEntry::new("key".to_string(), NeuralValue::NeuralNull);
        assert!(!invalid_value.is_valid());
    }

    #[test]
    fn test_memory_estimates() {
        let val = NeuralValue::ArrayF64(vec![1.0; 1000]);
        assert_eq!(val.estimated_byte_size(), 8000); // 1000 * 8 bytes
        
        let text = NeuralValue::NeuralText("hello world".to_string());
        assert_eq!(text.estimated_byte_size(), 11);
    }
}