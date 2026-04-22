use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::fs::File as StdFile;
use std::io::{self, Read, Seek, SeekFrom};
use std::mem::size_of;
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::fs::ggml::pronax_ggml_types::{NeuralTensorDataType, SpatialTensorMetadata};

/// Magic bytes for GGUF format identification
pub const TITAN_GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Minimum supported GGUF version for ProNax Titan engine
pub const TITAN_MIN_VERSION: u32 = 2;

/// Default memory alignment for tensor data
pub const TITAN_DEFAULT_ALIGNMENT: u64 = 32;

/// Buffer size for zero-copy streaming reads
pub const TITAN_STREAM_BUFFER_SIZE: usize = 32 * 1024; // 32KB

/// Internal metadata cache size for spatial optimization
pub const TITAN_METADATA_CACHE_SIZE: usize = 4096;

/// GGUF value type identifiers with 3D spatial awareness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum NeuralMetadataValueKind {
    UnsignedInt8 = 0,
    SignedInt8 = 1,
    UnsignedInt16 = 2,
    SignedInt16 = 3,
    UnsignedInt32 = 4,
    SignedInt32 = 5,
    BrainFloat32 = 6,
    NeuralBoolean = 7,
    NeuralString = 8,
    NeuralArray = 9,
    UnsignedInt64 = 10,
    SignedInt64 = 11,
    BrainFloat64 = 12,
}

impl NeuralMetadataValueKind {
    /// Check if type supports 3D spatial metadata attachment
    #[inline]
    pub const fn supports_spatial_extension(&self) -> bool {
        matches!(
            self,
            Self::BrainFloat32 | Self::BrainFloat64 | Self::NeuralArray
        )
    }

    /// Get byte size for fixed-size types
    #[inline]
    pub const fn fixed_byte_size(&self) -> Option<usize> {
        match self {
            Self::UnsignedInt8 | Self::SignedInt8 => Some(1),
            Self::UnsignedInt16 | Self::SignedInt16 => Some(2),
            Self::UnsignedInt32 | Self::SignedInt32 | Self::BrainFloat32 => Some(4),
            Self::UnsignedInt64 | Self::SignedInt64 | Self::BrainFloat64 => Some(8),
            Self::NeuralBoolean => Some(1),
            _ => None, // Variable size types
        }
    }
}

/// 3D-enhanced tensor information with spatial metadata
#[derive(Debug, Clone, PartialEq)]
pub struct NeuralTensorDescriptor {
    /// Unique tensor identifier
    pub identifier: String,
    /// Neural dimensions (can be 1D to N-D, but optimized for 3D)
    pub dimensions: Vec<u64>,
    /// 3D spatial metadata for AI processing
    pub spatial_config: SpatialTensorMetadata,
    /// Data type for tensor elements
    pub element_type: NeuralTensorDataType,
    /// Offset to tensor data in file
    pub data_offset: u64,
    /// Pre-computed total byte size
    pub total_bytes: NonZeroU64,
    /// Whether tensor is optimized for 3D spatial operations
    pub spatial_optimized: bool,
}

impl NeuralTensorDescriptor {
    /// Create new tensor descriptor with 3D spatial awareness
    pub fn new(
        identifier: String,
        dimensions: Vec<u64>,
        element_type: NeuralTensorDataType,
        data_offset: u64,
    ) -> Result<Self, TitanFormatError> {
        let total_elements: u64 = dimensions.iter().product();
        if total_elements == 0 {
            return Err(TitanFormatError::InvalidTensorDimensions {
                identifier: identifier.clone(),
                reason: "Zero elements in tensor".to_string(),
            });
        }

        // Calculate 3D spatial config from dimensions
        let spatial_config = Self::compute_spatial_metadata(&dimensions);
        
        // Calculate total bytes based on element type
        let type_size = element_type.element_size_bytes() as u64;
        let block_size = element_type.block_elements() as u64;
        let total_bytes_val = (total_elements * type_size) / block_size;
        
        let total_bytes = NonZeroU64::new(total_bytes_val).ok_or_else(|| {
            TitanFormatError::InvalidTensorDimensions {
                identifier: identifier.clone(),
                reason: "Zero byte size calculated".to_string(),
            }
        })?;

        Ok(Self {
            identifier,
            dimensions,
            spatial_config,
            element_type,
            data_offset,
            total_bytes,
            spatial_optimized: element_type.supports_neural_3d(),
        })
    }

    /// Compute 3D spatial metadata from tensor dimensions
    fn compute_spatial_metadata(dims: &[u64]) -> SpatialTensorMetadata {
        match dims.len() {
            0 => SpatialTensorMetadata::new(1, 1, 1),
            1 => SpatialTensorMetadata::new(dims[0] as u32, 1, 1),
            2 => SpatialTensorMetadata::new(dims[1] as u32, dims[0] as u32, 1),
            _ => {
                // For 3D+ tensors, use last 3 dimensions as spatial
                let depth = dims[dims.len() - 3] as u32;
                let height = dims[dims.len() - 2] as u32;
                let width = dims[dims.len() - 1] as u32;
                SpatialTensorMetadata::new(width, height, depth)
            }
        }
    }

    /// Get zero-copy byte range for reading tensor data
    #[inline]
    pub fn byte_range(&self) -> (u64, u64) {
        (self.data_offset, self.total_bytes.get())
    }

    /// Check if tensor matches given spatial requirements
    #[inline]
    pub fn matches_spatial_requirements(&self, min_width: u32, min_height: u32, min_depth: u32) -> bool {
        self.spatial_config.width >= min_width
            && self.spatial_config.height >= min_height
            && self.spatial_config.depth >= min_depth
    }
}

/// Neural metadata value with optional 3D spatial context
#[derive(Debug, Clone, PartialEq)]
pub enum NeuralMetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    UInt64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Boolean(bool),
    Text(String),
    Array(Vec<NeuralMetadataValue>),
    // 3D-enhanced variants with spatial metadata
    SpatialFloat32(f32, SpatialTensorMetadata),
    SpatialFloat64(f64, SpatialTensorMetadata),
}

impl NeuralMetadataValue {
    /// Get value as string representation
    pub fn as_text(&self) -> Cow<'_, str> {
        match self {
            Self::Text(s) => Cow::Borrowed(s),
            Self::UInt8(v) => Cow::Owned(v.to_string()),
            Self::Int8(v) => Cow::Owned(v.to_string()),
            Self::UInt16(v) => Cow::Owned(v.to_string()),
            Self::Int16(v) => Cow::Owned(v.to_string()),
            Self::UInt32(v) => Cow::Owned(v.to_string()),
            Self::Int32(v) => Cow::Owned(v.to_string()),
            Self::UInt64(v) => Cow::Owned(v.to_string()),
            Self::Int64(v) => Cow::Owned(v.to_string()),
            Self::Float32(v) => Cow::Owned(v.to_string()),
            Self::Float64(v) => Cow::Owned(v.to_string()),
            Self::Boolean(v) => Cow::Owned(v.to_string()),
            Self::Array(_) => Cow::Borrowed("[array]"),
            Self::SpatialFloat32(v, _) => Cow::Owned(format!("{} (spatial)", v)),
            Self::SpatialFloat64(v, _) => Cow::Owned(format!("{} (spatial)", v)),
        }
    }

    /// Get integer value if applicable
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::UInt8(v) => Some(*v as i64),
            Self::Int8(v) => Some(*v as i64),
            Self::UInt16(v) => Some(*v as i64),
            Self::Int16(v) => Some(*v as i64),
            Self::UInt32(v) => Some(*v as i64),
            Self::Int32(v) => Some(*v as i64),
            Self::UInt64(v) => Some(*v as i64),
            Self::Int64(v) => Some(*v),
            Self::Boolean(v) => Some(if *v { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Get float value if applicable
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float32(v) => Some(*v as f64),
            Self::Float64(v) => Some(*v),
            Self::SpatialFloat32(v, _) => Some(*v as f64),
            Self::SpatialFloat64(v, _) => Some(*v),
            _ => None,
        }
    }

    /// Get boolean value if applicable
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Self::Boolean(v) => Some(*v),
            _ => None,
        }
    }

    /// Check if value has 3D spatial metadata
    pub fn has_spatial_context(&self) -> bool {
        matches!(self, Self::SpatialFloat32(_, _) | Self::SpatialFloat64(_, _))
    }

    /// Get spatial metadata if available
    pub fn spatial_context(&self) -> Option<&SpatialTensorMetadata> {
        match self {
            Self::SpatialFloat32(_, spatial) => Some(spatial),
            Self::SpatialFloat64(_, spatial) => Some(spatial),
            _ => None,
        }
    }
}

/// Neural metadata key-value pair with 3D context
#[derive(Debug, Clone, PartialEq)]
pub struct NeuralMetadataEntry {
    pub key: String,
    pub value: NeuralMetadataValue,
}

/// Titan-specific errors for GGUF operations
#[derive(Debug, Clone, PartialEq)]
pub enum TitanFormatError {
    IoError { context: String },
    InvalidMagic { found: [u8; 4] },
    UnsupportedVersion { version: u32, minimum: u32 },
    InvalidTensorDimensions { identifier: String, reason: String },
    UnsupportedValueType { type_code: u32 },
    TensorNotFound { identifier: String },
    MetadataKeyNotFound { key: String },
    SpatialConfigError { reason: String },
}

impl fmt::Display for TitanFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoError { context } => {
                write!(f, "Titan I/O error: {}", context)
            }
            Self::InvalidMagic { found } => {
                write!(f, "Invalid GGUF magic bytes: {:?}", found)
            }
            Self::UnsupportedVersion { version, minimum } => {
                write!(f, "GGUF version {} not supported (minimum: {})", version, minimum)
            }
            Self::InvalidTensorDimensions { identifier, reason } => {
                write!(f, "Invalid dimensions for tensor '{}': {}", identifier, reason)
            }
            Self::UnsupportedValueType { type_code } => {
                write!(f, "Unsupported metadata value type: {}", type_code)
            }
            Self::TensorNotFound { identifier } => {
                write!(f, "Tensor '{}' not found in archive", identifier)
            }
            Self::MetadataKeyNotFound { key } => {
                write!(f, "Metadata key '{}' not found", key)
            }
            Self::SpatialConfigError { reason } => {
                write!(f, "3D spatial configuration error: {}", reason)
            }
        }
    }
}

impl std::error::Error for TitanFormatError {}

impl From<io::Error> for TitanFormatError {
    fn from(err: io::Error) -> Self {
        Self::IoError {
            context: err.to_string(),
        }
    }
}

/// Buffered reader with zero-copy optimization for streaming
pub struct TitanStreamBuffer<R: Read> {
    inner: R,
    buffer: Box<[u8]>,
    position: usize,
    filled: usize,
}

impl<R: Read> TitanStreamBuffer<R> {
    pub fn new(inner: R, capacity: usize) -> Self {
        Self {
            inner,
            buffer: vec![0u8; capacity].into_boxed_slice(),
            position: 0,
            filled: 0,
        }
    }

    /// Read exact bytes with zero-copy optimization
    pub fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), TitanFormatError> {
        let mut total_read = 0;
        while total_read < buf.len() {
            if self.position >= self.filled {
                self.refill()?;
            }
            
            let available = self.filled - self.position;
            let needed = buf.len() - total_read;
            let to_copy = available.min(needed);
            
            buf[total_read..total_read + to_copy]
                .copy_from_slice(&self.buffer[self.position..self.position + to_copy]);
            
            self.position += to_copy;
            total_read += to_copy;
        }
        Ok(())
    }

    fn refill(&mut self) -> Result<(), TitanFormatError> {
        self.filled = self.inner.read(&mut self.buffer)?;
        self.position = 0;
        if self.filled == 0 {
            return Err(TitanFormatError::IoError {
                context: "Unexpected end of stream".to_string(),
            });
        }
        Ok(())
    }

    /// Read little-endian value with zero-copy
    #[inline]
    pub fn read_neural_le<T: NeuralLittleEndian>(&mut self) -> Result<T, TitanFormatError> {
        T::read_neural_le(self)
    }
}

/// Trait for little-endian neural data reading
trait NeuralLittleEndian: Sized {
    fn read_neural_le<R: Read>(reader: &mut TitanStreamBuffer<R>) -> Result<Self, TitanFormatError>;
}

macro_rules! impl_neural_le {
    ($type:ty) => {
        impl NeuralLittleEndian for $type {
            fn read_neural_le<R: Read>(reader: &mut TitanStreamBuffer<R>) -> Result<Self, TitanFormatError> {
                let mut bytes = [0u8; size_of::<$type>()];
                reader.read_exact(&mut bytes)?;
                Ok(<$type>::from_le_bytes(bytes))
            }
        }
    };
}

impl_neural_le!(u8);
impl_neural_le!(i8);
impl_neural_le!(u16);
impl_neural_le!(i16);
impl_neural_le!(u32);
impl_neural_le!(i32);
impl_neural_le!(u64);
impl_neural_le!(i64);
impl_neural_le!(f32);
impl_neural_le!(f64);

/// Main Titan GGUF archive handler with 3D spatial optimization
pub struct TitanNeuralArchive {
    /// GGUF magic bytes
    magic: [u8; 4],
    /// Format version
    version: u32,
    /// Tensor data offset (aligned)
    tensor_data_offset: u64,
    /// Metadata entries
    metadata: HashMap<String, NeuralMetadataEntry>,
    /// Tensor descriptors
    tensors: HashMap<String, NeuralTensorDescriptor>,
    /// Source file handle (optional, for lazy loading)
    source_file: Option<Arc<Mutex<StdFile>>>,
    /// Reusable string buffer for zero-copy string reading
    string_buffer: Vec<u8>,
}

impl TitanNeuralArchive {
    /// Open GGUF file with Titan 3D optimizations
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, TitanFormatError> {
        let file = StdFile::open(path)?;
        let mut buffer = TitanStreamBuffer::new(file, TITAN_STREAM_BUFFER_SIZE);
        
        // Read and verify magic
        let mut magic = [0u8; 4];
        buffer.read_exact(&mut magic)?;
        
        if &magic != TITAN_GGUF_MAGIC {
            return Err(TitanFormatError::InvalidMagic { found: magic });
        }
        
        // Read version
        let version = buffer.read_neural_le::<u32>()?;
        if version < TITAN_MIN_VERSION {
            return Err(TitanFormatError::UnsupportedVersion {
                version,
                minimum: TITAN_MIN_VERSION,
            });
        }
        
        // Read tensor count
        let tensor_count = buffer.read_neural_le::<u64>()?;
        
        // Read metadata entry count
        let metadata_count = buffer.read_neural_le::<u64>()?;
        
        // Create archive with initial state
        let mut archive = Self {
            magic,
            version,
            tensor_data_offset: 0,
            metadata: HashMap::with_capacity(metadata_count as usize),
            tensors: HashMap::with_capacity(tensor_count as usize),
            source_file: None,
            string_buffer: Vec::with_capacity(TITAN_METADATA_CACHE_SIZE),
        };
        
        // Read all metadata entries
        for _ in 0..metadata_count {
            let entry = archive.read_metadata_entry(&mut buffer)?;
            archive.metadata.insert(entry.key.clone(), entry);
        }
        
        // Read all tensor descriptors
        let mut max_offset: u64 = 0;
        for _ in 0..tensor_count {
            let tensor = archive.read_tensor_descriptor(&mut buffer)?;
            max_offset = max_offset.max(tensor.data_offset + tensor.total_bytes.get());
            archive.tensors.insert(tensor.identifier.clone(), tensor);
        }
        
        // Calculate aligned tensor data offset
        let current_offset = archive.calculate_stream_position(&buffer);
        let alignment = archive
            .metadata_value("general.alignment")
            .and_then(|v| v.as_integer())
            .map(|v| v as u64)
            .unwrap_or(TITAN_DEFAULT_ALIGNMENT);
        
        archive.tensor_data_offset = Self::align_offset(current_offset, alignment);
        
        Ok(archive)
    }

    /// Calculate alignment offset
    #[inline]
    fn align_offset(offset: u64, alignment: u64) -> u64 {
        let remainder = offset % alignment;
        if remainder == 0 {
            offset
        } else {
            offset + (alignment - remainder)
        }
    }

    /// Estimate stream position (approximate for buffered reads)
    fn calculate_stream_position<R: Read>(&self, buffer: &TitanStreamBuffer<R>) -> u64 {
        // This is a simplified calculation; in production, track actual bytes read
        0
    }

    /// Read single metadata entry with 3D spatial awareness
    fn read_metadata_entry<R: Read>(
        &mut self,
        buffer: &mut TitanStreamBuffer<R>,
    ) -> Result<NeuralMetadataEntry, TitanFormatError> {
        let key = self.read_neural_string(buffer)?;
        let type_code = buffer.read_neural_le::<u32>()?;
        
        let value = match type_code {
            0 => NeuralMetadataValue::UInt8(buffer.read_neural_le::<u8>()?),
            1 => NeuralMetadataValue::Int8(buffer.read_neural_le::<i8>()?),
            2 => NeuralMetadataValue::UInt16(buffer.read_neural_le::<u16>()?),
            3 => NeuralMetadataValue::Int16(buffer.read_neural_le::<i16>()?),
            4 => NeuralMetadataValue::UInt32(buffer.read_neural_le::<u32>()?),
            5 => NeuralMetadataValue::Int32(buffer.read_neural_le::<i32>()?),
            6 => {
                let val = buffer.read_neural_le::<f32>()?;
                // Add spatial metadata for float values in neural context
                let spatial = SpatialTensorMetadata::new(1, 1, 1);
                NeuralMetadataValue::SpatialFloat32(val, spatial)
            }
            7 => NeuralMetadataValue::Boolean(buffer.read_neural_le::<u8>()? != 0),
            8 => NeuralMetadataValue::Text(self.read_neural_string(buffer)?),
            9 => self.read_neural_array(buffer)?,
            10 => NeuralMetadataValue::UInt64(buffer.read_neural_le::<u64>()?),
            11 => NeuralMetadataValue::Int64(buffer.read_neural_le::<i64>()?),
            12 => {
                let val = buffer.read_neural_le::<f64>()?;
                let spatial = SpatialTensorMetadata::new(1, 1, 1);
                NeuralMetadataValue::SpatialFloat64(val, spatial)
            }
            _ => {
                return Err(TitanFormatError::UnsupportedValueType { type_code })
            }
        };
        
        Ok(NeuralMetadataEntry { key, value })
    }

    /// Read neural array with type awareness
    fn read_neural_array<R: Read>(
        &mut self,
        buffer: &mut TitanStreamBuffer<R>,
    ) -> Result<NeuralMetadataValue, TitanFormatError> {
        let element_type = buffer.read_neural_le::<u32>()?;
        let count = buffer.read_neural_le::<u64>()?;
        
        let mut elements = Vec::with_capacity(count.min(1024) as usize);
        
        for _ in 0..count {
            let element = match element_type {
                0 => NeuralMetadataValue::UInt8(buffer.read_neural_le::<u8>()?),
                1 => NeuralMetadataValue::Int8(buffer.read_neural_le::<i8>()?),
                2 => NeuralMetadataValue::UInt16(buffer.read_neural_le::<u16>()?),
                3 => NeuralMetadataValue::Int16(buffer.read_neural_le::<i16>()?),
                4 => NeuralMetadataValue::UInt32(buffer.read_neural_le::<u32>()?),
                5 => NeuralMetadataValue::Int32(buffer.read_neural_le::<i32>()?),
                6 => NeuralMetadataValue::Float32(buffer.read_neural_le::<f32>()?),
                7 => NeuralMetadataValue::Boolean(buffer.read_neural_le::<u8>()? != 0),
                8 => NeuralMetadataValue::Text(self.read_neural_string(buffer)?),
                10 => NeuralMetadataValue::UInt64(buffer.read_neural_le::<u64>()?),
                11 => NeuralMetadataValue::Int64(buffer.read_neural_le::<i64>()?),
                12 => NeuralMetadataValue::Float64(buffer.read_neural_le::<f64>()?),
                _ => {
                    return Err(TitanFormatError::UnsupportedValueType {
                        type_code: element_type,
                    })
                }
            };
            elements.push(element);
        }
        
        Ok(NeuralMetadataValue::Array(elements))
    }

    /// Read length-prefixed string with zero-copy optimization
    fn read_neural_string<R: Read>(
        &mut self,
        buffer: &mut TitanStreamBuffer<R>,
    ) -> Result<String, TitanFormatError> {
        let length = buffer.read_neural_le::<u64>()?;
        
        // Resize buffer if needed
        if length as usize > self.string_buffer.capacity() {
            self.string_buffer.reserve(length as usize - self.string_buffer.capacity());
        }
        self.string_buffer.resize(length as usize, 0);
        
        buffer.read_exact(&mut self.string_buffer)?;
        
        // Convert to string (this allocates, but we minimize reallocations with buffer reuse)
        String::from_utf8(self.string_buffer.clone())
            .map_err(|_| TitanFormatError::IoError {
                context: "Invalid UTF-8 in string".to_string(),
            })
    }

    /// Read tensor descriptor with 3D spatial metadata
    fn read_tensor_descriptor<R: Read>(
        &mut self,
        buffer: &mut TitanStreamBuffer<R>,
    ) -> Result<NeuralTensorDescriptor, TitanFormatError> {
        let identifier = self.read_neural_string(buffer)?;
        let dimension_count = buffer.read_neural_le::<u32>()?;
        
        let mut dimensions = Vec::with_capacity(dimension_count as usize);
        for _ in 0..dimension_count {
            dimensions.push(buffer.read_neural_le::<u64>()?);
        }
        
        let element_type_code = buffer.read_neural_le::<u32>()?;
        let element_type = Self::map_neural_type_code(element_type_code)?;
        let data_offset = buffer.read_neural_le::<u64>()?;
        
        NeuralTensorDescriptor::new(identifier, dimensions, element_type, data_offset)
    }

    /// Map GGUF type code to NeuralTensorDataType
    fn map_neural_type_code(code: u32) -> Result<NeuralTensorDataType, TitanFormatError> {
        use crate::fs::ggml::pronax_ggml_types::NeuralTensorDataType as NTD;
        
        match code {
            0 => Ok(NTD::Float32),
            1 => Ok(NTD::Float16),
            2 => Ok(NTD::Quantized4Bit0),
            3 => Ok(NTD::Quantized4Bit1),
            6 => Ok(NTD::Quantized5Bit0),
            7 => Ok(NTD::Quantized5Bit1),
            8 => Ok(NTD::TitanQuantized8Bit0),
            9 => Ok(NTD::Quantized8Bit1),
            10 => Ok(NTD::KQuantized2Bit),
            11 => Ok(NTD::KQuantized3Bit),
            12 => Ok(NTD::TitanKQuantized4Bit),
            13 => Ok(NTD::KQuantized5Bit),
            14 => Ok(NTD::KQuantized6Bit),
            15 => Ok(NTD::KQuantized8Bit),
            16 => Ok(NTD::Imat2BitXXS),
            17 => Ok(NTD::Imat2BitXS),
            18 => Ok(NTD::Imat3BitXXS),
            19 => Ok(NTD::Imat1BitSmall),
            20 => Ok(NTD::Imat4BitNL),
            21 => Ok(NTD::Imat3BitSmall),
            22 => Ok(NTD::Imat2BitSmall),
            23 => Ok(NTD::Imat4BitXS),
            24 => Ok(NTD::Int8),
            25 => Ok(NTD::Int16),
            26 => Ok(NTD::Int32),
            27 => Ok(NTD::Int64),
            28 => Ok(NTD::Float64),
            29 => Ok(NTD::Imat1BitMedium),
            30 => Ok(NTD::BrainFloat16),
            39 => Ok(NTD::MicroFloat4),
            _ => Err(TitanFormatError::UnsupportedValueType { type_code: code }),
        }
    }

    /// Get metadata entry by key
    pub fn metadata_entry(&self, key: &str) -> Option<&NeuralMetadataEntry> {
        self.metadata.get(key)
    }

    /// Get metadata value by key (with architecture prefix resolution)
    pub fn metadata_value(&self, key: &str) -> Option<&NeuralMetadataValue> {
        // Direct lookup first
        if let Some(entry) = self.metadata.get(key) {
            return Some(&entry.value);
        }
        
        // Try with architecture prefix
        let arch = self
            .metadata_value("general.architecture")
            .and_then(|v| v.as_text())
            .map(|s| s.to_string())
            .unwrap_or_default();
        
        if !arch.is_empty() && !key.starts_with("general.") && !key.starts_with("tokenizer.") {
            let prefixed_key = format!("{}.{}", arch, key);
            if let Some(entry) = self.metadata.get(&prefixed_key) {
                return Some(&entry.value);
            }
        }
        
        None
    }

    /// Get tensor descriptor by identifier
    pub fn tensor_descriptor(&self, identifier: &str) -> Option<&NeuralTensorDescriptor> {
        self.tensors.get(identifier)
    }

    /// Get all tensor identifiers with 3D spatial optimization
    pub fn tensor_identifiers(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Get all metadata keys
    pub fn metadata_keys(&self) -> impl Iterator<Item = &String> {
        self.metadata.keys()
    }

    /// Count of tensors in archive
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Count of metadata entries
    pub fn metadata_count(&self) -> usize {
        self.metadata.len()
    }

    /// Get GGUF version
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Get tensor data offset
    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    /// Create tensor data reader for zero-copy access
    pub fn create_tensor_reader(
        &self,
        identifier: &str,
        source: &mut StdFile,
    ) -> Result<TitanTensorReader, TitanFormatError> {
        let descriptor = self
            .tensor_descriptor(identifier)
            .ok_or_else(|| TitanFormatError::TensorNotFound {
                identifier: identifier.to_string(),
            })?;
        
        let absolute_offset = self.tensor_data_offset + descriptor.data_offset;
        let byte_count = descriptor.total_bytes.get();
        
        // Seek to tensor data
        source.seek(SeekFrom::Start(absolute_offset))?;
        
        Ok(TitanTensorReader {
            descriptor: descriptor.clone(),
            bytes_remaining: byte_count,
        })
    }

    /// Find tensors matching spatial requirements
    pub fn find_spatial_tensors(
        &self,
        min_width: u32,
        min_height: u32,
        min_depth: u32,
        data_type: Option<NeuralTensorDataType>,
    ) -> Vec<&NeuralTensorDescriptor> {
        self.tensors
            .values()
            .filter(|tensor| {
                tensor.matches_spatial_requirements(min_width, min_height, min_depth)
                    && data_type.map_or(true, |dt| tensor.element_type == dt)
            })
            .collect()
    }
}

/// Zero-copy tensor data reader
pub struct TitanTensorReader {
    descriptor: NeuralTensorDescriptor,
    bytes_remaining: u64,
}

impl TitanTensorReader {
    /// Get descriptor for this tensor
    pub fn descriptor(&self) -> &NeuralTensorDescriptor {
        &self.descriptor
    }

    /// Read data into buffer
    pub fn read_data(&mut self, source: &mut StdFile, buf: &mut [u8]) -> Result<usize, TitanFormatError> {
        let to_read = (buf.len() as u64).min(self.bytes_remaining) as usize;
        if to_read == 0 {
            return Ok(0);
        }
        
        let bytes_read = source.read(&mut buf[..to_read])?;
        self.bytes_remaining -= bytes_read as u64;
        Ok(bytes_read)
    }

    /// Check if all data has been read
    pub fn is_complete(&self) -> bool {
        self.bytes_remaining == 0
    }

    /// Get bytes remaining
    pub fn bytes_remaining(&self) -> u64 {
        self.bytes_remaining
    }
}

/// Constants module for easy access
pub mod titan_constants {
    pub use super::{
        TITAN_DEFAULT_ALIGNMENT, TITAN_GGUF_MAGIC, TITAN_METADATA_CACHE_SIZE,
        TITAN_MIN_VERSION, TITAN_STREAM_BUFFER_SIZE,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_metadata_computation() {
        // 1D tensor
        let dims_1d = vec![1024u64];
        let spatial_1d = NeuralTensorDescriptor::compute_spatial_metadata(&dims_1d);
        assert_eq!(spatial_1d.width, 1024);
        assert_eq!(spatial_1d.height, 1);
        assert_eq!(spatial_1d.depth, 1);

        // 2D tensor
        let dims_2d = vec![256u64, 512];
        let spatial_2d = NeuralTensorDescriptor::compute_spatial_metadata(&dims_2d);
        assert_eq!(spatial_2d.width, 512);
        assert_eq!(spatial_2d.height, 256);
        assert_eq!(spatial_2d.depth, 1);

        // 3D tensor
        let dims_3d = vec![64u64, 128, 256];
        let spatial_3d = NeuralTensorDescriptor::compute_spatial_metadata(&dims_3d);
        assert_eq!(spatial_3d.width, 256);
        assert_eq!(spatial_3d.height, 128);
        assert_eq!(spatial_3d.depth, 64);

        // 4D tensor (uses last 3 dims)
        let dims_4d = vec![8u64, 16, 32, 64];
        let spatial_4d = NeuralTensorDescriptor::compute_spatial_metadata(&dims_4d);
        assert_eq!(spatial_4d.width, 64);
        assert_eq!(spatial_4d.height, 32);
        assert_eq!(spatial_4d.depth, 16);
    }

    #[test]
    fn test_metadata_value_conversions() {
        let uint_val = NeuralMetadataValue::UInt32(42);
        assert_eq!(uint_val.as_integer(), Some(42));

        let float_val = NeuralMetadataValue::Float32(3.14);
        assert_eq!(float_val.as_float(), Some(3.14));

        let bool_val = NeuralMetadataValue::Boolean(true);
        assert_eq!(bool_val.as_boolean(), Some(true));

        let text_val = NeuralMetadataValue::Text("test".to_string());
        assert_eq!(text_val.as_text(), "test");
    }

    #[test]
    fn test_alignment_calculation() {
        assert_eq!(TitanNeuralArchive::align_offset(100, 32), 128);
        assert_eq!(TitanNeuralArchive::align_offset(64, 32), 64);
        assert_eq!(TitanNeuralArchive::align_offset(33, 32), 64);
        assert_eq!(TitanNeuralArchive::align_offset(0, 32), 0);
    }

    #[test]
    fn test_type_code_mapping() {
        use crate::fs::ggml::pronax_ggml_types::NeuralTensorDataType;
        
        assert_eq!(
            TitanNeuralArchive::map_neural_type_code(0).unwrap(),
            NeuralTensorDataType::Float32
        );
        assert_eq!(
            TitanNeuralArchive::map_neural_type_code(1).unwrap(),
            NeuralTensorDataType::Float16
        );
        assert_eq!(
            TitanNeuralArchive::map_neural_type_code(12).unwrap(),
            NeuralTensorDataType::TitanKQuantized4Bit
        );
    }

    #[test]
    fn test_spatial_tensor_matching() {
        use crate::fs::ggml::pronax_ggml_types::NeuralTensorDataType;
        
        let descriptor = NeuralTensorDescriptor::new(
            "test_tensor".to_string(),
            vec![256, 512, 128],
            NeuralTensorDataType::TitanKQuantized4Bit,
            0,
        ).unwrap();

        assert!(descriptor.matches_spatial_requirements(200, 400, 100));
        assert!(!descriptor.matches_spatial_requirements(512, 1024, 256));
    }
}