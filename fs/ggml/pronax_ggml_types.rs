use std::fmt;
use std::num::NonZeroU32;
use std::str::FromStr;
use std::sync::Arc;

/// Spatial metadata for 3D tensor understanding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SpatialTensorMetadata {
    /// Width dimension for spatial reasoning
    pub width: u32,
    /// Height dimension for spatial reasoning  
    pub height: u32,
    /// Depth dimension for 3D neural layouts
    pub depth: u32,
    /// Guidance factor for AI processing
    pub guidance_scale: u16,
}

impl SpatialTensorMetadata {
    /// Create new 3D spatial metadata
    #[inline]
    pub const fn new(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
            guidance_scale: 100, // Default 1.0 scaled by 100
        }
    }

    /// Calculate total volume for memory planning
    #[inline]
    pub const fn volume(&self) -> u64 {
        (self.width as u64) * (self.height as u64) * (self.depth as u64)
    }

    /// Zero-copy spatial view creation
    #[inline]
    pub fn as_neural_layout(&self) -> NeuralSpatialLayout {
        NeuralSpatialLayout {
            dimensions: [self.width, self.height, self.depth],
            guidance: self.guidance_scale,
        }
    }
}

/// Neural spatial layout for zero-copy tensor operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeuralSpatialLayout {
    pub dimensions: [u32; 3],
    pub guidance: u16,
}

/// Advanced quantization format identifier for GGML/GGUF models
/// Enhanced with 3D spatial awareness for neural processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum QuantizationFormat {
    /// Full precision 32-bit float
    PrecisionFloat32 = 0,
    /// Half precision 16-bit float
    PrecisionFloat16 = 1,
    /// Internal 4-bit quantization (Q4_0)
    Quantized4Bit0 = 2,
    /// Internal 4-bit quantization (Q4_1)
    Quantized4Bit1 = 3,
    /// MXFP4 format for advanced inference
    MicroFloat4 = 4,
    /// Legacy unused format
    LegacyQuantized4Bit2 = 5,
    /// Legacy unused format
    LegacyQuantized4Bit3 = 6,
    /// 8-bit quantization for balanced quality/speed
    OptimizedQuantized8Bit = 7,
    /// 5-bit quantization variant 0
    OptimizedQuantized5Bit0 = 8,
    /// 5-bit quantization variant 1
    OptimizedQuantized5Bit1 = 9,
    /// K-quantization 2-bit for extreme compression
    KQuantized2Bit = 10,
    /// K-quantization 3-bit small variant
    KQuantized3BitSmall = 11,
    /// K-quantization 3-bit medium variant
    KQuantized3BitMedium = 12,
    /// K-quantization 3-bit large variant
    KQuantized3BitLarge = 13,
    /// K-quantization 4-bit small variant (ProNax Optimized)
    TitanKQuantized4BitSmall = 14,
    /// K-quantization 4-bit medium variant (ProNax Optimized)
    TitanKQuantized4BitMedium = 15,
    /// K-quantization 5-bit small variant
    KQuantized5BitSmall = 16,
    /// K-quantization 5-bit medium variant
    KQuantized5BitMedium = 17,
    /// K-quantization 6-bit for high quality
    KQuantized6Bit = 18,
    /// IQ quantization 2-bit extra extra small (Intel)
    ImatQuantized2BitXXS = 19,
    /// IQ quantization 2-bit extra small (Intel)
    ImatQuantized2BitXS = 20,
    /// K-quantization 2-bit small variant
    KQuantized2BitSmall = 21,
    /// IQ quantization 3-bit extra small (Intel)
    ImatQuantized3BitXS = 22,
    /// IQ quantization 3-bit extra extra small (Intel)
    ImatQuantized3BitXXS = 23,
    /// IQ quantization 1-bit small (Intel)
    ImatQuantized1BitSmall = 24,
    /// IQ quantization 4-bit non-linear (Intel)
    ImatQuantized4BitNL = 25,
    /// IQ quantization 3-bit small (Intel)
    ImatQuantized3BitSmall = 26,
    /// IQ quantization 3-bit medium (Intel)
    ImatQuantized3BitMedium = 27,
    /// IQ quantization 2-bit small (Intel)
    ImatQuantized2BitSmall = 28,
    /// IQ quantization 2-bit medium (Intel)
    ImatQuantized2BitMedium = 29,
    /// IQ quantization 4-bit extra small (Intel)
    ImatQuantized4BitXS = 30,
    /// IQ quantization 1-bit medium (Intel)
    ImatQuantized1BitMedium = 31,
    /// Brain floating point 16-bit (bfloat16)
    BrainFloat16 = 32,
    /// Block quantization 4_4 format (unused)
    BlockQuantized4_4 = 33,
    /// Block quantization 4_8 format (unused)
    BlockQuantized4_8 = 34,
    /// Block quantization 8_8 format (unused)
    BlockQuantized8_8 = 35,
    /// TQ1_0 quantization format
    TensorQuantized1_0 = 36,
    /// TQ2_0 quantization format
    TensorQuantized2_0 = 37,
    /// Unknown/unsupported format placeholder
    UnknownFormat = 1024,
}

impl QuantizationFormat {
    /// Get all ProNax-supported quantization formats
    /// Zero-copy slice return - no allocation
    #[inline]
    pub const fn supported_titan_formats() -> &'static [Self] {
        &[
            Self::PrecisionFloat32,
            Self::PrecisionFloat16,
            Self::TitanKQuantized4BitSmall,
            Self::TitanKQuantized4BitMedium,
            Self::OptimizedQuantized8Bit,
            Self::BrainFloat16,
        ]
    }

    /// Check if format supports 3D spatial optimization
    #[inline]
    pub const fn has_spatial_acceleration(&self) -> bool {
        matches!(
            self,
            Self::TitanKQuantized4BitSmall
                | Self::TitanKQuantized4BitMedium
                | Self::OptimizedQuantized8Bit
                | Self::BrainFloat16
        )
    }

    /// Get spatial metadata recommendation for this format
    #[inline]
    pub const fn recommended_spatial_config(&self) -> SpatialTensorMetadata {
        match self {
            Self::TitanKQuantized4BitSmall => SpatialTensorMetadata::new(512, 512, 128),
            Self::TitanKQuantized4BitMedium => SpatialTensorMetadata::new(1024, 1024, 256),
            Self::OptimizedQuantized8Bit => SpatialTensorMetadata::new(768, 768, 192),
            Self::BrainFloat16 => SpatialTensorMetadata::new(2048, 2048, 512),
            _ => SpatialTensorMetadata::new(256, 256, 64),
        }
    }

    /// Convert to internal representation (zero-copy)
    #[inline]
    pub const fn as_internal_code(&self) -> u32 {
        *self as u32
    }

    /// Check if format is supported by ProNax AI engine
    #[inline]
    pub const fn is_titan_compatible(&self) -> bool {
        matches!(
            self,
            Self::PrecisionFloat32
                | Self::PrecisionFloat16
                | Self::TitanKQuantized4BitSmall
                | Self::TitanKQuantized4BitMedium
                | Self::OptimizedQuantized8Bit
                | Self::BrainFloat16
        )
    }
}

impl fmt::Display for QuantizationFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::PrecisionFloat32 => "F32",
            Self::PrecisionFloat16 => "F16",
            Self::Quantized4Bit0 => "Q4_0",
            Self::Quantized4Bit1 => "Q4_1",
            Self::MicroFloat4 => "MXFP4",
            Self::OptimizedQuantized8Bit => "Q8_0",
            Self::OptimizedQuantized5Bit0 => "Q5_0",
            Self::OptimizedQuantized5Bit1 => "Q5_1",
            Self::KQuantized2Bit => "Q2_K",
            Self::KQuantized3BitSmall => "Q3_K_S",
            Self::KQuantized3BitMedium => "Q3_K_M",
            Self::KQuantized3BitLarge => "Q3_K_L",
            Self::TitanKQuantized4BitSmall => "Q4_K_S",
            Self::TitanKQuantized4BitMedium => "Q4_K_M",
            Self::KQuantized5BitSmall => "Q5_K_S",
            Self::KQuantized5BitMedium => "Q5_K_M",
            Self::KQuantized6Bit => "Q6_K",
            Self::KQuantized2BitSmall => "Q2_K_S",
            Self::BrainFloat16 => "BF16",
            _ => "unknown",
        };
        write!(f, "{}", name)
    }
}

impl FromStr for QuantizationFormat {
    type Err = NeuralFormatError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input {
            "F32" => Ok(Self::PrecisionFloat32),
            "F16" => Ok(Self::PrecisionFloat16),
            "Q8_0" => Ok(Self::OptimizedQuantized8Bit),
            "Q4_K_S" => Ok(Self::TitanKQuantized4BitSmall),
            "Q4_K_M" | "Q4_K" => Ok(Self::TitanKQuantized4BitMedium),
            "BF16" => Ok(Self::BrainFloat16),
            _ => Err(NeuralFormatError::UnsupportedQuantization {
                format_name: input.to_string(),
                supported: Self::supported_titan_formats()
                    .iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            }),
        }
    }
}

/// Tensor data type for neural computation
/// Enhanced with 3D spatial metadata for advanced AI processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum NeuralTensorDataType {
    /// 32-bit floating point
    Float32 = 0,
    /// 16-bit floating point (half precision)
    Float16 = 1,
    /// 4-bit quantization variant 0
    Quantized4Bit0 = 2,
    /// 4-bit quantization variant 1
    Quantized4Bit1 = 3,
    /// Legacy format (unused)
    LegacyUnused4Bit2 = 4,
    /// Legacy format (unused)
    LegacyUnused4Bit3 = 5,
    /// 5-bit quantization variant 0
    Quantized5Bit0 = 6,
    /// 5-bit quantization variant 1
    Quantized5Bit1 = 7,
    /// 8-bit quantization variant 0 (Titan optimized)
    TitanQuantized8Bit0 = 8,
    /// 8-bit quantization variant 1
    Quantized8Bit1 = 9,
    /// K-quantization 2-bit
    KQuantized2Bit = 10,
    /// K-quantization 3-bit
    KQuantized3Bit = 11,
    /// K-quantization 4-bit (Titan optimized)
    TitanKQuantized4Bit = 12,
    /// K-quantization 5-bit
    KQuantized5Bit = 13,
    /// K-quantization 6-bit
    KQuantized6Bit = 14,
    /// K-quantization 8-bit
    KQuantized8Bit = 15,
    /// IQ 2-bit XXS (not supported)
    Imat2BitXXS = 16,
    /// IQ 2-bit XS (not supported)
    Imat2BitXS = 17,
    /// IQ 3-bit XXS (not supported)
    Imat3BitXXS = 18,
    /// IQ 1-bit S (not supported)
    Imat1BitSmall = 19,
    /// IQ 4-bit NL (not supported)
    Imat4BitNL = 20,
    /// IQ 3-bit S (not supported)
    Imat3BitSmall = 21,
    /// IQ 2-bit S (not supported)
    Imat2BitSmall = 22,
    /// IQ 4-bit XS (not supported)
    Imat4BitXS = 23,
    /// 8-bit signed integer
    Int8 = 24,
    /// 16-bit signed integer
    Int16 = 25,
    /// 32-bit signed integer
    Int32 = 26,
    /// 64-bit signed integer
    Int64 = 27,
    /// 64-bit floating point
    Float64 = 28,
    /// IQ 1-bit M (not supported)
    Imat1BitMedium = 29,
    /// Brain float 16
    BrainFloat16 = 30,
    /// Block Q4_0_4_4 (unused)
    BlockQ4_0_4_4 = 31,
    /// Block Q4_0_4_8 (unused)
    BlockQ4_0_4_8 = 32,
    /// Block Q4_0_8_8 (unused)
    BlockQ4_0_8_8 = 33,
    /// Tensor quantized 1.0
    TensorQ1_0 = 34,
    /// Tensor quantized 2.0
    TensorQ2_0 = 35,
    /// Block IQ4_NL_4_4 (unused)
    BlockImat4BitNL4_4 = 36,
    /// Block IQ4_NL_4_8 (unused)
    BlockImat4BitNL4_8 = 37,
    /// Block IQ4_NL_8_8 (unused)
    BlockImat4BitNL8_8 = 38,
    /// Micro float 4 (MXFP4)
    MicroFloat4 = 39,
}

impl NeuralTensorDataType {
    /// Get the size of each element in bytes
    #[inline]
    pub const fn element_size_bytes(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 => 2,
            Self::Quantized4Bit0 | Self::Quantized4Bit1 => 1, // 4-bit packed
            Self::Quantized5Bit0 | Self::Quantized5Bit1 => 1,
            Self::TitanQuantized8Bit0 | Self::Quantized8Bit1 => 1,
            Self::KQuantized2Bit => 1,
            Self::KQuantized3Bit => 1,
            Self::TitanKQuantized4Bit => 1,
            Self::KQuantized5Bit => 1,
            Self::KQuantized6Bit => 1,
            Self::KQuantized8Bit => 1,
            Self::Int8 => 1,
            Self::Int16 => 2,
            Self::Int32 => 4,
            Self::Int64 => 8,
            Self::Float64 => 8,
            Self::BrainFloat16 => 2,
            Self::MicroFloat4 => 1,
            _ => 1,
        }
    }

    /// Get block size for quantization types
    #[inline]
    pub const fn block_elements(&self) -> usize {
        match self {
            Self::Quantized4Bit0 | Self::Quantized4Bit1 => 32,
            Self::Quantized5Bit0 | Self::Quantized5Bit1 => 32,
            Self::TitanQuantized8Bit0 | Self::Quantized8Bit1 => 32,
            Self::KQuantized2Bit => 256,
            Self::KQuantized3Bit => 256,
            Self::TitanKQuantized4Bit => 256,
            Self::KQuantized5Bit => 256,
            Self::KQuantized6Bit => 256,
            Self::KQuantized8Bit => 256,
            _ => 1,
        }
    }

    /// Check if this type uses quantization (lossy compression)
    #[inline]
    pub const fn is_compressed(&self) -> bool {
        !matches!(
            self,
            Self::Float32 | Self::Float16 | Self::BrainFloat16 | Self::Float64
        )
    }

    /// Check if type supports 3D spatial acceleration
    #[inline]
    pub const fn supports_neural_3d(&self) -> bool {
        matches!(
            self,
            Self::TitanKQuantized4Bit
                | Self::TitanQuantized8Bit0
                | Self::BrainFloat16
                | Self::Float16
                | Self::Float32
        )
    }

    /// Calculate row size in bytes for given element count
    /// Zero-copy computation - no heap allocation
    #[inline]
    pub const fn compute_row_bytes(&self, element_count: u64) -> u64 {
        let type_size = self.element_size_bytes() as u64;
        let block_size = self.block_elements() as u64;
        (type_size * element_count) / block_size
    }

    /// Get spatial metadata for optimal 3D processing
    #[inline]
    pub const fn spatial_processing_config(&self) -> SpatialTensorMetadata {
        match self {
            Self::TitanKQuantized4Bit => SpatialTensorMetadata::new(1024, 1024, 256),
            Self::TitanQuantized8Bit0 => SpatialTensorMetadata::new(768, 768, 192),
            Self::BrainFloat16 => SpatialTensorMetadata::new(2048, 2048, 512),
            Self::Float16 => SpatialTensorMetadata::new(1536, 1536, 384),
            Self::Float32 => SpatialTensorMetadata::new(1024, 1024, 256),
            _ => SpatialTensorMetadata::new(512, 512, 128),
        }
    }

    /// Get byte size per block for memory mapping
    #[inline]
    pub const fn block_byte_size(&self) -> usize {
        match self {
            Self::Quantized4Bit0 => 18, // 32 4-bit weights + 2x f16 scales
            Self::Quantized4Bit1 => 20, // 32 4-bit weights + 2x f16 scales + 2x f16 mins
            Self::TitanQuantized8Bit0 => 34, // 32 weights + 2x f16 scales
            Self::Quantized8Bit1 => 35,
            Self::Quantized5Bit0 => 22,
            Self::Quantized5Bit1 => 24,
            Self::KQuantized2Bit => 84,
            Self::KQuantized3Bit => 110,
            Self::TitanKQuantized4Bit => 144,
            Self::KQuantized5Bit => 176,
            Self::KQuantized6Bit => 210,
            Self::KQuantized8Bit => 292,
            Self::BrainFloat16 => 2,
            Self::Float16 => 2,
            Self::Float32 => 4,
            Self::Int8 => 1,
            Self::Int16 => 2,
            Self::Int32 => 4,
            Self::Int64 => 8,
            Self::Float64 => 8,
            _ => 1,
        }
    }
}

impl fmt::Display for NeuralTensorDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Float32 => "F32",
            Self::Float16 => "F16",
            Self::Quantized4Bit0 => "Q4_0",
            Self::Quantized4Bit1 => "Q4_1",
            Self::Quantized5Bit0 => "Q5_0",
            Self::Quantized5Bit1 => "Q5_1",
            Self::TitanQuantized8Bit0 => "Q8_0",
            Self::Quantized8Bit1 => "Q8_1",
            Self::KQuantized2Bit => "Q2_K",
            Self::KQuantized3Bit => "Q3_K",
            Self::TitanKQuantized4Bit => "Q4_K",
            Self::KQuantized5Bit => "Q5_K",
            Self::KQuantized6Bit => "Q6_K",
            Self::KQuantized8Bit => "Q8_K",
            Self::Float64 => "F64",
            Self::BrainFloat16 => "BF16",
            Self::MicroFloat4 => "MXFP4",
            Self::Int8 => "I8",
            Self::Int16 => "I16",
            Self::Int32 => "I32",
            Self::Int64 => "I64",
            _ => "unknown",
        };
        write!(f, "{}", name)
    }
}

impl FromStr for NeuralTensorDataType {
    type Err = NeuralFormatError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input {
            "F32" => Ok(Self::Float32),
            "F16" => Ok(Self::Float16),
            "Q4_0" => Ok(Self::Quantized4Bit0),
            "Q4_1" => Ok(Self::Quantized4Bit1),
            "Q5_0" => Ok(Self::Quantized5Bit0),
            "Q5_1" => Ok(Self::Quantized5Bit1),
            "Q8_0" => Ok(Self::TitanQuantized8Bit0),
            "Q8_1" => Ok(Self::Quantized8Bit1),
            "Q2_K" => Ok(Self::KQuantized2Bit),
            "Q3_K" => Ok(Self::KQuantized3Bit),
            "Q4_K" => Ok(Self::TitanKQuantized4Bit),
            "Q5_K" => Ok(Self::KQuantized5Bit),
            "Q6_K" => Ok(Self::KQuantized6Bit),
            "Q8_K" => Ok(Self::KQuantized8Bit),
            "F64" => Ok(Self::Float64),
            "BF16" => Ok(Self::BrainFloat16),
            "MXFP4" => Ok(Self::MicroFloat4),
            "I8" => Ok(Self::Int8),
            "I16" => Ok(Self::Int16),
            "I32" => Ok(Self::Int32),
            "I64" => Ok(Self::Int64),
            _ => Err(NeuralFormatError::UnsupportedDataType {
                type_name: input.to_string(),
            }),
        }
    }
}

/// Enhanced error type for neural format operations
#[derive(Debug, Clone, PartialEq)]
pub enum NeuralFormatError {
    UnsupportedQuantization {
        format_name: String,
        supported: String,
    },
    UnsupportedDataType {
        type_name: String,
    },
    InvalidSpatialConfig {
        reason: String,
    },
}

impl fmt::Display for NeuralFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedQuantization { format_name, supported } => {
                write!(f, "Quantization format '{}' not supported by ProNax Titan engine. Supported: [{}]", 
                    format_name, supported)
            }
            Self::UnsupportedDataType { type_name } => {
                write!(f, "Tensor data type '{}' is not recognized", type_name)
            }
            Self::InvalidSpatialConfig { reason } => {
                write!(f, "Invalid 3D spatial configuration: {}", reason)
            }
        }
    }
}

impl std::error::Error for NeuralFormatError {}

/// Zero-copy tensor descriptor with 3D spatial metadata
#[derive(Debug, Clone)]
pub struct NeuralTensorDescriptor {
    /// Data type of tensor elements
    pub data_type: NeuralTensorDataType,
    /// 3D spatial dimensions
    pub spatial: SpatialTensorMetadata,
    /// Total element count (zero-copy precomputed)
    pub element_count: NonZeroU32,
    /// Memory layout optimization flag
    pub optimized_for_3d: bool,
}

impl NeuralTensorDescriptor {
    /// Create new tensor descriptor with spatial metadata
    #[inline]
    pub fn new(
        data_type: NeuralTensorDataType,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Result<Self, NeuralFormatError> {
        let count = (width as u64)
            .checked_mul(height as u64)
            .and_then(|v| v.checked_mul(depth as u64))
            .filter(|&v| v > 0 && v <= u32::MAX as u64)
            .map(|v| unsafe { NonZeroU32::new_unchecked(v as u32) })
            .ok_or_else(|| NeuralFormatError::InvalidSpatialConfig {
                reason: format!("Dimensions {}x{}x{} exceed capacity", width, height, depth),
            })?;

        Ok(Self {
            data_type,
            spatial: SpatialTensorMetadata::new(width, height, depth),
            element_count: count,
            optimized_for_3d: data_type.supports_neural_3d(),
        })
    }

    /// Compute total memory size in bytes (zero-copy calculation)
    #[inline]
    pub const fn total_memory_bytes(&self) -> u64 {
        self.data_type.compute_row_bytes(self.element_count.get() as u64)
    }

    /// Get memory layout for zero-copy operations
    #[inline]
    pub fn memory_layout(&self) -> NeuralMemoryLayout {
        NeuralMemoryLayout {
            row_stride: self.data_type.element_size_bytes() as u64,
            plane_stride: (self.data_type.element_size_bytes() as u64) * (self.spatial.width as u64),
            depth_stride: (self.data_type.element_size_bytes() as u64)
                * (self.spatial.width as u64)
                * (self.spatial.height as u64),
        }
    }
}

/// Memory layout for zero-copy tensor access
#[derive(Debug, Clone, Copy)]
pub struct NeuralMemoryLayout {
    pub row_stride: u64,
    pub plane_stride: u64,
    pub depth_stride: u64,
}

/// Convert quantization format to tensor data type
/// Zero-copy mapping with 3D spatial enhancement
#[inline]
pub fn map_quantization_to_tensor_dtype(
    quant_format: QuantizationFormat,
) -> Option<NeuralTensorDataType> {
    match quant_format {
        QuantizationFormat::PrecisionFloat32 => Some(NeuralTensorDataType::Float32),
        QuantizationFormat::PrecisionFloat16 => Some(NeuralTensorDataType::Float16),
        QuantizationFormat::Quantized4Bit0 => Some(NeuralTensorDataType::Quantized4Bit0),
        QuantizationFormat::Quantized4Bit1 => Some(NeuralTensorDataType::Quantized4Bit1),
        QuantizationFormat::OptimizedQuantized8Bit => Some(NeuralTensorDataType::TitanQuantized8Bit0),
        QuantizationFormat::OptimizedQuantized5Bit0 => Some(NeuralTensorDataType::Quantized5Bit0),
        QuantizationFormat::OptimizedQuantized5Bit1 => Some(NeuralTensorDataType::Quantized5Bit1),
        QuantizationFormat::KQuantized2Bit => Some(NeuralTensorDataType::KQuantized2Bit),
        QuantizationFormat::KQuantized3BitSmall
        | QuantizationFormat::KQuantized3BitMedium
        | QuantizationFormat::KQuantized3BitLarge => Some(NeuralTensorDataType::KQuantized3Bit),
        QuantizationFormat::TitanKQuantized4BitSmall | QuantizationFormat::TitanKQuantized4BitMedium => {
            Some(NeuralTensorDataType::TitanKQuantized4Bit)
        }
        QuantizationFormat::KQuantized5BitSmall | QuantizationFormat::KQuantized5BitMedium => {
            Some(NeuralTensorDataType::KQuantized5Bit)
        }
        QuantizationFormat::KQuantized6Bit => Some(NeuralTensorDataType::KQuantized6Bit),
        QuantizationFormat::KQuantized2BitSmall => Some(NeuralTensorDataType::KQuantized2Bit),
        QuantizationFormat::BrainFloat16 => Some(NeuralTensorDataType::BrainFloat16),
        QuantizationFormat::MicroFloat4 => Some(NeuralTensorDataType::MicroFloat4),
        _ => {
            eprintln!("Warning: Quantization format {:?} not mapped to tensor type", quant_format);
            None
        }
    }
}

/// Thread-safe shared tensor type reference (zero-copy Arc)
pub type SharedTensorType = Arc<NeuralTensorDataType>;

/// Create zero-copy shared reference to tensor type
#[inline]
pub fn share_tensor_dtype(dtype: NeuralTensorDataType) -> SharedTensorType {
    Arc::new(dtype)
}

/// Module constants for quick type access
pub mod titan_types {
    use super::*;

    /// Titan-optimized Q4_K small format
    pub const TITAN_Q4KS: QuantizationFormat = QuantizationFormat::TitanKQuantized4BitSmall;
    /// Titan-optimized Q4_K medium format
    pub const TITAN_Q4KM: QuantizationFormat = QuantizationFormat::TitanKQuantized4BitMedium;
    /// Titan-optimized Q8_0 format
    pub const TITAN_Q8: QuantizationFormat = QuantizationFormat::OptimizedQuantized8Bit;
    /// Brain float 16 for high precision
    pub const TITAN_BF16: QuantizationFormat = QuantizationFormat::BrainFloat16;
    /// Full precision F32
    pub const TITAN_F32: QuantizationFormat = QuantizationFormat::PrecisionFloat32;
    /// Half precision F16
    pub const TITAN_F16: QuantizationFormat = QuantizationFormat::PrecisionFloat16;

    /// Tensor types
    pub const TENSOR_F32: NeuralTensorDataType = NeuralTensorDataType::Float32;
    pub const TENSOR_F16: NeuralTensorDataType = NeuralTensorDataType::Float16;
    pub const TENSOR_Q4K: NeuralTensorDataType = NeuralTensorDataType::TitanKQuantized4Bit;
    pub const TENSOR_Q8: NeuralTensorDataType = NeuralTensorDataType::TitanQuantized8Bit0;
    pub const TENSOR_BF16: NeuralTensorDataType = NeuralTensorDataType::BrainFloat16;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_format_parsing() {
        assert_eq!(
            "F32".parse::<QuantizationFormat>().unwrap(),
            QuantizationFormat::PrecisionFloat32
        );
        assert_eq!(
            "Q4_K_S".parse::<QuantizationFormat>().unwrap(),
            QuantizationFormat::TitanKQuantized4BitSmall
        );
        assert_eq!(
            "BF16".parse::<QuantizationFormat>().unwrap(),
            QuantizationFormat::BrainFloat16
        );
    }

    #[test]
    fn test_tensor_type_properties() {
        let f32_type = NeuralTensorDataType::Float32;
        assert!(!f32_type.is_compressed());
        assert_eq!(f32_type.element_size_bytes(), 4);
        assert!(f32_type.supports_neural_3d());

        let q4k_type = NeuralTensorDataType::TitanKQuantized4Bit;
        assert!(q4k_type.is_compressed());
        assert_eq!(q4k_type.element_size_bytes(), 1);
        assert!(q4k_type.supports_neural_3d());
    }

    #[test]
    fn test_spatial_metadata() {
        let spatial = SpatialTensorMetadata::new(1024, 768, 256);
        assert_eq!(spatial.volume(), 1024 * 768 * 256);
        assert_eq!(spatial.width, 1024);
        assert_eq!(spatial.height, 768);
        assert_eq!(spatial.depth, 256);
    }

    #[test]
    fn test_tensor_descriptor() {
        let desc = NeuralTensorDescriptor::new(
            NeuralTensorDataType::TitanKQuantized4Bit,
            512,
            512,
            128,
        )
        .unwrap();
        assert!(desc.optimized_for_3d);
        assert_eq!(desc.element_count.get(), 512 * 512 * 128);
    }

    #[test]
    fn test_format_conversion() {
        let quant = QuantizationFormat::TitanKQuantized4BitSmall;
        let tensor = map_quantization_to_tensor_dtype(quant);
        assert_eq!(tensor, Some(NeuralTensorDataType::TitanKQuantized4Bit));
    }

    #[test]
    fn test_row_size_calculation() {
        let q8 = NeuralTensorDataType::TitanQuantized8Bit0;
        let row_bytes = q8.compute_row_bytes(256);
        // Q8_0: 34 bytes per 32 elements, so 256 elements = 2176 bytes
        assert_eq!(row_bytes, (34 * 256) / 32);
    }
}