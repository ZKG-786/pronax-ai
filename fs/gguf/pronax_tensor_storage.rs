use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// Titan quantization format with 3D spatial optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TitanQuantizationFormat {
    // Standard floating point
    NeuralF32,
    NeuralF16,
    NeuralBF16,
    NeuralF64,
    
    // Integer types
    NeuralI8,
    NeuralI16,
    NeuralI32,
    NeuralI64,
    
    // Standard quantization
    NeuralQ4_0,
    NeuralQ4_1,
    NeuralQ5_0,
    NeuralQ5_1,
    NeuralQ8_0,
    NeuralQ8_1,
    
    // K-quantization variants
    NeuralQ2_K,
    NeuralQ3_K,
    NeuralQ4_K,
    NeuralQ5_K,
    NeuralQ6_K,
    NeuralQ8_K,
    
    // I-quantization variants (unused but defined)
    NeuralIQ2_XXS,
    NeuralIQ2_XS,
    NeuralIQ3_XXS,
    NeuralIQ1_S,
    NeuralIQ4_NL,
    NeuralIQ3_S,
    NeuralIQ2_S,
    NeuralIQ4_XS,
    NeuralIQ1_M,
    
    // Tile variants (unused)
    NeuralQ4_0_4_4,
    NeuralQ4_0_4_8,
    NeuralQ4_0_8_8,
    NeuralTQ1_0,
    NeuralTQ2_0,
    NeuralIQ4_NL_4_4,
    NeuralIQ4_NL_4_8,
    NeuralIQ4_NL_8_8,
    
    // 3D spatial-optimized variants (ProNax unique)
    SpatialQ4_0(SpatialTensorMetadata),
    SpatialQ5_0(SpatialTensorMetadata),
    SpatialQ8_0(SpatialTensorMetadata),
    SpatialF16(SpatialTensorMetadata),
}

impl TitanQuantizationFormat {
    /// Get element size in bytes
    #[inline]
    pub fn element_bytes(&self) -> usize {
        match self {
            Self::NeuralF32 => 4,
            Self::NeuralF16 => 2,
            Self::NeuralBF16 => 2,
            Self::NeuralF64 => 8,
            Self::NeuralI8 => 1,
            Self::NeuralI16 => 2,
            Self::NeuralI32 => 4,
            Self::NeuralI64 => 8,
            _ => 0, // Quantized types calculated differently
        }
    }

    /// Get quantization block size
    #[inline]
    pub fn block_dimensions(&self) -> usize {
        match self {
            Self::NeuralF32 | Self::NeuralF16 | Self::NeuralBF16 |
            Self::NeuralI8 | Self::NeuralI16 | Self::NeuralI32 | Self::NeuralI64 | Self::NeuralF64 => 1,
            
            Self::NeuralQ4_0 | Self::NeuralQ4_1 | Self::NeuralQ5_0 |
            Self::NeuralQ5_1 | Self::NeuralQ8_0 | Self::NeuralQ8_1 |
            Self::NeuralIQ4_NL => 32,
            
            _ => 256, // K-quants and others
        }
    }

    /// Calculate type size per block in bytes
    #[inline]
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::NeuralF32 => 4,
            Self::NeuralF16 => 2,
            Self::NeuralBF16 => 2,
            Self::NeuralF64 => 8,
            Self::NeuralI8 => 1,
            Self::NeuralI16 => 2,
            Self::NeuralI32 => 4,
            Self::NeuralI64 => 8,
            
            Self::NeuralQ4_0 => 2 + 16, // scale + 32/2 quantized
            Self::NeuralQ4_1 => 2 + 2 + 16, // scale + min + quantized
            Self::NeuralQ5_0 => 2 + 4 + 16, // scale + 4-bit masks + quantized
            Self::NeuralQ5_1 => 2 + 2 + 4 + 16, // scale + min + masks + quantized
            Self::NeuralQ8_0 => 2 + 32, // scale + quantized
            Self::NeuralQ8_1 => 2 + 2 + 32, // scale + min + quantized
            
            Self::NeuralQ2_K => 16 + 64 + 2 + 2, // scales + quantized + scale/min
            Self::NeuralQ3_K => 32 + 64 + 12 + 2, // bits + quantized + scales + scale
            Self::NeuralQ4_K => 2 + 2 + 12 + 128, // scale/min + scales + quantized
            Self::NeuralQ5_K => 2 + 2 + 12 + 32 + 128, // scale/min + scales + bits + quantized
            Self::NeuralQ6_K => 128 + 64 + 16 + 2, // quantized + 6-bit + scales + scale
            Self::NeuralQ8_K => 4 + 256 + 2 * 16, // scale + quantized + scales
            
            Self::NeuralIQ2_XXS => 2 + 64, // scale + quantized
            Self::NeuralIQ2_XS => 2 + 64 + 8, // scale + quantized + extra
            Self::NeuralIQ3_XXS => 2 + 64 + 32, // scale + quantized + extra
            Self::NeuralIQ1_S => 2 + 32 + 16, // scale + quantized + extra
            Self::NeuralIQ4_NL => 2 + 128, // scale + quantized
            Self::NeuralIQ3_S => 2 + 64 + 32 + 8 + 4, // scale + quantized + extras
            Self::NeuralIQ2_S => 2 + 64 + 16, // scale + quantized + extra
            Self::NeuralIQ4_XS => 2 + 2 + 128 + 4, // scales + quantized + extra
            Self::NeuralIQ1_M => 32 + 16 + 8, // quantized components
            
            // Spatial variants use same calculation
            Self::SpatialQ4_0(_) => 2 + 16,
            Self::SpatialQ5_0(_) => 2 + 4 + 16,
            Self::SpatialQ8_0(_) => 2 + 32,
            Self::SpatialF16(_) => 2,
        }
    }

    /// Calculate bytes per element (including quantization)
    #[inline]
    pub fn bytes_per_element(&self) -> f64 {
        let block_bytes = self.bytes_per_block() as f64;
        let block_size = self.block_dimensions() as f64;
        block_bytes / block_size
    }

    /// Check if floating point type
    #[inline]
    pub fn is_floating_point(&self) -> bool {
        matches!(self, 
            Self::NeuralF32 | Self::NeuralF16 | Self::NeuralBF16 | Self::NeuralF64 |
            Self::SpatialF16(_)
        )
    }

    /// Check if quantized type
    #[inline]
    pub fn is_quantized(&self) -> bool {
        !self.is_floating_point() && !self.is_integer()
    }

    /// Check if integer type
    #[inline]
    pub fn is_integer(&self) -> bool {
        matches!(self,
            Self::NeuralI8 | Self::NeuralI16 | Self::NeuralI32 | Self::NeuralI64
        )
    }

    /// Check if has 3D spatial context
    #[inline]
    pub fn has_spatial_context(&self) -> bool {
        matches!(self,
            Self::SpatialQ4_0(_) | Self::SpatialQ5_0(_) |
            Self::SpatialQ8_0(_) | Self::SpatialF16(_)
        )
    }

    /// Get spatial context if available
    pub fn spatial_context(&self) -> Option<&SpatialTensorMetadata> {
        match self {
            Self::SpatialQ4_0(s) |
            Self::SpatialQ5_0(s) |
            Self::SpatialQ8_0(s) |
            Self::SpatialF16(s) => Some(s),
            _ => None,
        }
    }

    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NeuralF32 => "f32",
            Self::NeuralF16 => "f16",
            Self::NeuralBF16 => "bf16",
            Self::NeuralF64 => "f64",
            Self::NeuralI8 => "i8",
            Self::NeuralI16 => "i16",
            Self::NeuralI32 => "i32",
            Self::NeuralI64 => "i64",
            Self::NeuralQ4_0 => "q4_0",
            Self::NeuralQ4_1 => "q4_1",
            Self::NeuralQ5_0 => "q5_0",
            Self::NeuralQ5_1 => "q5_1",
            Self::NeuralQ8_0 => "q8_0",
            Self::NeuralQ8_1 => "q8_1",
            Self::NeuralQ2_K => "q2_k",
            Self::NeuralQ3_K => "q3_k",
            Self::NeuralQ4_K => "q4_k",
            Self::NeuralQ5_K => "q5_k",
            Self::NeuralQ6_K => "q6_k",
            Self::NeuralQ8_K => "q8_k",
            Self::NeuralIQ2_XXS => "iq2_xxs",
            Self::NeuralIQ2_XS => "iq2_xs",
            Self::NeuralIQ3_XXS => "iq3_xxs",
            Self::NeuralIQ1_S => "iq1_s",
            Self::NeuralIQ4_NL => "iq4_nl",
            Self::NeuralIQ3_S => "iq3_s",
            Self::NeuralIQ2_S => "iq2_s",
            Self::NeuralIQ4_XS => "iq4_xs",
            Self::NeuralIQ1_M => "iq1_m",
            Self::NeuralQ4_0_4_4 => "q4_0_4_4",
            Self::NeuralQ4_0_4_8 => "q4_0_4_8",
            Self::NeuralQ4_0_8_8 => "q4_0_8_8",
            Self::NeuralTQ1_0 => "tq1_0",
            Self::NeuralTQ2_0 => "tq2_0",
            Self::NeuralIQ4_NL_4_4 => "iq4_nl_4_4",
            Self::NeuralIQ4_NL_4_8 => "iq4_nl_4_8",
            Self::NeuralIQ4_NL_8_8 => "iq4_nl_8_8",
            Self::SpatialQ4_0(_) => "q4_0:spatial",
            Self::SpatialQ5_0(_) => "q5_0:spatial",
            Self::SpatialQ8_0(_) => "q8_0:spatial",
            Self::SpatialF16(_) => "f16:spatial",
        }
    }

    /// Create from string (basic types only)
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(Self::NeuralF32),
            "f16" => Some(Self::NeuralF16),
            "bf16" => Some(Self::NeuralBF16),
            "f64" => Some(Self::NeuralF64),
            "i8" => Some(Self::NeuralI8),
            "i16" => Some(Self::NeuralI16),
            "i32" => Some(Self::NeuralI32),
            "i64" => Some(Self::NeuralI64),
            "q4_0" => Some(Self::NeuralQ4_0),
            "q4_1" => Some(Self::NeuralQ4_1),
            "q5_0" => Some(Self::NeuralQ5_0),
            "q5_1" => Some(Self::NeuralQ5_1),
            "q8_0" => Some(Self::NeuralQ8_0),
            "q8_1" => Some(Self::NeuralQ8_1),
            "q2_k" => Some(Self::NeuralQ2_K),
            "q3_k" => Some(Self::NeuralQ3_K),
            "q4_k" => Some(Self::NeuralQ4_K),
            "q5_k" => Some(Self::NeuralQ5_K),
            "q6_k" => Some(Self::NeuralQ6_K),
            "q8_k" => Some(Self::NeuralQ8_K),
            _ => None,
        }
    }

    /// Create spatial variant with 3D context
    pub fn with_spatial(self, width: u32, height: u32, depth: u32) -> Self {
        let spatial = SpatialTensorMetadata::new(width, height, depth);
        match self {
            Self::NeuralQ4_0 => Self::SpatialQ4_0(spatial),
            Self::NeuralQ5_0 => Self::SpatialQ5_0(spatial),
            Self::NeuralQ8_0 => Self::SpatialQ8_0(spatial),
            Self::NeuralF16 => Self::SpatialF16(spatial),
            _ => self, // Return as-is if not supported
        }
    }

    /// Get all supported types
    pub fn all_types() -> Vec<Self> {
        vec![
            Self::NeuralF32, Self::NeuralF16, Self::NeuralBF16, Self::NeuralF64,
            Self::NeuralI8, Self::NeuralI16, Self::NeuralI32, Self::NeuralI64,
            Self::NeuralQ4_0, Self::NeuralQ4_1, Self::NeuralQ5_0, Self::NeuralQ5_1,
            Self::NeuralQ8_0, Self::NeuralQ8_1,
            Self::NeuralQ2_K, Self::NeuralQ3_K, Self::NeuralQ4_K,
            Self::NeuralQ5_K, Self::NeuralQ6_K, Self::NeuralQ8_K,
        ]
    }
}

impl fmt::Display for TitanQuantizationFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Neural tensor descriptor with 3D spatial information
#[derive(Debug, Clone, PartialEq)]
pub struct NeuralTensorDescriptor {
    /// Tensor identifier
    pub identifier: String,
    /// File offset for data
    pub storage_offset: u64,
    /// Tensor dimensions (shape)
    pub dimensions: Vec<u64>,
    /// Quantization format
    pub format: TitanQuantizationFormat,
    /// 3D spatial metadata
    pub spatial_metadata: SpatialTensorMetadata,
    /// Data alignment requirement
    pub alignment: u32,
}

impl NeuralTensorDescriptor {
    /// Create new tensor descriptor
    pub fn new(
        identifier: impl Into<String>,
        storage_offset: u64,
        dimensions: Vec<u64>,
        format: TitanQuantizationFormat,
    ) -> Self {
        let dims = &dimensions;
        let spatial = Self::compute_spatial_metadata(dims);
        
        Self {
            identifier: identifier.into(),
            storage_offset,
            dimensions,
            format,
            spatial_metadata: spatial,
            alignment: 64,
        }
    }

    /// Compute 3D spatial metadata from dimensions
    fn compute_spatial_metadata(dims: &[u64]) -> SpatialTensorMetadata {
        match dims.len() {
            0 => SpatialTensorMetadata::new(1, 1, 1),
            1 => SpatialTensorMetadata::new(dims[0] as u32, 1, 1),
            2 => SpatialTensorMetadata::new(dims[0] as u32, dims[1] as u32, 1),
            3 | _ => SpatialTensorMetadata::new(
                dims[0] as u32,
                dims[1] as u32,
                dims[2] as u32,
            ),
        }
    }

    /// Check if tensor is valid (has identifier and non-zero size)
    #[inline]
    pub fn is_valid(&self) -> bool {
        !self.identifier.is_empty() && self.total_bytes() > 0
    }

    /// Calculate total number of elements
    #[inline]
    pub fn total_elements(&self) -> u64 {
        self.dimensions.iter().product()
    }

    /// Calculate total bytes needed for storage
    #[inline]
    pub fn total_bytes(&self) -> u64 {
        let elements = self.total_elements() as f64;
        let bytes_per_element = self.format.bytes_per_element();
        (elements * bytes_per_element).ceil() as u64
    }

    /// Calculate aligned size
    #[inline]
    pub fn aligned_size(&self) -> u64 {
        let bytes = self.total_bytes();
        let align = self.alignment as u64;
        ((bytes + align - 1) / align) * align
    }

    /// Get number of dimensions
    #[inline]
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Check if scalar (0-D tensor)
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.dimensions.is_empty()
    }

    /// Check if vector (1-D tensor)
    #[inline]
    pub fn is_vector(&self) -> bool {
        self.dimensions.len() == 1
    }

    /// Check if matrix (2-D tensor)
    #[inline]
    pub fn is_matrix(&self) -> bool {
        self.dimensions.len() == 2
    }

    /// Check if 3-D tensor (volume)
    #[inline]
    pub fn is_volume(&self) -> bool {
        self.dimensions.len() >= 3
    }

    /// Get 3D spatial volume
    #[inline]
    pub fn spatial_volume(&self) -> u64 {
        self.spatial_metadata.volume()
    }

    /// Get format as string
    #[inline]
    pub fn format_name(&self) -> &'static str {
        self.format.as_str()
    }

    /// Create logging/debugging representation
    pub fn debug_info(&self) -> String {
        format!(
            "Tensor[{}] {}: offset={}, dims={:?}, elements={}, bytes={}, format={}",
            self.identifier,
            self.spatial_metadata,
            self.storage_offset,
            self.dimensions,
            self.total_elements(),
            self.total_bytes(),
            self.format_name()
        )
    }

    /// Check if tensor is quantized
    #[inline]
    pub fn is_quantized(&self) -> bool {
        self.format.is_quantized()
    }

    /// Estimate memory footprint with overhead
    pub fn estimated_memory(&self) -> u64 {
        let data_bytes = self.aligned_size();
        let overhead = (self.identifier.len() + 64) as u64; // String + struct overhead
        data_bytes + overhead
    }
}

impl Default for NeuralTensorDescriptor {
    fn default() -> Self {
        Self::new("", 0, vec![], TitanQuantizationFormat::NeuralF32)
    }
}

/// Titan tensor storage with zero-copy memory management
pub struct TitanTensorStorage {
    /// Tensor descriptors
    descriptors: Vec<NeuralTensorDescriptor>,
    /// Name to index mapping
    name_index: HashMap<String, usize>,
    /// Total storage bytes
    total_bytes: u64,
    /// Maximum tensor size
    max_tensor_size: u64,
    /// Spatial registry for 3D lookup
    spatial_registry: HashMap<(u32, u32, u32), Vec<String>>,
}

impl TitanTensorStorage {
    /// Create new empty storage
    pub fn new() -> Self {
        Self {
            descriptors: Vec::new(),
            name_index: HashMap::new(),
            total_bytes: 0,
            max_tensor_size: 0,
            spatial_registry: HashMap::new(),
        }
    }

    /// Register tensor descriptor
    pub fn register(&mut self, descriptor: NeuralTensorDescriptor) {
        let name = descriptor.identifier.clone();
        let spatial_key = (
            descriptor.spatial_metadata.width,
            descriptor.spatial_metadata.height,
            descriptor.spatial_metadata.depth,
        );
        
        // Update spatial registry
        self.spatial_registry
            .entry(spatial_key)
            .or_default()
            .push(name.clone());
        
        // Update statistics
        let bytes = descriptor.total_bytes();
        self.total_bytes += bytes;
        self.max_tensor_size = self.max_tensor_size.max(bytes);
        
        // Store descriptor
        let index = self.descriptors.len();
        self.descriptors.push(descriptor);
        self.name_index.insert(name, index);
    }

    /// Get tensor by name
    pub fn get(&self, name: &str) -> Option<&NeuralTensorDescriptor> {
        self.name_index.get(name).map(|&idx| &self.descriptors[idx])
    }

    /// Get mutable tensor by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut NeuralTensorDescriptor> {
        self.name_index.get(name).copied().map(|idx| &mut self.descriptors[idx])
    }

    /// Find tensors by spatial dimensions
    pub fn find_by_spatial(&self, width: u32, height: u32, depth: u32) -> Vec<&NeuralTensorDescriptor> {
        self.spatial_registry
            .get(&(width, height, depth))
            .map(|names| {
                names.iter()
                    .filter_map(|name| self.get(name))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find tensors by format
    pub fn find_by_format(&self, format: TitanQuantizationFormat) -> Vec<&NeuralTensorDescriptor> {
        self.descriptors
            .iter()
            .filter(|d| d.format == format)
            .collect()
    }

    /// Get total registered tensors
    #[inline]
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// Get total storage bytes
    #[inline]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Get maximum tensor size
    #[inline]
    pub fn max_tensor_size(&self) -> u64 {
        self.max_tensor_size
    }

    /// Iterate all descriptors
    pub fn iter(&self) -> impl Iterator<Item = &NeuralTensorDescriptor> {
        self.descriptors.iter()
    }

    /// Calculate alignment padding needed
    pub fn calculate_alignment_padding(&self, offset: u64) -> u64 {
        let alignment = 64u64;
        let remainder = offset % alignment;
        if remainder == 0 {
            0
        } else {
            alignment - remainder
        }
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> TensorMemoryStats {
        let total_tensors = self.descriptors.len();
        let quantized_count = self.descriptors.iter().filter(|d| d.is_quantized()).count();
        let fp_count = total_tensors - quantized_count;
        
        TensorMemoryStats {
            total_tensors,
            total_bytes: self.total_bytes,
            max_tensor_bytes: self.max_tensor_size,
            average_tensor_bytes: if total_tensors > 0 {
                self.total_bytes / total_tensors as u64
            } else {
                0
            },
            quantized_tensors: quantized_count,
            floating_point_tensors: fp_count,
        }
    }

    /// Get sorted tensors by offset (for sequential access)
    pub fn sorted_by_offset(&self) -> Vec<&NeuralTensorDescriptor> {
        let mut sorted: Vec<_> = self.descriptors.iter().collect();
        sorted.sort_by_key(|d| d.storage_offset);
        sorted
    }
}

impl Default for TitanTensorStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory statistics for tensor storage
#[derive(Debug, Clone, Copy)]
pub struct TensorMemoryStats {
    pub total_tensors: usize,
    pub total_bytes: u64,
    pub max_tensor_bytes: u64,
    pub average_tensor_bytes: u64,
    pub quantized_tensors: usize,
    pub floating_point_tensors: usize,
}

impl fmt::Display for TensorMemoryStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorMemoryStats {{ tensors: {}, total: {}MB, max: {}KB, avg: {}KB, q: {}, fp: {} }}",
            self.total_tensors,
            self.total_bytes / 1_048_576,
            self.max_tensor_bytes / 1024,
            self.average_tensor_bytes / 1024,
            self.quantized_tensors,
            self.floating_point_tensors
        )
    }
}

/// Tensor view for zero-copy access
pub struct NeuralTensorView<'a> {
    descriptor: &'a NeuralTensorDescriptor,
    data: &'a [u8],
}

impl<'a> NeuralTensorView<'a> {
    /// Create new tensor view
    pub fn new(descriptor: &'a NeuralTensorDescriptor, data: &'a [u8]) -> Self {
        Self { descriptor, data }
    }

    /// Get descriptor
    #[inline]
    pub fn descriptor(&self) -> &NeuralTensorDescriptor {
        self.descriptor
    }

    /// Get raw data slice
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Get data at index (unsafe - bounds checking on caller)
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &[u8] {
        let element_size = self.descriptor.format.bytes_per_block();
        let start = index * element_size;
        &self.data[start..start + element_size]
    }

    /// Check if data matches expected size
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.data.len() >= self.descriptor.total_bytes() as usize
    }
}

/// Type aliases
pub type QuantizationFormat = TitanQuantizationFormat;
pub type TensorDescriptor = NeuralTensorDescriptor;
pub type TensorStorage = TitanTensorStorage;
pub type TensorView<'a> = NeuralTensorView<'a>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_formats() {
        assert_eq!(QuantizationFormat::NeuralF32.element_bytes(), 4);
        assert_eq!(QuantizationFormat::NeuralF16.element_bytes(), 2);
        assert_eq!(QuantizationFormat::NeuralI8.element_bytes(), 1);
        
        assert!(QuantizationFormat::NeuralQ4_0.is_quantized());
        assert!(!QuantizationFormat::NeuralF32.is_quantized());
        assert!(QuantizationFormat::NeuralF32.is_floating_point());
    }

    #[test]
    fn test_block_sizes() {
        assert_eq!(QuantizationFormat::NeuralF32.block_dimensions(), 1);
        assert_eq!(QuantizationFormat::NeuralQ4_0.block_dimensions(), 32);
        assert_eq!(QuantizationFormat::NeuralQ2_K.block_dimensions(), 256);
    }

    #[test]
    fn test_bytes_per_element() {
        let f32_bytes = QuantizationFormat::NeuralF32.bytes_per_element();
        assert_eq!(f32_bytes, 4.0);
        
        let q4_0_bytes = QuantizationFormat::NeuralQ4_0.bytes_per_element();
        assert!(q4_0_bytes < 4.0); // Compressed
    }

    #[test]
    fn test_tensor_descriptor() {
        let desc = TensorDescriptor::new(
            "test_tensor",
            1024,
            vec![256, 256, 128],
            QuantizationFormat::NeuralF32,
        );
        
        assert_eq!(desc.identifier, "test_tensor");
        assert_eq!(desc.storage_offset, 1024);
        assert_eq!(desc.dimensions, vec![256, 256, 128]);
        assert_eq!(desc.total_elements(), 256 * 256 * 128);
        assert_eq!(desc.total_bytes(), 256 * 256 * 128 * 4);
        assert!(desc.is_volume());
    }

    #[test]
    fn test_spatial_metadata() {
        let desc = TensorDescriptor::new(
            "spatial_test",
            0,
            vec![64, 64, 64],
            QuantizationFormat::NeuralQ4_0,
        );
        
        assert_eq!(desc.spatial_metadata.width, 64);
        assert_eq!(desc.spatial_metadata.height, 64);
        assert_eq!(desc.spatial_metadata.depth, 64);
        assert_eq!(desc.spatial_volume(), 64 * 64 * 64);
    }

    #[test]
    fn test_tensor_storage() {
        let mut storage = TensorStorage::new();
        
        let desc1 = TensorDescriptor::new("tensor1", 0, vec![100, 100], QuantizationFormat::NeuralF32);
        let desc2 = TensorDescriptor::new("tensor2", 40000, vec![50, 50], QuantizationFormat::NeuralQ4_0);
        
        storage.register(desc1);
        storage.register(desc2);
        
        assert_eq!(storage.len(), 2);
        assert!(storage.get("tensor1").is_some());
        assert!(storage.get("tensor2").is_some());
        assert!(storage.get("tensor3").is_none());
    }

    #[test]
    fn test_spatial_lookup() {
        let mut storage = TensorStorage::new();
        
        let desc = TensorDescriptor::new(
            "spatial_tensor",
            0,
            vec![256, 256, 128],
            QuantizationFormat::SpatialQ4_0(SpatialTensorMetadata::new(256, 256, 128)),
        );
        storage.register(desc);
        
        let found = storage.find_by_spatial(256, 256, 128);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_memory_stats() {
        let mut storage = TensorStorage::new();
        
        storage.register(TensorDescriptor::new("fp32", 0, vec![100], QuantizationFormat::NeuralF32));
        storage.register(TensorDescriptor::new("q4", 400, vec![100], QuantizationFormat::NeuralQ4_0));
        storage.register(TensorDescriptor::new("f16", 500, vec![100], QuantizationFormat::NeuralF16));
        
        let stats = storage.memory_stats();
        assert_eq!(stats.total_tensors, 3);
        assert_eq!(stats.quantized_tensors, 1);
        assert_eq!(stats.floating_point_tensors, 2);
    }

    #[test]
    fn test_format_from_str() {
        assert_eq!(QuantizationFormat::from_str("f32"), Some(QuantizationFormat::NeuralF32));
        assert_eq!(QuantizationFormat::from_str("q4_0"), Some(QuantizationFormat::NeuralQ4_0));
        assert_eq!(QuantizationFormat::from_str("unknown"), None);
    }

    #[test]
    fn test_spatial_quantization() {
        let base = QuantizationFormat::NeuralQ4_0;
        let spatial = base.with_spatial(512, 512, 256);
        
        assert!(spatial.has_spatial_context());
        assert_eq!(spatial.as_str(), "q4_0:spatial");
    }

    #[test]
    fn test_tensor_view() {
        let desc = TensorDescriptor::new("view_test", 0, vec![10], QuantizationFormat::NeuralF32);
        let data = vec![0u8; 40]; // 10 * 4 bytes
        
        let view = TensorView::new(&desc, &data);
        assert!(view.is_valid());
        assert_eq!(view.data().len(), 40);
    }
}