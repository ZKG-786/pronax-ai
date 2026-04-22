use std::io::{self, Write};
use std::sync::Arc;

use crate::convert::pronax_converter_core::{ConverterError, ConversionCoordinate};
use crate::fs::ggml::pronax_ggml_types::{
    NeuralTensorDataType, SpatialTensorMetadata, NeuralMemoryLayout,
    NeuralTensorDescriptor,
};

/// 3D-aware tensor split configuration
/// Ollama's split struct ka ProNax 3D upgrade - width, height, depth guidance ke saath
#[derive(Debug, Clone)]
pub struct NeuralTensorFragment {
    /// Original tensor name transform pattern
    pub name_transform: Option<NameTransform>,
    /// Target dimension for spatial splitting (0=width, 1=height, 2=depth)
    pub spatial_axis: u8,
    /// Custom slice boundaries per dimension [width_start, width_end, height_start, ...]
    pub boundary_slices: Vec<SpatialBoundary>,
    /// Optional post-split transformation with 3D spatial awareness
    pub spatial_transform: Option<Arc<dyn Fn(NeuralTensorView) -> Result<NeuralTensorView, TensorOpError> + Send + Sync>>,
    /// 3D guidance factor for AI processing enhancement
    pub guidance_multiplier: f32,
}

impl NeuralTensorFragment {
    /// Create new fragment with axis alignment
    #[inline]
    pub fn axis_aligned(axis: u8, guidance: f32) -> Self {
        Self {
            name_transform: None,
            spatial_axis: axis.min(2),
            boundary_slices: Vec::new(),
            spatial_transform: None,
            guidance_multiplier: guidance.clamp(0.1, 10.0),
        }
    }

    /// Add name transformation pattern
    #[inline]
    pub fn with_name_transform(mut self, pattern: &str, replacement: &str) -> Self {
        self.name_transform = Some(NameTransform {
            pattern: pattern.to_string(),
            replacement: replacement.to_string(),
        });
        self
    }

    /// Add spatial boundary constraints
    #[inline]
    pub fn with_boundaries(mut self, boundaries: Vec<SpatialBoundary>) -> Self {
        self.boundary_slices = boundaries;
        self
    }
}

/// Name transformation for tensor identity
#[derive(Debug, Clone)]
pub struct NameTransform {
    pub pattern: String,
    pub replacement: String,
}

impl NameTransform {
    /// Apply transformation to tensor name
    #[inline]
    pub fn apply(&self, name: &str) -> String {
        // Simple pattern replacement - advanced regex not needed for core ops
        if let Some(pos) = name.find(&self.pattern) {
            let mut result = name[..pos].to_string();
            result.push_str(&self.replacement);
            result.push_str(&name[pos + self.pattern.len()..]);
            result
        } else {
            name.to_string()
        }
    }
}

/// Spatial boundary definition for 3D slicing
#[derive(Debug, Clone, Copy)]
pub struct SpatialBoundary {
    pub axis: u8,
    pub start: u32,
    pub end: u32,
}

impl SpatialBoundary {
    #[inline]
    pub const fn new(axis: u8, start: u32, end: u32) -> Self {
        Self { axis: axis.min(2), start, end }
    }

    /// Validate boundary against spatial dimensions
    #[inline]
    pub fn validate(&self, spatial: &SpatialTensorMetadata) -> bool {
        let dim_size = match self.axis {
            0 => spatial.width,
            1 => spatial.height,
            2 => spatial.depth,
            _ => return false,
        };
        self.start < self.end && self.end <= dim_size
    }
}

/// Zero-copy tensor view into underlying data
/// Ollama's materialized tensor concept ka ProNax upgrade with 3D spatial metadata
#[derive(Debug)]
pub struct NeuralTensorView {
    pub descriptor: NeuralTensorDescriptor,
    pub data_offset: usize,
    pub data_length: usize,
    pub spatial_subregion: SpatialSubregion,
    pub memory_layout: NeuralMemoryLayout,
}

/// 3D subregion within parent tensor
#[derive(Debug, Clone, Copy)]
pub struct SpatialSubregion {
    pub x_start: u32,
    pub x_end: u32,
    pub y_start: u32,
    pub y_end: u32,
    pub z_start: u32,
    pub z_end: u32,
    pub guidance_factor: f32,
}

impl SpatialSubregion {
    #[inline]
    pub const fn full(spatial: &SpatialTensorMetadata) -> Self {
        Self {
            x_start: 0,
            x_end: spatial.width,
            y_start: 0,
            y_end: spatial.height,
            z_start: 0,
            z_end: spatial.depth,
            guidance_factor: spatial.guidance_scale as f32 / 100.0,
        }
    }

    #[inline]
    pub const fn width(&self) -> u32 {
        self.x_end - self.x_start
    }

    #[inline]
    pub const fn height(&self) -> u32 {
        self.y_end - self.y_start
    }

    #[inline]
    pub const fn depth(&self) -> u32 {
        self.z_end - self.z_start
    }

    #[inline]
    pub const fn element_count(&self) -> u64 {
        (self.width() as u64) * (self.height() as u64) * (self.depth() as u64)
    }
}

/// Core tensor entity for conversion operations
/// Ollama's Tensor interface ka ProNax implementation with 3D enhancements
#[derive(Debug)]
pub struct NeuralTensorCore {
    pub identifier: String,
    pub data_type: NeuralTensorDataType,
    pub spatial: SpatialTensorMetadata,
    pub raw_data: Vec<u8>,
    pub coordinate: ConversionCoordinate,
}

impl NeuralTensorCore {
    /// Create new tensor core with 3D spatial awareness
    #[inline]
    pub fn new(
        identifier: String,
        data_type: NeuralTensorDataType,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Result<Self, TensorOpError> {
        let spatial = SpatialTensorMetadata::new(width, height, depth);
        let element_count = spatial.volume();
        let byte_size = data_type.element_size_bytes() as u64 * element_count;
        
        if byte_size > usize::MAX as u64 {
            return Err(TensorOpError::TensorTooLarge {
                identifier: identifier.clone(),
                requested_bytes: byte_size,
            });
        }

        Ok(Self {
            identifier,
            data_type,
            spatial,
            raw_data: Vec::with_capacity(byte_size as usize),
            coordinate: ConversionCoordinate::standard(),
        })
    }

    /// Get zero-copy descriptor
    #[inline]
    pub fn descriptor(&self) -> Result<NeuralTensorDescriptor, TensorOpError> {
        NeuralTensorDescriptor::new(
            self.data_type,
            self.spatial.width,
            self.spatial.height,
            self.spatial.depth,
        )
        .map_err(|e| TensorOpError::DescriptorError(e.to_string()))
    }

    /// Clone with new identifier (for split operations)
    #[inline]
    pub fn fork(&self, new_id: String) -> Self {
        Self {
            identifier: new_id,
            data_type: self.data_type,
            spatial: self.spatial,
            raw_data: self.raw_data.clone(),
            coordinate: self.coordinate,
        }
    }
}

/// Spatial tensor splitting operation
/// Ollama's splitDim function ka ProNax 3D rewrite - zero-copy lazy evaluation
/// 
/// # Arguments
/// * `source` - Source tensor to split
/// * `spatial_axis` - Which dimension to split along (0=width, 1=height, 2=depth)
/// * `fragments` - Split configurations for each output tensor
/// 
/// # Returns
/// Iterator of split tensor views - zero-copy until materialization
pub fn spatial_split_stream<'a>(
    source: &'a NeuralTensorCore,
    spatial_axis: u8,
    fragments: &'a [NeuralTensorFragment],
) -> impl Iterator<Item = Result<NeuralSplitResult, TensorOpError>> + 'a {
    let axis = spatial_axis.min(2);
    let dim_size = match axis {
        0 => source.spatial.width,
        1 => source.spatial.height,
        2 => source.spatial.depth,
        _ => 0,
    };

    let even_split_size = dim_size / fragments.len() as u32;
    let mut current_offset = 0u32;

    fragments.iter().map(move |fragment| {
        // Calculate split dimensions
        let split_dim = if fragment.boundary_slices.is_empty() {
            even_split_size
        } else {
            // Find boundary for this axis
            fragment.boundary_slices
                .iter()
                .find(|b| b.axis == axis)
                .map(|b| b.end - b.start)
                .unwrap_or(even_split_size)
        };

        // Build subregion based on axis
        let subregion = match axis {
            0 => SpatialSubregion {
                x_start: current_offset,
                x_end: current_offset + split_dim,
                y_start: 0,
                y_end: source.spatial.height,
                z_start: 0,
                z_end: source.spatial.depth,
                guidance_factor: fragment.guidance_multiplier,
            },
            1 => SpatialSubregion {
                x_start: 0,
                x_end: source.spatial.width,
                y_start: current_offset,
                y_end: current_offset + split_dim,
                z_start: 0,
                z_end: source.spatial.depth,
                guidance_factor: fragment.guidance_multiplier,
            },
            2 => SpatialSubregion {
                x_start: 0,
                x_end: source.spatial.width,
                y_start: 0,
                y_end: source.spatial.height,
                z_start: current_offset,
                z_end: current_offset + split_dim,
                guidance_factor: fragment.guidance_multiplier,
            },
            _ => unreachable!(),
        };

        // Apply name transform
        let result_name = fragment.name_transform
            .as_ref()
            .map(|t| t.apply(&source.identifier))
            .unwrap_or_else(|| format!("{}_split_{}", source.identifier, current_offset));

        // Update offset for next fragment
        current_offset += split_dim;

        // Build memory layout for zero-copy access
        let layout = NeuralMemoryLayout {
            row_stride: source.data_type.element_size_bytes() as u64,
            plane_stride: (source.data_type.element_size_bytes() as u64) * (subregion.width() as u64),
            depth_stride: (source.data_type.element_size_bytes() as u64) 
                * (subregion.width() as u64) 
                * (subregion.height() as u64),
        };

        // Calculate data offsets
        let element_size = source.data_type.element_size_bytes();
        let data_offset = match axis {
            0 => (subregion.x_start as usize) * element_size,
            1 => (subregion.y_start as usize) * (source.spatial.width as usize) * element_size,
            2 => (subregion.z_start as usize) 
                * (source.spatial.width as usize) 
                * (source.spatial.height as usize) 
                * element_size,
            _ => 0,
        };

        let data_length = subregion.element_count() as usize * element_size;

        Ok(NeuralSplitResult {
            identifier: result_name,
            data_type: source.data_type,
            subregion,
            layout,
            data_offset,
            data_length,
            source_guidance: fragment.guidance_multiplier,
        })
    })
}

/// Result of a tensor split operation
#[derive(Debug, Clone)]
pub struct NeuralSplitResult {
    pub identifier: String,
    pub data_type: NeuralTensorDataType,
    pub subregion: SpatialSubregion,
    pub layout: NeuralMemoryLayout,
    pub data_offset: usize,
    pub data_length: usize,
    pub source_guidance: f32,
}

impl NeuralSplitResult {
    /// Materialize into owned tensor core
    #[inline]
    pub fn materialize(&self, parent_data: &[u8]) -> Result<NeuralTensorCore, TensorOpError> {
        if self.data_offset + self.data_length > parent_data.len() {
            return Err(TensorOpError::BufferOverflow {
                requested: self.data_offset + self.data_length,
                available: parent_data.len(),
            });
        }

        let mut core = NeuralTensorCore::new(
            self.identifier.clone(),
            self.data_type,
            self.subregion.width(),
            self.subregion.height(),
            self.subregion.depth(),
        )?;

        // Copy data slice
        core.raw_data.extend_from_slice(&parent_data[self.data_offset..self.data_offset + self.data_length]);
        
        Ok(core)
    }
}

/// Tensor merge configuration
/// Ollama's merge struct ka ProNax 3D upgrade with pattern matching
#[derive(Debug, Clone)]
pub struct NeuralTensorFusion {
    pub match_pattern: String,
    pub output_name: String,
    pub merge_axis: u8,
    pub guidance_blend: f32,
}

impl NeuralTensorFusion {
    #[inline]
    pub fn new(pattern: &str, output: &str, axis: u8) -> Self {
        Self {
            match_pattern: pattern.to_string(),
            output_name: output.to_string(),
            merge_axis: axis.min(2),
            guidance_blend: 1.0,
        }
    }

    /// Check if tensor name matches pattern
    #[inline]
    pub fn matches(&self, name: &str) -> bool {
        // Use glob-style pattern matching
        Self::glob_match(&self.match_pattern, name)
    }

    /// Simple glob pattern matching (* and ? supported)
    fn glob_match(pattern: &str, text: &str) -> bool {
        let mut pattern_chars = pattern.chars().peekable();
        let mut text_chars = text.chars().peekable();

        while let Some(p) = pattern_chars.next() {
            match p {
                '*' => {
                    // Match any sequence
                    if pattern_chars.peek().is_none() {
                        return true; // * at end matches everything
                    }
                    let next_p = *pattern_chars.peek().unwrap();
                    loop {
                        if text_chars.peek() == Some(&next_p) 
                            || Self::glob_match(&pattern_chars.clone().collect::<String>(), 
                                               &text_chars.clone().collect::<String>()) {
                            break;
                        }
                        if text_chars.next().is_none() {
                            return false;
                        }
                    }
                }
                '?' => {
                    if text_chars.next().is_none() {
                        return false;
                    }
                }
                c => {
                    if text_chars.next() != Some(c) {
                        return false;
                    }
                }
            }
        }

        text_chars.next().is_none()
    }
}

/// Spatial tensor merging operation
/// Ollama's mergeTensors function ka ProNax rewrite with 3D spatial concatenation
/// 
/// # Arguments
/// * `candidates` - Unmatched tensors that may be merged
/// * `fusion_configs` - Merge configurations defining patterns and output names
/// 
/// # Returns
/// Tuple of (merged_tensors, remaining_unmatched)
pub fn spatial_tensor_fusion(
    candidates: Vec<NeuralTensorCore>,
    fusion_configs: &[NeuralTensorFusion],
) -> (Vec<NeuralMergedTensor>, Vec<NeuralTensorCore>) {
    let mut merged: Vec<NeuralMergedTensor> = Vec::new();
    let mut unmatched: Vec<NeuralTensorCore> = candidates;

    for config in fusion_configs {
        // Partition: matched vs unmatched
        let mut matched: Vec<NeuralTensorCore> = Vec::new();
        let mut new_unmatched: Vec<NeuralTensorCore> = Vec::new();

        for tensor in unmatched {
            if config.matches(&tensor.identifier) {
                matched.push(tensor);
            } else {
                new_unmatched.push(tensor);
            }
        }
        unmatched = new_unmatched;

        if matched.is_empty() {
            continue;
        }

        // Stable sort by name for deterministic ordering
        matched.sort_by(|a, b| {
            let parts_a: Vec<&str> = a.identifier.split('.').collect();
            let parts_b: Vec<&str> = b.identifier.split('.').collect();

            // Compare by length first
            let len_cmp = parts_a.len().cmp(&parts_b.len());
            if len_cmp != std::cmp::Ordering::Equal {
                return len_cmp;
            }

            // Compare each part - numeric if possible
            for (pa, pb) in parts_a.iter().zip(parts_b.iter()) {
                let cmp = if let (Ok(na), Ok(nb)) = (pa.parse::<u64>(), pb.parse::<u64>()) {
                    na.cmp(&nb)
                } else {
                    pa.cmp(pb)
                };
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }

            std::cmp::Ordering::Equal
        });

        // Build merged tensor with 3D spatial expansion along merge axis
        let first = &matched[0];
        let merge_count = matched.len() as u32;

        let merged_spatial = match config.merge_axis {
            0 => SpatialTensorMetadata::new(
                first.spatial.width * merge_count,
                first.spatial.height,
                first.spatial.depth,
            ),
            1 => SpatialTensorMetadata::new(
                first.spatial.width,
                first.spatial.height * merge_count,
                first.spatial.depth,
            ),
            2 => SpatialTensorMetadata::new(
                first.spatial.width,
                first.spatial.height,
                first.spatial.depth * merge_count,
            ),
            _ => first.spatial,
        };

        merged.push(NeuralMergedTensor {
            identifier: config.output_name.clone(),
            data_type: first.data_type,
            spatial: merged_spatial,
            sources: matched,
            merge_axis: config.merge_axis,
            blended_guidance: config.guidance_blend,
        });
    }

    (merged, unmatched)
}

/// Merged tensor from multiple sources
#[derive(Debug)]
pub struct NeuralMergedTensor {
    pub identifier: String,
    pub data_type: NeuralTensorDataType,
    pub spatial: SpatialTensorMetadata,
    pub sources: Vec<NeuralTensorCore>,
    pub merge_axis: u8,
    pub blended_guidance: f32,
}

impl NeuralMergedTensor {
    /// Write merged tensor data to output
    /// Zero-copy where possible, streaming for large tensors
    pub fn write_streaming<W: Write>(&self, writer: &mut W) -> Result<u64, TensorOpError> {
        let mut total_written: u64 = 0;

        // Write source tensors in order along merge axis
        for source in &self.sources {
            total_written += Self::write_tensor_data(writer, source)?;
        }

        Ok(total_written)
    }

    fn write_tensor_data<W: Write>(writer: &mut W, tensor: &NeuralTensorCore) -> Result<u64, TensorOpError> {
        let bytes = tensor.raw_data.len() as u64;
        writer.write_all(&tensor.raw_data)
            .map_err(|e| TensorOpError::IoError(e))?;
        Ok(bytes)
    }

    /// Materialize into single tensor core
    pub fn materialize(&self) -> Result<NeuralTensorCore, TensorOpError> {
        let mut core = NeuralTensorCore::new(
            self.identifier.clone(),
            self.data_type,
            self.spatial.width,
            self.spatial.height,
            self.spatial.depth,
        )?;

        // Pre-allocate capacity
        let total_bytes: usize = self.sources.iter()
            .map(|s| s.raw_data.len())
            .sum();
        core.raw_data.reserve(total_bytes);

        // Append all source data
        for source in &self.sources {
            core.raw_data.extend_from_slice(&source.raw_data);
        }

        Ok(core)
    }
}

impl Write for NeuralMergedTensor {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Delegate to streaming write
        let mut cursor = io::Cursor::new(buf);
        match self.write_streaming(&mut cursor) {
            Ok(n) => Ok(n as usize),
            Err(_) => Err(io::Error::new(io::ErrorKind::Other, "Tensor write failed")),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Tensor operation errors
#[derive(Debug, Clone)]
pub enum TensorOpError {
    TensorTooLarge { identifier: String, requested_bytes: u64 },
    BufferOverflow { requested: usize, available: usize },
    InvalidSpatialAxis { axis: u8, max: u8 },
    DescriptorError(String),
    BoundaryError { boundary: SpatialBoundary, reason: String },
    IoError(String),
    MergeError { reason: String },
}

impl std::fmt::Display for TensorOpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TensorTooLarge { identifier, requested_bytes } => {
                write!(f, "Tensor '{}' too large: {} bytes exceeds limit", identifier, requested_bytes)
            }
            Self::BufferOverflow { requested, available } => {
                write!(f, "Buffer overflow: requested {} bytes, {} available", requested, available)
            }
            Self::InvalidSpatialAxis { axis, max } => {
                write!(f, "Invalid spatial axis {} (max {})", axis, max)
            }
            Self::DescriptorError(msg) => {
                write!(f, "Descriptor error: {}", msg)
            }
            Self::BoundaryError { boundary, reason } => {
                write!(f, "Boundary error {:?}: {}", boundary, reason)
            }
            Self::IoError(msg) => {
                write!(f, "I/O error: {}", msg)
            }
            Self::MergeError { reason } => {
                write!(f, "Merge error: {}", reason)
            }
        }
    }
}

impl std::error::Error for TensorOpError {}

impl From<io::Error> for TensorOpError {
    fn from(e: io::Error) -> Self {
        Self::IoError(e.to_string())
    }
}

/// Zero-copy tensor buffer view
/// Direct memory access without allocation
pub struct NeuralBufferView<'a> {
    data: &'a [u8],
    layout: NeuralMemoryLayout,
    spatial: SpatialTensorMetadata,
}

impl<'a> NeuralBufferView<'a> {
    #[inline]
    pub const fn new(data: &'a [u8], layout: NeuralMemoryLayout, spatial: SpatialTensorMetadata) -> Self {
        Self { data, layout, spatial }
    }

    /// Get element at 3D coordinates - zero-copy direct access
    #[inline]
    pub fn get_at(&self, x: u32, y: u32, z: u32) -> Option<&'a [u8]> {
        if x >= self.spatial.width || y >= self.spatial.height || z >= self.spatial.depth {
            return None;
        }

        let offset = (z as u64 * self.layout.depth_stride)
            + (y as u64 * self.layout.plane_stride)
            + (x as u64 * self.layout.row_stride);

        let start = offset as usize;
        let end = start + self.layout.row_stride as usize;
        
        if end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    /// Create subregion view - zero-copy
    #[inline]
    pub fn subregion(&self, region: SpatialSubregion) -> Option<NeuralBufferView<'a>> {
        if region.x_end > self.spatial.width 
            || region.y_end > self.spatial.height 
            || region.z_end > self.spatial.depth {
            return None;
        }

        let offset = (region.z_start as u64 * self.layout.depth_stride)
            + (region.y_start as u64 * self.layout.plane_stride)
            + (region.x_start as u64 * self.layout.row_stride);

        let start = offset as usize;
        let sub_spatial = SpatialTensorMetadata::new(
            region.width(),
            region.height(),
            region.depth(),
        );

        Some(NeuralBufferView {
            data: &self.data[start..],
            layout: self.layout,
            spatial: sub_spatial,
        })
    }
}

/// Advanced 3D tensor operations module
pub mod spatial_ops {
    use super::*;

    /// Calculate 3D convolution output dimensions
    #[inline]
    pub const fn conv_output_dim(
        input_size: u32,
        kernel_size: u32,
        stride: u32,
        padding: u32,
    ) -> u32 {
        if stride == 0 {
            return 0;
        }
        (input_size + 2 * padding - kernel_size) / stride + 1
    }

    /// Calculate 3D pooling output dimensions  
    #[inline]
    pub const fn pool_output_dim(input_size: u32, pool_size: u32, stride: u32) -> u32 {
        if stride == 0 {
            return 0;
        }
        (input_size - pool_size) / stride + 1
    }

    /// 3D tensor transpose operation planning
    #[inline]
    pub fn plan_transpose_3d(
        src_layout: &NeuralMemoryLayout,
        src_spatial: &SpatialTensorMetadata,
        axis_order: [u8; 3],
    ) -> Result<NeuralMemoryLayout, TensorOpError> {
        // Validate axis order
        let mut seen = [false; 3];
        for &axis in &axis_order {
            if axis > 2 {
                return Err(TensorOpError::InvalidSpatialAxis { axis, max: 2 });
            }
            if seen[axis as usize] {
                return Err(TensorOpError::InvalidSpatialAxis { axis, max: 2 });
            }
            seen[axis as usize] = true;
        }

        let dims = [src_spatial.width, src_spatial.height, src_spatial.depth];
        let new_dims = [dims[axis_order[0] as usize], dims[axis_order[1] as usize], dims[axis_order[2] as usize]];

        let new_spatial = SpatialTensorMetadata::new(new_dims[0], new_dims[1], new_dims[2]);
        Ok(new_spatial.as_neural_layout())
    }
}

/// Tensor conversion utilities
pub mod convert_utils {
    use super::*;

    /// Split iterator adapter for parallel processing
    pub struct ParallelSplitIterator<'a> {
        source: &'a NeuralTensorCore,
        fragments: &'a [NeuralTensorFragment],
        current: usize,
    }

    impl<'a> ParallelSplitIterator<'a> {
        pub fn new(source: &'a NeuralTensorCore, fragments: &'a [NeuralTensorFragment]) -> Self {
            Self { source, fragments, current: 0 }
        }
    }

    impl<'a> Iterator for ParallelSplitIterator<'a> {
        type Item = Result<NeuralSplitResult, TensorOpError>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current >= self.fragments.len() {
                return None;
            }
            
            // Yield next split result
            let result = spatial_split_stream(self.source, 0, &self.fragments[self.current..self.current+1])
                .next()?;
            
            self.current += 1;
            Some(result)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let remaining = self.fragments.len() - self.current;
            (remaining, Some(remaining))
        }
    }

    impl<'a> ExactSizeIterator for ParallelSplitIterator<'a> {}

    /// Batch merge multiple fusion configs
    #[inline]
    pub fn batch_fusion(
        tensors: Vec<NeuralTensorCore>,
        configs: &[NeuralTensorFusion],
    ) -> (Vec<NeuralMergedTensor>, Vec<NeuralTensorCore>) {
        spatial_tensor_fusion(tensors, configs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_transform() {
        let transform = NameTransform {
            pattern: "layer".to_string(),
            replacement: "L".to_string(),
        };
        assert_eq!(transform.apply("model.layer.0.weight"), "model.L.0.weight");
    }

    #[test]
    fn test_spatial_boundary() {
        let spatial = SpatialTensorMetadata::new(1024, 768, 256);
        let boundary = SpatialBoundary::new(0, 0, 512);
        assert!(boundary.validate(&spatial));
        
        let bad_boundary = SpatialBoundary::new(0, 500, 2000);
        assert!(!bad_boundary.validate(&spatial));
    }

    #[test]
    fn test_tensor_core_creation() {
        let core = NeuralTensorCore::new(
            "test_tensor".to_string(),
            NeuralTensorDataType::Float32,
            256,
            256,
            64,
        ).unwrap();
        
        assert_eq!(core.spatial.volume(), 256 * 256 * 64);
        assert_eq!(core.identifier, "test_tensor");
    }

    #[test]
    fn test_glob_matching() {
        assert!(NeuralTensorFusion::glob_match("model.*.weight", "model.layer.0.weight"));
        assert!(NeuralTensorFusion::glob_match("*.weight", "attention.weight"));
        assert!(!NeuralTensorFusion::glob_match("*.bias", "attention.weight"));
    }

    #[test]
    fn test_spatial_split_stream() {
        let core = NeuralTensorCore::new(
            "input".to_string(),
            NeuralTensorDataType::Float32,
            1024,
            1,
            1,
        ).unwrap();

        let fragments = vec![
            NeuralTensorFragment::axis_aligned(0, 1.0),
            NeuralTensorFragment::axis_aligned(0, 1.0),
        ];

        let results: Vec<_> = spatial_split_stream(&core, 0, &fragments).collect();
        assert_eq!(results.len(), 2);
        
        // Each should have width 512
        for result in &results {
            assert!(result.is_ok());
            let r = result.as_ref().unwrap();
            assert_eq!(r.subregion.width(), 512);
        }
    }

    #[test]
    fn test_tensor_fusion() {
        let tensors = vec![
            NeuralTensorCore::new("layer.0.weight".to_string(), NeuralTensorDataType::Float32, 256, 256, 1).unwrap(),
            NeuralTensorCore::new("layer.1.weight".to_string(), NeuralTensorDataType::Float32, 256, 256, 1).unwrap(),
            NeuralTensorCore::new("other.tensor".to_string(), NeuralTensorDataType::Float32, 100, 100, 1).unwrap(),
        ];

        let configs = vec![
            NeuralTensorFusion::new("layer.*.weight", "merged_weights", 0),
        ];

        let (merged, unmatched) = spatial_tensor_fusion(tensors, &configs);
        
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].sources.len(), 2);
        assert_eq!(unmatched.len(), 1);
    }

    #[test]
    fn test_buffer_view_access() {
        let data = vec![0u8; 1024 * 4]; // 1024 floats
        let spatial = SpatialTensorMetadata::new(16, 16, 1);
        let layout = spatial.as_neural_layout();
        let view = NeuralBufferView::new(&data, layout, spatial);

        assert!(view.get_at(0, 0, 0).is_some());
        assert!(view.get_at(15, 15, 0).is_some());
        assert!(view.get_at(16, 16, 0).is_none()); // Out of bounds
    }

    #[test]
    fn test_spatial_conv_output() {
        use spatial_ops::*;
        
        // 224x224 input, 3x3 kernel, stride 2, padding 1
        let out = conv_output_dim(224, 3, 2, 1);
        assert_eq!(out, 112);
    }
}
