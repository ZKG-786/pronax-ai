use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// Gemma3 image processor errors
#[derive(Debug, Clone)]
pub enum Gemma3ImageProcError {
    InvalidImage(String),
    ResizeError(String),
    NormalizeError(String),
    ChannelError(String),
    DimensionError(String),
}

impl std::fmt::Display for Gemma3ImageProcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidImage(s) => write!(f, "Invalid image: {}", s),
            Self::ResizeError(s) => write!(f, "Resize error: {}", s),
            Self::NormalizeError(s) => write!(f, "Normalize error: {}", s),
            Self::ChannelError(s) => write!(f, "Channel error: {}", s),
            Self::DimensionError(s) => write!(f, "Dimension error: {}", s),
        }
    }
}

impl std::error::Error for Gemma3ImageProcError {}

/// ImageNet standard normalization constants
pub struct ImageNetNormalization;

impl ImageNetNormalization {
    /// Standard mean for RGB channels
    pub const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    /// Standard std for RGB channels
    pub const STD: [f32; 3] = [0.229, 0.224, 0.225];
    /// Alternative mean for some vision models
    pub const ALTERNATE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
    /// Alternative std for some vision models
    pub const ALTERNATE_STD: [f32; 3] = [0.5, 0.5, 0.5];
}

/// 3D-aware pixel coordinate
#[derive(Debug, Clone, Copy)]
pub struct PixelCoordinate3D {
    pub x: usize,
    pub y: usize,
    pub channel: u8,
    pub depth_value: f32,
}

impl PixelCoordinate3D {
    pub fn new(x: usize, y: usize, channel: u8) -> Self {
        Self {
            x,
            y,
            channel,
            depth_value: 1.0,
        }
    }
}

/// 3D-aware image tensor representation
#[derive(Debug, Clone)]
pub struct SpatialImageTensor {
    /// Width of image
    pub width: usize,
    /// Height of image
    pub height: usize,
    /// Number of channels (typically 3 for RGB)
    pub channels: usize,
    /// Raw pixel data in CHW format (channels, height, width)
    pub data: Vec<f32>,
    /// 3D spatial metadata
    pub spatial: SpatialTensorMetadata,
    /// Spatial position in processing pipeline
    pub spatial_position: ConversionCoordinate,
}

impl SpatialImageTensor {
    /// Create new image tensor from dimensions
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self {
            width,
            height,
            channels,
            data: vec![0.0; width * height * channels],
            spatial: SpatialTensorMetadata::new(width as u32, height as u32, channels as u32),
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Create from raw RGB data (HWC format)
    pub fn from_rgb_hwc(pixels: &[u8], width: usize, height: usize) -> Self {
        let mut tensor = Self::new(width, height, 3);
        
        for y in 0..height {
            for x in 0..width {
                let pixel_idx = (y * width + x) * 3;
                if pixel_idx + 2 < pixels.len() {
                    // Store in CHW format
                    tensor.data[0 * width * height + y * width + x] = pixels[pixel_idx] as f32;
                    tensor.data[1 * width * height + y * width + x] = pixels[pixel_idx + 1] as f32;
                    tensor.data[2 * width * height + y * width + x] = pixels[pixel_idx + 2] as f32;
                }
            }
        }
        
        tensor
    }
    
    /// Get pixel value at coordinate
    pub fn get_pixel(&self, x: usize, y: usize, channel: usize) -> f32 {
        if x >= self.width || y >= self.height || channel >= self.channels {
            return 0.0;
        }
        self.data[channel * self.width * self.height + y * self.width + x]
    }
    
    /// Set pixel value at coordinate
    pub fn set_pixel(&mut self, x: usize, y: usize, channel: usize, value: f32) {
        if x < self.width && y < self.height && channel < self.channels {
            self.data[channel * self.width * self.height + y * self.width + x] = value;
        }
    }
    
    /// Convert to planar format (HWC -> CHW is already done in storage)
    pub fn to_hwc(&self) -> Vec<f32> {
        let mut hwc = vec![0.0; self.width * self.height * self.channels];
        
        for y in 0..self.height {
            for x in 0..self.width {
                for c in 0..self.channels {
                    let chw_idx = c * self.width * self.height + y * self.width + x;
                    let hwc_idx = (y * self.width + x) * self.channels + c;
                    hwc[hwc_idx] = self.data[chw_idx];
                }
            }
        }
        
        hwc
    }
}

/// Resize interpolation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeInterpolation {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos,
}

/// 3D-aware image resizer
pub struct SpatialImageResizer3D;

impl SpatialImageResizer3D {
    /// Resize image using bilinear interpolation
    pub fn resize(
        source: &SpatialImageTensor,
        target_width: usize,
        target_height: usize,
        method: ResizeInterpolation,
    ) -> Result<SpatialImageTensor, Gemma3ImageProcError> {
        match method {
            ResizeInterpolation::Bilinear => {
                Self::resize_bilinear(source, target_width, target_height)
            }
            ResizeInterpolation::Nearest => {
                Self::resize_nearest(source, target_width, target_height)
            }
            _ => Err(Gemma3ImageProcError::ResizeError(
                "Unsupported resize method".to_string()
            )),
        }
    }
    
    /// Bilinear resize implementation
    fn resize_bilinear(
        source: &SpatialImageTensor,
        target_width: usize,
        target_height: usize,
    ) -> Result<SpatialImageTensor, Gemma3ImageProcError> {
        let mut result = SpatialImageTensor::new(target_width, target_height, source.channels);
        
        let x_ratio = source.width as f32 / target_width as f32;
        let y_ratio = source.height as f32 / target_height as f32;
        
        for y in 0..target_height {
            for x in 0..target_width {
                let src_x = x as f32 * x_ratio;
                let src_y = y as f32 * y_ratio;
                
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = (x0 + 1).min(source.width - 1);
                let y1 = (y0 + 1).min(source.height - 1);
                
                let dx = src_x - x0 as f32;
                let dy = src_y - y0 as f32;
                
                for c in 0..source.channels {
                    let v00 = source.get_pixel(x0, y0, c);
                    let v01 = source.get_pixel(x1, y0, c);
                    let v10 = source.get_pixel(x0, y1, c);
                    let v11 = source.get_pixel(x1, y1, c);
                    
                    let v0 = v00 * (1.0 - dx) + v01 * dx;
                    let v1 = v10 * (1.0 - dx) + v11 * dx;
                    let v = v0 * (1.0 - dy) + v1 * dy;
                    
                    result.set_pixel(x, y, c, v);
                }
            }
        }
        
        Ok(result)
    }
    
    /// Nearest neighbor resize
    fn resize_nearest(
        source: &SpatialImageTensor,
        target_width: usize,
        target_height: usize,
    ) -> Result<SpatialImageTensor, Gemma3ImageProcError> {
        let mut result = SpatialImageTensor::new(target_width, target_height, source.channels);
        
        let x_ratio = source.width as f32 / target_width as f32;
        let y_ratio = source.height as f32 / target_height as f32;
        
        for y in 0..target_height {
            for x in 0..target_width {
                let src_x = (x as f32 * x_ratio) as usize;
                let src_y = (y as f32 * y_ratio) as usize;
                
                for c in 0..source.channels {
                    let v = source.get_pixel(src_x.min(source.width - 1), src_y.min(source.height - 1), c);
                    result.set_pixel(x, y, c, v);
                }
            }
        }
        
        Ok(result)
    }
}

/// 3D-aware image normalizer
pub struct SpatialImageNormalizer3D;

impl SpatialImageNormalizer3D {
    /// Apply ImageNet standard normalization
    pub fn normalize_imagenet(tensor: &mut SpatialImageTensor) {
        Self::normalize_with_params(tensor, &ImageNetNormalization::MEAN, &ImageNetNormalization::STD);
    }
    
    /// Apply alternative normalization (0.5 mean, 0.5 std)
    pub fn normalize_alternate(tensor: &mut SpatialImageTensor) {
        Self::normalize_with_params(tensor, &ImageNetNormalization::ALTERNATE_MEAN, &ImageNetNormalization::ALTERNATE_STD);
    }
    
    /// Normalize with custom parameters
    pub fn normalize_with_params(tensor: &mut SpatialImageTensor, mean: &[f32; 3], std: &[f32; 3]) {
        for y in 0..tensor.height {
            for x in 0..tensor.width {
                for c in 0..tensor.channels.min(3) {
                    let value = tensor.get_pixel(x, y, c);
                    let normalized = (value / 255.0 - mean[c]) / std[c];
                    tensor.set_pixel(x, y, c, normalized);
                }
            }
        }
    }
    
    /// Normalize to [-1, 1] range
    pub fn normalize_to_range(tensor: &mut SpatialImageTensor, min: f32, max: f32) {
        for v in &mut tensor.data {
            *v = *v / 255.0 * (max - min) + min;
        }
    }
    
    /// Denormalize from ImageNet
    pub fn denormalize_imagenet(tensor: &mut SpatialImageTensor) {
        for y in 0..tensor.height {
            for x in 0..tensor.width {
                for c in 0..tensor.channels.min(3) {
                    let value = tensor.get_pixel(x, y, c);
                    let denormalized = (value * ImageNetNormalization::STD[c] + ImageNetNormalization::MEAN[c]) * 255.0;
                    tensor.set_pixel(x, y, c, denormalized.clamp(0.0, 255.0));
                }
            }
        }
    }
}

/// 3D-aware image compositor for transparency handling
pub struct SpatialImageCompositor3D;

impl SpatialImageCompositor3D {
    /// Composite image onto white background (handle transparency)
    pub fn composite_onto_white(rgba_data: &[u8], width: usize, height: usize) -> SpatialImageTensor {
        let mut result = SpatialImageTensor::new(width, height, 3);
        
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4; // RGBA
                if idx + 3 < rgba_data.len() {
                    let r = rgba_data[idx] as f32;
                    let g = rgba_data[idx + 1] as f32;
                    let b = rgba_data[idx + 2] as f32;
                    let a = rgba_data[idx + 3] as f32 / 255.0;
                    
                    // Composite onto white background
                    let bg = 255.0;
                    let comp_r = r * a + bg * (1.0 - a);
                    let comp_g = g * a + bg * (1.0 - a);
                    let comp_b = b * a + bg * (1.0 - a);
                    
                    result.set_pixel(x, y, 0, comp_r);
                    result.set_pixel(x, y, 1, comp_g);
                    result.set_pixel(x, y, 2, comp_b);
                }
            }
        }
        
        result
    }
    
    /// Composite with custom background color
    pub fn composite_with_background(
        rgba_data: &[u8],
        width: usize,
        height: usize,
        bg_r: f32,
        bg_g: f32,
        bg_b: f32,
    ) -> SpatialImageTensor {
        let mut result = SpatialImageTensor::new(width, height, 3);
        
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4; // RGBA
                if idx + 3 < rgba_data.len() {
                    let r = rgba_data[idx] as f32;
                    let g = rgba_data[idx + 1] as f32;
                    let b = rgba_data[idx + 2] as f32;
                    let a = rgba_data[idx + 3] as f32 / 255.0;
                    
                    let comp_r = r * a + bg_r * (1.0 - a);
                    let comp_g = g * a + bg_g * (1.0 - a);
                    let comp_b = b * a + bg_b * (1.0 - a);
                    
                    result.set_pixel(x, y, 0, comp_r);
                    result.set_pixel(x, y, 1, comp_g);
                    result.set_pixel(x, y, 2, comp_b);
                }
            }
        }
        
        result
    }
}

/// 3D-aware Gemma3 image processor
#[derive(Debug, Clone)]
pub struct Gemma3ImageProcessor3D {
    /// Target image size
    pub image_size: usize,
    /// Patch size for vision model
    pub patch_size: usize,
    /// Number of channels
    pub num_channels: usize,
    /// Normalization method
    pub normalization: ImageNormalizationType,
    /// 3D spatial position
    pub spatial_position: ConversionCoordinate,
}

/// Normalization type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageNormalizationType {
    ImageNetStandard,
    Alternate,
    None,
}

impl Gemma3ImageProcessor3D {
    /// Create new image processor with default ImageNet normalization
    pub fn new(image_size: usize, patch_size: usize, num_channels: usize) -> Self {
        Self {
            image_size,
            patch_size,
            num_channels,
            normalization: ImageNormalizationType::ImageNetStandard,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Create with specific normalization
    pub fn with_normalization(
        image_size: usize,
        patch_size: usize,
        num_channels: usize,
        normalization: ImageNormalizationType,
    ) -> Self {
        Self {
            image_size,
            patch_size,
            num_channels,
            normalization,
            spatial_position: ConversionCoordinate::standard(),
        }
    }
    
    /// Process raw RGB image data
    pub fn process_rgb(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, Gemma3ImageProcError> {
        // Create tensor from raw data
        let mut tensor = SpatialImageTensor::from_rgb_hwc(rgb_data, width, height);
        
        // Resize if needed
        if width != self.image_size || height != self.image_size {
            tensor = SpatialImageResizer3D::resize(
                &tensor,
                self.image_size,
                self.image_size,
                ResizeInterpolation::Bilinear,
            )?;
        }
        
        // Apply normalization
        match self.normalization {
            ImageNormalizationType::ImageNetStandard => {
                SpatialImageNormalizer3D::normalize_imagenet(&mut tensor);
            }
            ImageNormalizationType::Alternate => {
                SpatialImageNormalizer3D::normalize_alternate(&mut tensor);
            }
            ImageNormalizationType::None => {
                // Just scale to [0, 1]
                for v in &mut tensor.data {
                    *v /= 255.0;
                }
            }
        }
        
        // Convert to CHW format (already in CHW, return raw data)
        Ok(tensor.data)
    }
    
    /// Process RGBA image with compositing
    pub fn process_rgba(
        &self,
        rgba_data: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, Gemma3ImageProcError> {
        // Composite onto white background
        let mut tensor = SpatialImageCompositor3D::composite_onto_white(rgba_data, width, height);
        
        // Resize if needed
        if width != self.image_size || height != self.image_size {
            tensor = SpatialImageResizer3D::resize(
                &tensor,
                self.image_size,
                self.image_size,
                ResizeInterpolation::Bilinear,
            )?;
        }
        
        // Apply normalization
        match self.normalization {
            ImageNormalizationType::ImageNetStandard => {
                SpatialImageNormalizer3D::normalize_imagenet(&mut tensor);
            }
            ImageNormalizationType::Alternate => {
                SpatialImageNormalizer3D::normalize_alternate(&mut tensor);
            }
            ImageNormalizationType::None => {
                for v in &mut tensor.data {
                    *v /= 255.0;
                }
            }
        }
        
        Ok(tensor.data)
    }
    
    /// Pack image tensor to flat vector in CHW format
    pub fn pack_to_chw(&self, tensor: &SpatialImageTensor) -> Vec<f32> {
        // Data is already in CHW format, return clone
        tensor.data.clone()
    }
    
    /// Pack image tensor to flat vector in HWC format
    pub fn pack_to_hwc(&self, tensor: &SpatialImageTensor) -> Vec<f32> {
        tensor.to_hwc()
    }
    
    /// Get number of patches for this processor config
    pub fn num_patches(&self) -> usize {
        let patches_per_dim = self.image_size / self.patch_size;
        patches_per_dim * patches_per_dim
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), Gemma3ImageProcError> {
        if self.image_size % self.patch_size != 0 {
            return Err(Gemma3ImageProcError::DimensionError(
                format!("image_size {} not divisible by patch_size {}",
                    self.image_size, self.patch_size)
            ));
        }
        
        if self.num_channels != 3 {
            return Err(Gemma3ImageProcError::ChannelError(
                format!("Expected 3 channels (RGB), got {}", self.num_channels)
            ));
        }
        
        Ok(())
    }
}

/// Utility functions for image processing
pub mod gemma3_image_utils {
    use super::*;
    
    /// Calculate image dimensions from byte length (assuming RGB)
    pub fn calculate_dimensions(rgb_bytes: usize) -> Option<(usize, usize)> {
        let pixels = rgb_bytes / 3;
        let sqrt = (pixels as f64).sqrt() as usize;
        
        if sqrt * sqrt * 3 == rgb_bytes {
            Some((sqrt, sqrt))
        } else {
            None
        }
    }
    
    /// Create a checkerboard pattern image (for testing)
    pub fn create_checkerboard(size: usize, channels: usize) -> SpatialImageTensor {
        let mut tensor = SpatialImageTensor::new(size, size, channels);
        
        for y in 0..size {
            for x in 0..size {
                let is_white = (x / 10 + y / 10) % 2 == 0;
                let value = if is_white { 255.0 } else { 0.0 };
                
                for c in 0..channels {
                    tensor.set_pixel(x, y, c, value);
                }
            }
        }
        
        tensor
    }
    
    /// Create a gradient image (for testing)
    pub fn create_gradient(size: usize, channels: usize) -> SpatialImageTensor {
        let mut tensor = SpatialImageTensor::new(size, size, channels);
        
        for y in 0..size {
            for x in 0..size {
                let value = (x + y) as f32 / (2.0 * size as f32) * 255.0;
                
                for c in 0..channels {
                    tensor.set_pixel(x, y, c, value);
                }
            }
        }
        
        tensor
    }
    
    /// Verify image dimensions are valid for vision model
    pub fn verify_dimensions(width: usize, height: usize, patch_size: usize) -> bool {
        width % patch_size == 0 && height % patch_size == 0
    }
    
    /// Calculate memory requirement for processed image
    pub fn estimate_memory(image_size: usize, num_channels: usize) -> usize {
        image_size * image_size * num_channels * 4 // f32 = 4 bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_image_tensor_creation() {
        let tensor = SpatialImageTensor::new(224, 224, 3);
        assert_eq!(tensor.width, 224);
        assert_eq!(tensor.height, 224);
        assert_eq!(tensor.channels, 3);
        assert_eq!(tensor.data.len(), 224 * 224 * 3);
    }
    
    #[test]
    fn test_from_rgb_hwc() {
        let rgb = vec![255u8; 224 * 224 * 3];
        let tensor = SpatialImageTensor::from_rgb_hwc(&rgb, 224, 224);
        
        assert_eq!(tensor.data[0], 255.0); // First pixel, R channel
        assert_eq!(tensor.get_pixel(0, 0, 0), 255.0);
    }
    
    #[test]
    fn test_pixel_operations() {
        let mut tensor = SpatialImageTensor::new(10, 10, 3);
        
        tensor.set_pixel(5, 5, 1, 128.0);
        assert_eq!(tensor.get_pixel(5, 5, 1), 128.0);
        
        // Out of bounds should return 0
        assert_eq!(tensor.get_pixel(20, 20, 0), 0.0);
    }
    
    #[test]
    fn test_resize_bilinear() {
        let tensor = SpatialImageTensor::new(100, 100, 3);
        let resized = SpatialImageResizer3D::resize(&tensor, 50, 50, ResizeInterpolation::Bilinear);
        
        assert!(resized.is_ok());
        let result = resized.unwrap();
        assert_eq!(result.width, 50);
        assert_eq!(result.height, 50);
    }
    
    #[test]
    fn test_resize_nearest() {
        let tensor = SpatialImageTensor::new(100, 100, 3);
        let resized = SpatialImageResizer3D::resize(&tensor, 50, 50, ResizeInterpolation::Nearest);
        
        assert!(resized.is_ok());
        let result = resized.unwrap();
        assert_eq!(result.width, 50);
        assert_eq!(result.height, 50);
    }
    
    #[test]
    fn test_normalization() {
        let rgb = vec![255u8; 224 * 224 * 3];
        let mut tensor = SpatialImageTensor::from_rgb_hwc(&rgb, 224, 224);
        
        SpatialImageNormalizer3D::normalize_imagenet(&mut tensor);
        
        // After normalization, max value should be around (1 - mean) / std
        let max_val = tensor.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_val > 0.0);
    }
    
    #[test]
    fn test_compositor() {
        let rgba = vec![255u8, 0, 0, 128, 0, 255, 0, 128]; // 2 pixels, half transparent
        let tensor = SpatialImageCompositor3D::composite_onto_white(&rgba, 2, 1);
        
        assert_eq!(tensor.channels, 3);
        // Red pixel with 50% transparency on white background
        assert!(tensor.get_pixel(0, 0, 0) > 127.0 && tensor.get_pixel(0, 0, 0) < 255.0);
    }
    
    #[test]
    fn test_image_processor() {
        let processor = Gemma3ImageProcessor3D::new(896, 14, 3);
        
        assert_eq!(processor.num_patches(), 4096); // (896/14)^2
        assert!(processor.validate().is_ok());
    }
    
    #[test]
    fn test_processor_validation() {
        let processor = Gemma3ImageProcessor3D::new(100, 14, 3);
        assert!(processor.validate().is_err()); // 100 % 14 != 0
    }
    
    #[test]
    fn test_process_rgb() {
        let processor = Gemma3ImageProcessor3D::new(224, 14, 3);
        let rgb = vec![128u8; 224 * 224 * 3];
        
        let result = processor.process_rgb(&rgb, 224, 224);
        assert!(result.is_ok());
        
        let data = result.unwrap();
        assert_eq!(data.len(), 224 * 224 * 3);
    }
    
    #[test]
    fn test_process_rgba() {
        let processor = Gemma3ImageProcessor3D::new(224, 14, 3);
        let rgba = vec![128u8, 128, 128, 255; 224 * 224 * 4];
        
        let result = processor.process_rgba(&rgba, 224, 224);
        assert!(result.is_ok());
        
        let data = result.unwrap();
        assert_eq!(data.len(), 224 * 224 * 3);
    }
    
    #[test]
    fn test_hwc_conversion() {
        let tensor = SpatialImageTensor::new(10, 10, 3);
        let hwc = tensor.to_hwc();
        
        assert_eq!(hwc.len(), 300); // 10 * 10 * 3
    }
    
    #[test]
    fn test_utils() {
        let dims = gemma3_image_utils::calculate_dimensions(224 * 224 * 3);
        assert_eq!(dims, Some((224, 224)));
        
        let checkerboard = gemma3_image_utils::create_checkerboard(100, 3);
        assert_eq!(checkerboard.width, 100);
        assert_eq!(checkerboard.height, 100);
        
        let valid = gemma3_image_utils::verify_dimensions(896, 896, 14);
        assert!(valid);
        
        let invalid = gemma3_image_utils::verify_dimensions(100, 100, 14);
        assert!(!invalid);
        
        let mem = gemma3_image_utils::estimate_memory(896, 3);
        assert_eq!(mem, 896 * 896 * 3 * 4);
    }
}
