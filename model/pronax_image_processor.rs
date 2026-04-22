use std::io::Cursor;

use image::{imageops, DynamicImage, ImageBuffer, Rgba};
use thiserror::Error;

use crate::convert::pronax_converter_core::ConversionCoordinate;
use crate::fs::ggml::pronax_ggml_types::SpatialTensorMetadata;

/// Image processing errors
#[derive(Error, Debug, Clone)]
pub enum ImageProcError {
    #[error("Invalid image format: {0}")]
    InvalidFormat(String),
    
    #[error("Resize method not supported: {0}")]
    UnsupportedResizeMethod(String),
    
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
    
    #[error("Normalization error: {0}")]
    NormalizationError(String),
    
    #[error("IO error: {0}")]
    IoError(String),
}

/// Normalization constants for different model families
pub mod norm_constants {
    /// ImageNet standard normalization (ResNet, etc.)
    pub const IMAGENET_DEFAULT_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    pub const IMAGENET_DEFAULT_STD: [f32; 3] = [0.229, 0.224, 0.225];
    
    /// ImageNet standard [0.5, 0.5, 0.5] normalization
    pub const IMAGENET_STANDARD_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
    pub const IMAGENET_STANDARD_STD: [f32; 3] = [0.5, 0.5, 0.5];
    
    /// CLIP model normalization
    pub const CLIP_DEFAULT_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    pub const CLIP_DEFAULT_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
    
    /// 3D spatial normalization for vision transformers
    pub const VISION_3D_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
    pub const VISION_3D_STD: [f32; 3] = [0.25, 0.25, 0.25];
}

/// Image resize interpolation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMethod {
    /// Bilinear interpolation
    Bilinear,
    /// Nearest neighbor (fastest)
    NearestNeighbor,
    /// Approximate bilinear (faster than exact)
    ApproxBilinear,
    /// Catmull-Rom spline (higher quality)
    CatmullRom,
    /// Lanczos3 (highest quality)
    Lanczos3,
}

impl ResizeMethod {
    /// Convert to imageops filter type
    pub fn to_filter(&self) -> imageops::FilterType {
        match self {
            Self::Bilinear => imageops::FilterType::Triangle,
            Self::NearestNeighbor => imageops::FilterType::Nearest,
            Self::ApproxBilinear => imageops::FilterType::Triangle,
            Self::CatmullRom => imageops::FilterType::CatmullRom,
            Self::Lanczos3 => imageops::FilterType::Lanczos3,
        }
    }
    
    /// Get method name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bilinear => "bilinear",
            Self::NearestNeighbor => "nearest",
            Self::ApproxBilinear => "approx_bilinear",
            Self::CatmullRom => "catmull_rom",
            Self::Lanczos3 => "lanczos3",
        }
    }
}

impl Default for ResizeMethod {
    fn default() -> Self {
        Self::Bilinear
    }
}

/// 3D image dimensions with spatial metadata
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageDimensions3D {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Depth (for 3D images/volume data)
    pub depth: u32,
    /// Number of channels (1, 3, or 4)
    pub channels: u8,
    /// Spatial coordinate for 3D positioning
    pub spatial_coord: ConversionCoordinate,
}

impl ImageDimensions3D {
    /// Create new 2D image dimensions
    pub fn new_2d(width: u32, height: u32, channels: u8) -> Self {
        Self {
            width,
            height,
            depth: 1,
            channels: channels.clamp(1, 4),
            spatial_coord: ConversionCoordinate::standard(),
        }
    }
    
    /// Create new 3D volume dimensions
    pub fn new_3d(width: u32, height: u32, depth: u32, channels: u8) -> Self {
        Self {
            width,
            height,
            depth,
            channels: channels.clamp(1, 4),
            spatial_coord: ConversionCoordinate::high_precision(),
        }
    }
    
    /// Total pixel count
    pub fn pixel_count(&self) -> u64 {
        (self.width as u64) * (self.height as u64) * (self.depth as u64)
    }
    
    /// Total float values when normalized
    pub fn tensor_size(&self) -> u64 {
        self.pixel_count() * (self.channels as u64)
    }
    
    /// Convert to spatial tensor metadata
    pub fn to_spatial(&self) -> SpatialTensorMetadata {
        SpatialTensorMetadata::new(self.width, self.height, self.depth)
    }
    
    /// Aspect ratio (width / height)
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            return 1.0;
        }
        self.width as f32 / self.height as f32
    }
    
    /// Check if landscape orientation
    pub fn is_landscape(&self) -> bool {
        self.width > self.height
    }
    
    /// Check if portrait orientation
    pub fn is_portrait(&self) -> bool {
        self.height > self.width
    }
}

/// Color padding options
#[derive(Debug, Clone, Copy)]
pub enum PadColor {
    /// White background
    White,
    /// Black background
    Black,
    /// Transparent (if supported)
    Transparent,
    /// Custom RGB color
    RGB(u8, u8, u8),
    /// Custom RGBA color
    RGBA(u8, u8, u8, u8),
}

impl PadColor {
    /// Convert to RGBA values
    pub fn to_rgba(&self) -> [u8; 4] {
        match self {
            Self::White => [255, 255, 255, 255],
            Self::Black => [0, 0, 0, 255],
            Self::Transparent => [0, 0, 0, 0],
            Self::RGB(r, g, b) => [*r, *g, *b, 255],
            Self::RGBA(r, g, b, a) => [*r, *g, *b, *a],
        }
    }
}

impl Default for PadColor {
    fn default() -> Self {
        Self::White
    }
}

/// 3D image processor for multimodal models
pub struct ImageProcessor3D;

impl ImageProcessor3D {
    /// Remove alpha channel by compositing over background color
    pub fn composite(img: &DynamicImage, bg: PadColor) -> Result<DynamicImage, ImageProcError> {
        let rgba = img.to_rgba8();
        let (width, height) = (rgba.width(), rgba.height());
        let bg_color = bg.to_rgba();
        
        // Create background image
        let mut output = ImageBuffer::from_pixel(width, height, Rgba(bg_color));
        
        // Composite original image over background
        for (x, y, pixel) in rgba.enumerate_pixels() {
            let alpha = pixel[3] as f32 / 255.0;
            let inv_alpha = 1.0 - alpha;
            
            let r = (pixel[0] as f32 * alpha + bg_color[0] as f32 * inv_alpha) as u8;
            let g = (pixel[1] as f32 * alpha + bg_color[1] as f32 * inv_alpha) as u8;
            let b = (pixel[2] as f32 * alpha + bg_color[2] as f32 * inv_alpha) as u8;
            
            output.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
        
        Ok(DynamicImage::ImageRgba8(output))
    }
    
    /// Resize image to target dimensions using specified method
    pub fn resize(img: &DynamicImage, target: ImageDimensions3D, method: ResizeMethod) -> Result<DynamicImage, ImageProcError> {
        if target.width == 0 || target.height == 0 {
            return Err(ImageProcError::InvalidDimensions(
                format!("Invalid target size: {}x{}", target.width, target.height)
            ));
        }
        
        let filter = method.to_filter();
        let resized = img.resize(target.width, target.height, filter);
        
        Ok(resized)
    }
    
    /// Pad image to target size preserving aspect ratio
    pub fn pad(img: &DynamicImage, target: ImageDimensions3D, color: PadColor, method: ResizeMethod) -> Result<DynamicImage, ImageProcError> {
        let (img_width, img_height) = (img.width(), img.height());
        let target_aspect = target.width as f32 / target.height as f32;
        let img_aspect = img_width as f32 / img_height as f32;
        
        // Calculate scaled dimensions preserving aspect ratio
        let (scaled_w, scaled_h) = if img_aspect > target_aspect {
            // Image is wider than target - fit to width
            let h = (target.width as f32 / img_aspect) as u32;
            (target.width, h)
        } else {
            // Image is taller than target - fit to height
            let w = (target.height as f32 * img_aspect) as u32;
            (w, target.height)
        };
        
        // Resize image to scaled dimensions
        let filter = method.to_filter();
        let scaled = img.resize(scaled_w, scaled_h, filter);
        
        // Create padded canvas
        let bg_color = color.to_rgba();
        let mut output = ImageBuffer::from_pixel(target.width, target.height, Rgba(bg_color));
        
        // Center the scaled image
        let x_offset = (target.width.saturating_sub(scaled_w)) / 2;
        let y_offset = (target.height.saturating_sub(scaled_h)) / 2;
        
        // Overlay scaled image
        imageops::overlay(&mut output, &scaled.to_rgba8(), x_offset as i64, y_offset as i64);
        
        Ok(DynamicImage::ImageRgba8(output))
    }
    
    /// Normalize image to float32 tensor with 3D spatial metadata
    pub fn normalize(
        img: &DynamicImage,
        mean: [f32; 3],
        std: [f32; 3],
        rescale: bool,
        channel_first: bool,
    ) -> Result<NormalizedImageTensor, ImageProcError> {
        let rgba = img.to_rgba8();
        let (width, height) = (rgba.width(), rgba.height());
        let pixel_count = (width * height) as usize;
        
        let mut tensor = NormalizedImageTensor {
            data: Vec::with_capacity(pixel_count * 3),
            dimensions: ImageDimensions3D::new_2d(width, height, 3),
            channel_first,
        };
        
        if channel_first {
            // Channel-first format: [C, H, W]
            let mut r_vals = Vec::with_capacity(pixel_count);
            let mut g_vals = Vec::with_capacity(pixel_count);
            let mut b_vals = Vec::with_capacity(pixel_count);
            
            for y in 0..height {
                for x in 0..width {
                    let pixel = rgba.get_pixel(x, y);
                    let (r, g, b) = if rescale {
                        (pixel[0] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[2] as f32 / 255.0)
                    } else {
                        (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32)
                    };
                    
                    r_vals.push((r - mean[0]) / std[0]);
                    g_vals.push((g - mean[1]) / std[1]);
                    b_vals.push((b - mean[2]) / std[2]);
                }
            }
            
            tensor.data.extend(r_vals);
            tensor.data.extend(g_vals);
            tensor.data.extend(b_vals);
        } else {
            // Channel-last format: [H, W, C]
            for y in 0..height {
                for x in 0..width {
                    let pixel = rgba.get_pixel(x, y);
                    let (r, g, b) = if rescale {
                        (pixel[0] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[2] as f32 / 255.0)
                    } else {
                        (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32)
                    };
                    
                    tensor.data.push((r - mean[0]) / std[0]);
                    tensor.data.push((g - mean[1]) / std[1]);
                    tensor.data.push((b - mean[2]) / std[2]);
                }
            }
        }
        
        Ok(tensor)
    }
    
    /// Extract image patches for vision transformers (3D spatial)
    pub fn extract_patches(
        img: &DynamicImage,
        patch_size: u32,
        overlap: u32,
    ) -> Result<Vec<ImagePatch>, ImageProcError> {
        let (width, height) = (img.width(), img.height());
        let step = patch_size.saturating_sub(overlap);
        
        if step == 0 {
            return Err(ImageProcError::InvalidDimensions(
                "Patch size must be larger than overlap".to_string()
            ));
        }
        
        let num_patches_x = (width + step - 1) / step;
        let num_patches_y = (height + step - 1) / step;
        let total_patches = num_patches_x * num_patches_y;
        
        let mut patches = Vec::with_capacity(total_patches as usize);
        let rgba = img.to_rgba8();
        
        for py in 0..num_patches_y {
            for px in 0..num_patches_x {
                let x = px * step;
                let y = py * step;
                let w = patch_size.min(width - x);
                let h = patch_size.min(height - y);
                
                // Extract patch
                let patch_img = imageops::crop_imm(&rgba, x, y, w, h).to_image();
                
                patches.push(ImagePatch {
                    image: DynamicImage::ImageRgba8(patch_img),
                    position: (x, y),
                    grid_position: (px, py),
                    spatial_coord: ConversionCoordinate::new(
                        (px * patch_size) as u64,
                        (py * patch_size) as u16,
                        0,
                        1.0,
                    ),
                });
            }
        }
        
        Ok(patches)
    }
    
    /// Load image from bytes
    pub fn from_bytes(data: &[u8]) -> Result<DynamicImage, ImageProcError> {
        image::load_from_memory(data)
            .map_err(|e| ImageProcError::InvalidFormat(format!("Failed to load image: {}", e)))
    }
}

/// Normalized image tensor with 3D metadata
#[derive(Debug, Clone)]
pub struct NormalizedImageTensor {
    /// Normalized float data
    pub data: Vec<f32>,
    /// Image dimensions
    pub dimensions: ImageDimensions3D,
    /// Channel ordering
    pub channel_first: bool,
}

impl NormalizedImageTensor {
    /// Get shape as [C, H, W] or [H, W, C]
    pub fn shape(&self) -> Vec<usize> {
        let h = self.dimensions.height as usize;
        let w = self.dimensions.width as usize;
        let c = self.dimensions.channels as usize;
        
        if self.channel_first {
            vec![c, h, w]
        } else {
            vec![h, w, c]
        }
    }
    
    /// Total element count
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Get value at position (handles both channel-first and channel-last)
    pub fn get(&self, c: usize, h: usize, w: usize) -> Option<f32> {
        let (height, width) = (self.dimensions.height as usize, self.dimensions.width as usize);
        
        if h >= height || w >= width || c >= self.dimensions.channels as usize {
            return None;
        }
        
        let idx = if self.channel_first {
            c * height * width + h * width + w
        } else {
            h * width * self.dimensions.channels as usize + w * self.dimensions.channels as usize + c
        };
        
        self.data.get(idx).copied()
    }
    
    /// Convert to 3D spatial tensor metadata
    pub fn to_spatial_metadata(&self) -> SpatialTensorMetadata {
        self.dimensions.to_spatial()
    }
}

/// Image patch with 3D spatial position
#[derive(Debug, Clone)]
pub struct ImagePatch {
    /// Patch image data
    pub image: DynamicImage,
    /// Position in original image (x, y)
    pub position: (u32, u32),
    /// Grid position (patch_x, patch_y)
    pub grid_position: (u32, u32),
    /// 3D spatial coordinate
    pub spatial_coord: ConversionCoordinate,
}

/// Image preprocessing pipeline
pub struct ImagePreprocessingPipeline {
    /// Target dimensions
    pub target_size: ImageDimensions3D,
    /// Resize method
    pub resize_method: ResizeMethod,
    /// Padding color
    pub pad_color: PadColor,
    /// Normalization mean
    pub mean: [f32; 3],
    /// Normalization std
    pub std: [f32; 3],
    /// Whether to rescale [0,255] to [0,1]
    pub rescale: bool,
    /// Channel-first output
    pub channel_first: bool,
    /// Whether to apply padding
    pub use_padding: bool,
}

impl ImagePreprocessingPipeline {
    /// Create CLIP preprocessing pipeline
    pub fn clip(size: u32) -> Self {
        Self {
            target_size: ImageDimensions3D::new_2d(size, size, 3),
            resize_method: ResizeMethod::Bilinear,
            pad_color: PadColor::Black,
            mean: norm_constants::CLIP_DEFAULT_MEAN,
            std: norm_constants::CLIP_DEFAULT_STD,
            rescale: true,
            channel_first: true,
            use_padding: true,
        }
    }
    
    /// Create ImageNet preprocessing pipeline
    pub fn imagenet(size: u32) -> Self {
        Self {
            target_size: ImageDimensions3D::new_2d(size, size, 3),
            resize_method: ResizeMethod::Bilinear,
            pad_color: PadColor::Black,
            mean: norm_constants::IMAGENET_DEFAULT_MEAN,
            std: norm_constants::IMAGENET_DEFAULT_STD,
            rescale: true,
            channel_first: true,
            use_padding: false,
        }
    }
    
    /// Create 3D spatial preprocessing for vision transformers
    pub fn vision_3d(width: u32, height: u32) -> Self {
        Self {
            target_size: ImageDimensions3D::new_2d(width, height, 3),
            resize_method: ResizeMethod::Lanczos3,
            pad_color: PadColor::White,
            mean: norm_constants::VISION_3D_MEAN,
            std: norm_constants::VISION_3D_STD,
            rescale: true,
            channel_first: true,
            use_padding: true,
        }
    }
    
    /// Process image through full pipeline
    pub fn process(&self, img: &DynamicImage) -> Result<NormalizedImageTensor, ImageProcError> {
        // Composite alpha if present
        let img = if img.color().has_alpha() {
            ImageProcessor3D::composite(img, self.pad_color)?
        } else {
            img.clone()
        };
        
        // Resize or pad
        let img = if self.use_padding {
            ImageProcessor3D::pad(&img, self.target_size, self.pad_color, self.resize_method)?
        } else {
            ImageProcessor3D::resize(&img, self.target_size, self.resize_method)?
        };
        
        // Normalize
        ImageProcessor3D::normalize(&img, self.mean, self.std, self.rescale, self.channel_first)
    }
    
    /// Process from raw bytes
    pub fn process_bytes(&self, data: &[u8]) -> Result<NormalizedImageTensor, ImageProcError> {
        let img = ImageProcessor3D::from_bytes(data)?;
        self.process(&img)
    }
}

impl Default for ImagePreprocessingPipeline {
    fn default() -> Self {
        Self::imagenet(224)
    }
}

/// Utility functions
pub mod image_utils {
    use super::*;
    
    /// Calculate 3D embedding position for image patch
    pub fn patch_spatial_position(
        patch_x: u32,
        patch_y: u32,
        patch_size: u32,
        image_dims: &ImageDimensions3D,
    ) -> ConversionCoordinate {
        let x = (patch_x * patch_size) as u64;
        let y = (patch_y * patch_size) as u16;
        let depth = (image_dims.channels as u16) * 10; // Depth based on channels
        
        ConversionCoordinate::new(x, y as u16, depth, 1.0)
    }
    
    /// Estimate memory usage for processed image
    pub fn estimate_memory_bytes(dims: &ImageDimensions3D) -> u64 {
        dims.tensor_size() * 4 // f32 = 4 bytes
    }
    
    /// Validate image format is supported
    pub fn validate_format(data: &[u8]) -> Result<&'static str, ImageProcError> {
        // Check magic bytes
        if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            Ok("JPEG")
        } else if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
            Ok("PNG")
        } else if data.starts_with(b"GIF89a") || data.starts_with(b"GIF87a") {
            Ok("GIF")
        } else if data.starts_with(&[0x42, 0x4D]) {
            Ok("BMP")
        } else if data.starts_with(&[0x52, 0x49, 0x46, 0x46]) {
            Ok("WEBP")
        } else {
            Err(ImageProcError::InvalidFormat("Unknown image format".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dimensions_3d() {
        let dims = ImageDimensions3D::new_2d(224, 224, 3);
        assert_eq!(dims.pixel_count(), 224 * 224);
        assert_eq!(dims.aspect_ratio(), 1.0);
        assert!(dims.is_landscape() == false && dims.is_portrait() == false);
    }
    
    #[test]
    fn test_resize_method() {
        assert_eq!(ResizeMethod::Bilinear.as_str(), "bilinear");
        assert_eq!(ResizeMethod::Lanczos3.as_str(), "lanczos3");
    }
    
    #[test]
    fn test_pad_color() {
        let white = PadColor::White.to_rgba();
        assert_eq!(white, [255, 255, 255, 255]);
        
        let custom = PadColor::RGB(128, 64, 32).to_rgba();
        assert_eq!(custom, [128, 64, 32, 255]);
    }
    
    #[test]
    fn test_normalize_tensor() {
        // Create a simple 2x2 test image
        let img = DynamicImage::ImageRgba8(ImageBuffer::from_fn(2, 2, |_, _| {
            Rgba([128, 64, 32, 255])
        }));
        
        let mean = [0.5, 0.5, 0.5];
        let std = [0.5, 0.5, 0.5];
        
        let tensor = ImageProcessor3D::normalize(&img, mean, std, true, true).unwrap();
        assert_eq!(tensor.len(), 12); // 2x2x3
        assert!(tensor.channel_first);
    }
    
    #[test]
    fn test_image_pipeline() {
        let pipeline = ImagePreprocessingPipeline::clip(224);
        
        // Test with a simple image
        let img = DynamicImage::ImageRgba8(ImageBuffer::from_fn(100, 100, |_, _| {
            Rgba([255, 128, 64, 255])
        }));
        
        let tensor = pipeline.process(&img).unwrap();
        assert_eq!(tensor.shape(), vec![3, 224, 224]);
    }
    
    #[test]
    fn test_patch_extraction() {
        let img = DynamicImage::ImageRgba8(ImageBuffer::from_fn(64, 64, |_, _| {
            Rgba([128, 128, 128, 255])
        }));
        
        let patches = ImageProcessor3D::extract_patches(&img, 16, 0).unwrap();
        assert_eq!(patches.len(), 16); // 4x4 grid
        
        // Check first patch position
        assert_eq!(patches[0].position, (0, 0));
        assert_eq!(patches[0].grid_position, (0, 0));
    }
    
    #[test]
    fn test_spatial_coordinate() {
        let coord = image_utils::patch_spatial_position(
            2, 3, 16, &ImageDimensions3D::new_2d(64, 64, 3)
        );
        assert_eq!(coord.tensor_sequence, 32); // 2 * 16
        assert_eq!(coord.architecture_tier, 48); // 3 * 16
    }
}
