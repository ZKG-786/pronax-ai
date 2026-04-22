//! ProNax Gemma4 - Advanced 3D Multimodal AI Model Module

pub mod pronax_gemma4_model;
pub mod pronax_gemma4_image_proc;
pub mod pronax_gemma4_audio_proc;
pub mod pronax_gemma4_audio;
pub mod pronax_gemma4_vision;

// Re-export main types for convenience
pub use pronax_gemma4_model::{
    Gemma4Model3D,
    Gemma4Config3D,
    Gemma4Error,
    Gemma4ModelInfo,
    Gemma4TextModel3D,
    SlidingWindowCache3D,
    CacheType,
};

pub use pronax_gemma4_image_proc::{
    Gemma4ImageProcessor3D,
    Gemma4VisionEncoder3D,
    Gemma4VisionProjector3D,
    VisionConfig3D,
    Gemma4VisionError,
};

pub use pronax_gemma4_audio_proc::{
    Gemma4AudioProcessor3D,
    Gemma4AudioEncoder3D,
    Gemma4AudioProjector3D,
    AudioConfig3D,
    Gemma4AudioError,
};

pub use pronax_gemma4_audio::{
    PronaxAudioEncoder3D,
    PronaxAudioHyperparams3D,
    PronaxAudioTextProjector3D,
    PronaxAudioEncoderError,
    PronaxTensorView3D,
    PronaxSSCPBlock3D,
    PronaxBoundedLinear3D,
    PronaxConformerBlock3D,
    PronaxAudioOutput3D,
};

pub use pronax_gemma4_vision::{
    PronaxVisionEncoder3D,
    PronaxVisionHyperparams3D,
    PronaxVisionTextProjector3D,
    PronaxVisionEncoderError,
    PronaxVisionTensorView3D,
    PronaxClippableLinear3D,
    PronaxTwoDimensionalRoPE3D,
    PronaxVisionSelfAttention3D,
    PronaxVisionMLP3D,
    PronaxVisionEncoderLayer3D,
};

/// Gemma4 model factory for easy instantiation
pub struct Gemma4Factory;

impl Gemma4Factory {
    /// Create Gemma4 2B model
    pub fn create_2b() -> Result<Gemma4Model3D, Gemma4Error> {
        let config = Gemma4Config3D::gemma4_2b();
        Gemma4Model3D::new(config)
    }
    
    /// Create Gemma4 9B model
    pub fn create_9b() -> Result<Gemma4Model3D, Gemma4Error> {
        let config = Gemma4Config3D::gemma4_9b();
        Gemma4Model3D::new(config)
    }
    
    /// Create Gemma4 27B model
    pub fn create_27b() -> Result<Gemma4Model3D, Gemma4Error> {
        let config = Gemma4Config3D::gemma4_27b();
        Gemma4Model3D::new(config)
    }
    
    /// Create custom Gemma4 model
    pub fn create_custom(config: Gemma4Config3D) -> Result<Gemma4Model3D, Gemma4Error> {
        Gemma4Model3D::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_factory_2b() {
        let model = Gemma4Factory::create_2b();
        assert!(model.is_ok());
        
        let model = model.unwrap();
        let info = model.model_info();
        assert_eq!(info.variant, "2B");
    }
    
    #[test]
    fn test_factory_9b() {
        let model = Gemma4Factory::create_9b();
        assert!(model.is_ok());
        
        let model = model.unwrap();
        let info = model.model_info();
        assert_eq!(info.variant, "9B");
    }
    
    #[test]
    fn test_factory_27b() {
        let model = Gemma4Factory::create_27b();
        assert!(model.is_ok());
        
        let model = model.unwrap();
        let info = model.model_info();
        assert_eq!(info.variant, "27B");
    }
}
