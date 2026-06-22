use thiserror::Error;

/// Errors produced by SRS model loading and inference.
#[derive(Debug, Error)]
pub enum Error {
    /// An MLX operation failed.
    #[error(transparent)]
    Mlx(#[from] mlx_rs::error::Exception),

    /// MLX could not read a checkpoint.
    #[error(transparent)]
    MlxIo(#[from] mlx_rs::error::IoError),

    /// WAV decoding or encoding failed.
    #[error(transparent)]
    Wav(#[from] hound::Error),

    /// The input WAV encoding is not supported by the exact decoder.
    #[error("unsupported WAV encoding: {0}")]
    UnsupportedWav(String),

    /// An audio stream has no channels or samples.
    #[error("invalid audio: {0}")]
    InvalidAudio(String),

    /// A required checkpoint tensor was absent.
    #[error("checkpoint is missing tensor {0}")]
    MissingTensor(String),

    /// A checkpoint tensor has a layout incompatible with this port.
    #[error("tensor {name} has shape {actual:?}, expected {expected:?}")]
    TensorShape {
        /// Tensor name.
        name: String,
        /// Loaded shape.
        actual: Vec<i32>,
        /// Required MLX-native shape.
        expected: Vec<i32>,
    },

    /// Safetensors metadata does not describe the required converted layout.
    #[error("invalid checkpoint metadata: {0}")]
    CheckpointMetadata(String),

    /// The checkpoint contains tensors that this exact architecture did not consume.
    #[error("checkpoint contains unexpected tensors: {0:?}")]
    UnexpectedTensors(Vec<String>),

    /// Input dimensions do not match the fixed SRS architecture.
    #[error("invalid input shape {actual:?}; expected [batch, frames, 2049, 2]")]
    InputShape {
        /// Actual input shape.
        actual: Vec<i32>,
    },
}

/// Result type used by this crate.
pub type Result<T> = std::result::Result<T, Error>;
