//! PyTorch-compatible centered STFT and inverse STFT implemented with MLX FFT primitives.

use std::f32::consts::PI;

use mlx_rs::fft::{irfft, rfft};
use mlx_rs::ops::indexing::{IndexOp, scatter_add_single};
use mlx_rs::ops::{broadcast_to, concatenate_axis, stack_axis};
use mlx_rs::{Array, complex64};

use crate::{Error, Result};

/// FFT and window length used by SRS.
pub const N_FFT: i32 = 4096;
/// STFT frame hop used by SRS.
pub const HOP_LENGTH: i32 = 2048;

/// Periodic Hann-window STFT/iSTFT pair matching the Python reference settings.
#[derive(Debug)]
pub struct SpectralTransform {
    window: Array,
}

impl SpectralTransform {
    /// Construct the fixed 4096/2048 transform.
    pub fn new() -> Self {
        let window: Vec<f32> = (0..N_FFT)
            .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / N_FFT as f32).cos())
            .collect();
        Self {
            window: Array::from_slice(&window, &[N_FFT]),
        }
    }

    /// Transform `[batch, samples]` FP32 waveforms into `[batch, frames, frequency, 2]`.
    pub fn stft(&self, waveform: &Array) -> Result<Array> {
        if waveform.shape().len() != 2 || waveform.dim(1) <= N_FFT / 2 {
            return Err(Error::InvalidAudio(format!(
                "STFT expects [batch, samples] with more than {} samples, got {:?}",
                N_FFT / 2,
                waveform.shape()
            )));
        }
        let batch = waveform.dim(0);
        let samples = waveform.dim(1);
        let reflect = N_FFT / 2;
        let left_indices: Vec<i32> = (1..=reflect).rev().collect();
        let right_indices: Vec<i32> = ((samples - reflect - 1)..=(samples - 2)).rev().collect();
        let left = waveform.take_axis(Array::from_slice(&left_indices, &[reflect]), 1)?;
        let right = waveform.take_axis(Array::from_slice(&right_indices, &[reflect]), 1)?;
        let padded = concatenate_axis(&[left, waveform.clone(), right], 1)?;
        let padded_samples = padded.dim(1);
        let frames = 1 + (padded_samples - N_FFT) / HOP_LENGTH;
        let framed = padded.as_strided(
            &[batch, frames, N_FFT],
            &[i64::from(padded_samples), i64::from(HOP_LENGTH), 1],
            0,
        )?;
        let spectrum = rfft(framed.multiply(&self.window)?, N_FFT, -1)?;
        Ok(stack_axis(&[spectrum.real()?, spectrum.imag()?], -1)?)
    }

    /// Invert `[batch, frames, frequency, 2]` into the centered waveform length used by torch.istft.
    pub fn istft(&self, spectrum: &Array) -> Result<Array> {
        if spectrum.shape().len() != 4 || spectrum.dim(2) != N_FFT / 2 + 1 || spectrum.dim(3) != 2 {
            return Err(Error::InvalidAudio(format!(
                "iSTFT expects [batch, frames, {}, 2], got {:?}",
                N_FFT / 2 + 1,
                spectrum.shape()
            )));
        }
        let real = spectrum
            .index((.., .., .., 0))
            .as_dtype(mlx_rs::Dtype::Complex64)?;
        let imaginary = spectrum
            .index((.., .., .., 1))
            .as_dtype(mlx_rs::Dtype::Complex64)?;
        let complex =
            real.add(imaginary.multiply(Array::from_complex(complex64::new(0.0, 1.0)))?)?;
        let frames = irfft(&complex, N_FFT, -1)?.multiply(&self.window)?;

        let batch = frames.dim(0);
        let frame_count = frames.dim(1);
        let overlap_length = N_FFT + HOP_LENGTH * (frame_count - 1);
        let indices: Vec<u32> = (0..frame_count)
            .flat_map(|frame| {
                let start = frame * HOP_LENGTH;
                (0..N_FFT).map(move |sample| (start + sample) as u32)
            })
            .collect();
        let index_array = Array::from_slice(&indices, &[frame_count * N_FFT]);
        let waveform = scatter_add_single(
            Array::zeros::<f32>(&[batch, overlap_length])?,
            &index_array,
            frames
                .reshape(&[batch, frame_count * N_FFT])?
                .transpose_axes(&[1, 0])?
                .expand_dims(-1)?,
            1,
        )?;

        let squared_window = self.window.square()?;
        let envelope_updates =
            broadcast_to(squared_window.reshape(&[1, N_FFT])?, &[frame_count, N_FFT])?
                .reshape(&[frame_count * N_FFT])?;
        let envelope = scatter_add_single(
            Array::zeros::<f32>(&[overlap_length])?,
            &index_array,
            envelope_updates.expand_dims(-1)?,
            0,
        )?;
        let normalized = waveform.divide(envelope)?;
        Ok(normalized.index((.., N_FFT / 2..overlap_length - N_FFT / 2)))
    }
}

impl Default for SpectralTransform {
    fn default() -> Self {
        Self::new()
    }
}
