//! Audio decoding and the exact preprocessing/postprocessing used by the reference.

use std::f64::consts::PI;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::ops::{conv1d, indexing::IndexOp, pad};

use crate::{Error, Result};

/// Model sample rate.
pub const SAMPLE_RATE: u32 = 48_000;
const LOWPASS_FILTER_WIDTH: i32 = 6;
const RESAMPLE_ROLLOFF: f64 = 0.99;
const HIGHPASS_CUTOFF: f32 = 60.0;
const HIGHPASS_Q: f32 = 0.707;

/// A mono floating-point waveform and its sample rate.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Interleaved-free mono samples.
    pub samples: Vec<f32>,
    /// Samples per second.
    pub sample_rate: u32,
}

impl AudioBuffer {
    /// Decode a WAV with the same normalized PCM convention used by torchaudio.
    pub fn load_wav(path: impl AsRef<Path>) -> Result<Self> {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        if spec.channels == 0 {
            return Err(Error::InvalidAudio("WAV declares zero channels".to_owned()));
        }

        let interleaved = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, 32) => reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()?,
            (hound::SampleFormat::Int, bits @ 1..=8) => {
                let scale = 2.0f32.powi(i32::from(bits) - 1);
                reader
                    .samples::<i8>()
                    .map(|sample| sample.map(|value| f32::from(value) / scale))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
            (hound::SampleFormat::Int, bits @ 9..=16) => {
                let scale = 2.0f32.powi(i32::from(bits) - 1);
                reader
                    .samples::<i16>()
                    .map(|sample| sample.map(|value| f32::from(value) / scale))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
            (hound::SampleFormat::Int, bits @ 17..=32) => {
                let scale = 2.0f32.powi(i32::from(bits) - 1);
                reader
                    .samples::<i32>()
                    .map(|sample| sample.map(|value| value as f32 / scale))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
            (format, bits) => {
                return Err(Error::UnsupportedWav(format!(
                    "{format:?} with {bits} bits per sample"
                )));
            }
        };

        if interleaved.is_empty() {
            return Err(Error::InvalidAudio("WAV contains no samples".to_owned()));
        }
        let channels = usize::from(spec.channels);
        if interleaved.len() % channels != 0 {
            return Err(Error::InvalidAudio(
                "interleaved sample count is not divisible by channel count".to_owned(),
            ));
        }

        let samples = if channels == 1 {
            interleaved
        } else {
            interleaved
                .chunks_exact(channels)
                .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
                .collect()
        };
        Ok(Self {
            samples,
            sample_rate: spec.sample_rate,
        })
    }

    /// Apply torchaudio's default sinc resampler and 60 Hz high-pass biquad.
    pub fn preprocess(mut self) -> Result<Self> {
        if self.sample_rate != SAMPLE_RATE {
            self.samples = sinc_resample(&self.samples, self.sample_rate, SAMPLE_RATE)?;
            self.sample_rate = SAMPLE_RATE;
        }
        highpass_biquad_in_place(&mut self.samples, SAMPLE_RATE);
        Ok(self)
    }

    /// Normalize by absolute peak, returning the original peak for later restoration.
    pub fn normalize(&mut self) -> f32 {
        let peak = self
            .samples
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0, f32::max);
        if peak > 0.0 {
            for sample in &mut self.samples {
                *sample /= peak;
            }
        }
        peak
    }

    /// Write a mono IEEE-float WAV, matching `torchaudio.save` for FP32 output.
    pub fn save_f32_wav(&self, path: impl AsRef<Path>) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for &sample in &self.samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
        Ok(())
    }
}

fn greatest_common_divisor(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn sinc_resample(waveform: &[f32], original_rate: u32, new_rate: u32) -> Result<Vec<f32>> {
    let gcd = greatest_common_divisor(original_rate, new_rate);
    let original = (original_rate / gcd) as i32;
    let new = (new_rate / gcd) as i32;
    let base_frequency = f64::from(original.min(new)) * RESAMPLE_ROLLOFF;
    let width = (f64::from(LOWPASS_FILTER_WIDTH * original) / base_frequency).ceil() as i32;
    let kernel_size = width * 2 + original;
    let mut kernel = Vec::with_capacity((new * kernel_size) as usize);

    // This is torchaudio.functional._get_sinc_resample_kernel's float64
    // construction followed by its default conversion to float32.
    for phase in 0..new {
        for offset in -width..(width + original) {
            let index = f64::from(offset) / f64::from(original);
            let mut time = (-f64::from(phase) / f64::from(new) + index) * base_frequency;
            time = time.clamp(
                -f64::from(LOWPASS_FILTER_WIDTH),
                f64::from(LOWPASS_FILTER_WIDTH),
            );
            let window = (time * PI / f64::from(LOWPASS_FILTER_WIDTH) / 2.0)
                .cos()
                .powi(2);
            let angle = time * PI;
            let sinc = if angle == 0.0 {
                1.0
            } else {
                angle.sin() / angle
            };
            kernel.push((sinc * window * base_frequency / f64::from(original)) as f32);
        }
    }

    let input = Array::from_slice(waveform, &[1, waveform.len() as i32, 1]);
    let padded = pad(
        &input,
        &[(0, 0), (width, width + original), (0, 0)],
        None,
        None,
    )?;
    let weights = Array::from_slice(&kernel, &[new, kernel_size, 1]);
    let resampled = conv1d(&padded, &weights, original, 0, 1, 1)?;
    let target_length =
        ((u64::from(new_rate) * waveform.len() as u64).div_ceil(u64::from(original_rate))) as i32;
    let flattened = resampled.reshape(&[1, -1])?.index((.., 0..target_length));
    Ok(flattened.as_slice::<f32>().to_vec())
}

fn highpass_biquad_in_place(waveform: &mut [f32], sample_rate: u32) {
    let angular = 2.0f32 * std::f32::consts::PI * HIGHPASS_CUTOFF / sample_rate as f32;
    let cosine = angular.cos();
    let alpha = angular.sin() / 2.0 / HIGHPASS_Q;
    let a0 = 1.0 + alpha;
    let b0 = ((1.0 + cosine) / 2.0) / a0;
    let b1 = (-1.0 - cosine) / a0;
    let b2 = b0;
    let a1 = (-2.0 * cosine) / a0;
    let a2 = (1.0 - alpha) / a0;

    let mut input_1 = 0.0f32;
    let mut input_2 = 0.0f32;
    let mut output_1 = 0.0f32;
    let mut output_2 = 0.0f32;
    for sample in waveform {
        let input = *sample;
        let output = b0 * input + b1 * input_1 + b2 * input_2 - a1 * output_1 - a2 * output_2;
        input_2 = input_1;
        input_1 = input;
        output_2 = output_1;
        output_1 = output;
        *sample = output.clamp(-1.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_reduces_audio_rates() {
        assert_eq!(greatest_common_divisor(44_100, 48_000), 300);
    }
}
