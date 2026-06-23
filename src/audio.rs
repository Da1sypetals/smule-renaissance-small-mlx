//! Audio decoding and the preprocessing/postprocessing used by the reference.

use std::path::Path;
use std::process::Command;

use babycat::{Signal, Waveform, WaveformArgs};

use crate::{Error, Result};

/// Model sample rate.
pub const SAMPLE_RATE: u32 = 48_000;
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
    /// Decode audio, convert it to mono, and resample to the model rate.
    ///
    /// Babycat is used as the primary file decoder. If Babycat's bundled decoder cannot
    /// identify the container, the local `ffmpeg` binary is used to transcode the input
    /// to in-memory WAV bytes, which are then decoded through Babycat as the common path.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        match load_with_babycat_file(path) {
            Ok(buffer) => Ok(buffer),
            Err(babycat_error) => match load_with_ffmpeg_fallback(path) {
                Ok(buffer) => Ok(buffer),
                Err(fallback_error) => Err(Error::AudioDecode {
                    babycat: babycat_error,
                    fallback: fallback_error,
                }),
            },
        }
    }

    /// Apply the reference 60 Hz high-pass biquad after decode-time sample-rate conversion.
    pub fn preprocess(mut self) -> Result<Self> {
        if self.sample_rate != SAMPLE_RATE {
            return Err(Error::InvalidAudio(format!(
                "decoded audio has sample rate {}, expected {}",
                self.sample_rate, SAMPLE_RATE
            )));
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

fn waveform_args() -> WaveformArgs {
    WaveformArgs {
        frame_rate_hz: SAMPLE_RATE,
        convert_to_mono: true,
        ..Default::default()
    }
}

fn audio_buffer_from_waveform(waveform: Waveform) -> Result<AudioBuffer> {
    let samples = waveform.to_interleaved_samples().to_vec();
    if samples.is_empty() {
        return Err(Error::InvalidAudio("audio contains no samples".to_owned()));
    }
    Ok(AudioBuffer {
        samples,
        sample_rate: waveform.frame_rate_hz(),
    })
}

fn load_with_babycat_file(path: &Path) -> std::result::Result<AudioBuffer, String> {
    let path = path
        .to_str()
        .ok_or_else(|| format!("path is not valid UTF-8: {path:?}"))?;
    let waveform = Waveform::from_file(path, waveform_args()).map_err(|error| error.to_string())?;
    audio_buffer_from_waveform(waveform).map_err(|error| error.to_string())
}

fn load_with_ffmpeg_fallback(path: &Path) -> std::result::Result<AudioBuffer, String> {
    let output = Command::new("ffmpeg")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(path)
        .arg("-f")
        .arg("wav")
        .arg("-acodec")
        .arg("pcm_f32le")
        .arg("pipe:1")
        .output()
        .map_err(|error| format!("failed to start ffmpeg: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffmpeg exited with {}: {stderr}", output.status));
    }

    let waveform =
        Waveform::from_encoded_bytes_with_hint(&output.stdout, waveform_args(), "wav", "audio/wav")
            .map_err(|error| error.to_string())?;
    audio_buffer_from_waveform(waveform).map_err(|error| error.to_string())
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
    fn preprocess_rejects_unexpected_rate() {
        let buffer = AudioBuffer {
            samples: vec![0.0],
            sample_rate: 44_100,
        };
        assert!(buffer.preprocess().is_err());
    }
}
