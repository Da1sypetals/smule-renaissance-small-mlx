use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use mlx_rs::{Array, Device};
use srs_inference::Renaissance;
use srs_inference::audio::{AudioBuffer, SAMPLE_RATE};
use srs_inference::spectral::SpectralTransform;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ComputeDevice {
    Cpu,
    Gpu,
}

#[derive(Debug, Parser)]
#[command(about = "Smule Renaissance Small inference with Rust and MLX")]
struct Args {
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(short, long)]
    checkpoint: PathBuf,

    #[arg(long, value_enum, default_value_t = ComputeDevice::Gpu)]
    device: ComputeDevice,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = match args.device {
        ComputeDevice::Cpu => Device::cpu(),
        ComputeDevice::Gpu => Device::gpu(),
    };
    Device::set_default(&device);
    println!("Using device: {device} with FP32 precision");

    println!("Loading model from {}...", args.checkpoint.display());
    let mut model = Renaissance::load(&args.checkpoint)?;
    println!("Loading audio from {}...", args.input.display());
    let decoded = AudioBuffer::load_wav(&args.input)?;
    if decoded.sample_rate != SAMPLE_RATE {
        println!(
            "Resampling from {} Hz to {} Hz",
            decoded.sample_rate, SAMPLE_RATE
        );
    }
    let mut waveform = decoded.preprocess()?;
    println!(
        "Audio duration: {:.2} seconds",
        waveform.samples.len() as f64 / f64::from(SAMPLE_RATE)
    );
    let normalization_factor = waveform.normalize();

    let spectral = SpectralTransform::new();
    let input = Array::from_slice(&waveform.samples, &[1, waveform.samples.len() as i32]);
    let input_spectrum = spectral.stft(&input)?;
    println!("Input STFT shape: {:?}", input_spectrum.shape());

    println!("Processing audio...");
    let inference_start = Instant::now();
    let enhanced_spectrum = model.forward(&input_spectrum)?;
    enhanced_spectrum.eval()?;
    let inference_elapsed = inference_start.elapsed();
    println!("Output STFT shape: {:?}", enhanced_spectrum.shape());

    let enhanced = spectral.istft(&enhanced_spectrum)?;
    let output_samples: Vec<f32> = enhanced
        .as_slice::<f32>()
        .iter()
        .map(|sample| sample * normalization_factor)
        .collect();
    let output = AudioBuffer {
        samples: output_samples,
        sample_rate: SAMPLE_RATE,
    };
    println!("Saving enhanced audio to {}...", args.output.display());
    output.save_f32_wav(&args.output)?;

    let duration = waveform.samples.len() as f64 / f64::from(SAMPLE_RATE);
    println!("Inference time: {:.3} s", inference_elapsed.as_secs_f64());
    println!(
        "Inference RTF: {:.4}",
        inference_elapsed.as_secs_f64() / duration
    );
    println!("Done!");
    Ok(())
}
