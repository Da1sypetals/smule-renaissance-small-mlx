use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use mlx_rs::error::Exception;
use mlx_rs::ops::{concatenate_axis, indexing::IndexOp};
use mlx_rs::transforms::compile::compile_with_state;
use mlx_rs::{Array, Device};
use srs_inference::audio::{AudioBuffer, SAMPLE_RATE};
use srs_inference::spectral::{HOP_LENGTH, SpectralTransform};
use srs_inference::{Renaissance, TEMPORAL_RECEPTIVE_RADIUS_FRAMES};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ComputeDevice {
    Cpu,
    Gpu,
}

#[derive(Debug, Parser)]
#[command(about = "Fixed-graph streaming inference for Smule Renaissance Small")]
struct Args {
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(short, long)]
    checkpoint: PathBuf,

    #[arg(long, value_enum, default_value_t = ComputeDevice::Gpu)]
    device: ComputeDevice,

    /// Number of newly emitted STFT frames per model call.
    #[arg(long, default_value_t = 8)]
    chunk_frames: i32,

    /// Available past frames. Use 129 for exact steady-state inference.
    #[arg(long, default_value_t = TEMPORAL_RECEPTIVE_RADIUS_FRAMES)]
    left_context_frames: i32,

    /// Future frames. Use 129 for exact output; lower values trade quality for latency.
    #[arg(long, default_value_t = TEMPORAL_RECEPTIVE_RADIUS_FRAMES)]
    right_context_frames: i32,
}

fn compiled_forward(model: &mut Renaissance, input: &Array) -> Result<Array, Exception> {
    model
        .forward(input)
        .map_err(|error| Exception::custom(error.to_string()))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.chunk_frames <= 0 {
        bail!("--chunk-frames must be positive");
    }
    if args.left_context_frames < 0 || args.right_context_frames < 0 {
        bail!("context frame counts cannot be negative");
    }

    let device = match args.device {
        ComputeDevice::Cpu => Device::cpu(),
        ComputeDevice::Gpu => Device::gpu(),
    };
    Device::set_default(&device);
    println!("Using device: {device}");

    let mut model = Renaissance::load(&args.checkpoint)
        .with_context(|| format!("loading {}", args.checkpoint.display()))?;
    let decoded = AudioBuffer::load(&args.input)
        .with_context(|| format!("loading {}", args.input.display()))?;
    let mut waveform = decoded.preprocess()?;
    let normalization_factor = waveform.normalize();
    let audio_duration = waveform.samples.len() as f64 / f64::from(SAMPLE_RATE);

    let spectral = SpectralTransform::new();
    let input = Array::from_slice(&waveform.samples, &[1, waveform.samples.len() as i32]);
    let input_spectrum = spectral.stft(&input)?;
    input_spectrum.eval()?;

    let total_frames = input_spectrum.dim(1);
    let window_frames = args.left_context_frames + args.chunk_frames + args.right_context_frames;
    if window_frames > total_frames {
        bail!(
            "the fixed {}-frame window exceeds the {}-frame input; reduce chunk/context sizes",
            window_frames,
            total_frames
        );
    }

    println!(
        "Streaming geometry: chunk={}, left={}, right={}, fixed input={}",
        args.chunk_frames, args.left_context_frames, args.right_context_frames, window_frames
    );
    if args.left_context_frames < TEMPORAL_RECEPTIVE_RADIUS_FRAMES
        || args.right_context_frames < TEMPORAL_RECEPTIVE_RADIUS_FRAMES
    {
        println!(
            "Approximate mode: exact temporal context requires {} frames on both sides",
            TEMPORAL_RECEPTIVE_RADIUS_FRAMES
        );
    } else {
        println!("Exact temporal-context mode");
    }

    // MLX compiles on the first invocation. The model is explicit compilation
    // state, so all 910 checkpoint tensors are graph inputs rather than unsafe
    // captured arrays. Every subsequent call has the same shape and dtype.
    let mut compiled = compile_with_state(compiled_forward, false);
    let warmup_window = input_spectrum.index((.., 0..window_frames, .., ..));
    let compile_started = Instant::now();
    compiled(&mut model, &warmup_window)?.eval()?;
    let compile_elapsed = compile_started.elapsed();
    println!(
        "Graph compilations: 1 (warm-up {:.3} s)",
        compile_elapsed.as_secs_f64()
    );

    let stream_started = Instant::now();
    let mut model_elapsed = Duration::ZERO;
    let mut maximum_block_elapsed = Duration::ZERO;
    let mut output_chunks = Vec::new();
    let mut output_start = 0;
    let maximum_window_start = total_frames - window_frames;

    while output_start < total_frames {
        let output_end = (output_start + args.chunk_frames).min(total_frames);
        let preferred_window_start = (output_start - args.left_context_frames).max(0);
        let window_start = preferred_window_start.min(maximum_window_start);
        let local_start = output_start - window_start;
        let local_end = output_end - window_start;
        let window = input_spectrum.index((.., window_start..window_start + window_frames, .., ..));
        if window.shape() != [1, window_frames, 2049, 2] {
            bail!(
                "streaming window shape changed unexpectedly: {:?}",
                window.shape()
            );
        }

        let block_started = Instant::now();
        let estimated_window = compiled(&mut model, &window)?;
        estimated_window.eval()?;
        let block_elapsed = block_started.elapsed();
        model_elapsed += block_elapsed;
        maximum_block_elapsed = maximum_block_elapsed.max(block_elapsed);
        output_chunks.push(estimated_window.index((.., local_start..local_end, .., ..)));
        output_start = output_end;
    }

    let streamed_spectrum = concatenate_axis(&output_chunks, 1)?;
    streamed_spectrum.eval()?;
    if streamed_spectrum.dim(1) != total_frames {
        bail!(
            "stream produced {} frames, expected {}",
            streamed_spectrum.dim(1),
            total_frames
        );
    }

    let enhanced = spectral.istft(&streamed_spectrum)?;
    let output_samples: Vec<f32> = enhanced
        .as_slice::<f32>()
        .iter()
        .map(|sample| sample * normalization_factor)
        .collect();
    AudioBuffer {
        samples: output_samples,
        sample_rate: SAMPLE_RATE,
    }
    .save_f32_wav(&args.output)?;

    let stream_elapsed = stream_started.elapsed();
    let model_rtf = model_elapsed.as_secs_f64() / audio_duration;
    let block_audio_duration = f64::from(args.chunk_frames * HOP_LENGTH) / f64::from(SAMPLE_RATE);
    let block_deadline_ratio = maximum_block_elapsed.as_secs_f64() / block_audio_duration;
    let upper_bound_latency =
        f64::from((args.chunk_frames + args.right_context_frames + 1) * HOP_LENGTH)
            / f64::from(SAMPLE_RATE);

    println!("Processed blocks: {}", output_chunks.len());
    println!("Model time: {:.3} s", model_elapsed.as_secs_f64());
    println!("Model streaming RTF: {model_rtf:.4}");
    println!(
        "Maximum block time: {:.3} ms ({block_deadline_ratio:.3}x its audio deadline)",
        maximum_block_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "Block latency upper bound: {:.1} ms (STFT + chunk accumulation + lookahead)",
        upper_bound_latency * 1000.0
    );
    println!(
        "End-to-end file processing after warm-up: {:.3} s",
        stream_elapsed.as_secs_f64()
    );
    println!("Saved: {}", args.output.display());
    Ok(())
}
