use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use mlx_rs::{Array, Device};
use srs_inference::Renaissance;

#[derive(Debug, Parser)]
#[command(about = "Run the MLX network on a saved PyTorch STFT alignment tensor")]
struct Args {
    #[arg(short, long)]
    checkpoint: PathBuf,

    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(long)]
    trace_directory: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    Device::set_default(&Device::gpu());
    let input = Array::load_numpy(&args.input)?;
    let mut model = Renaissance::load(&args.checkpoint)?;
    if let Some(directory) = &args.trace_directory {
        std::fs::create_dir_all(directory)?;
    }
    let output = if let Some(directory) = &args.trace_directory {
        model.forward_with_trace(&input, |name, value| {
            value.save_numpy(directory.join(format!("{name}.npy")))?;
            Ok(())
        })?
    } else {
        model.forward(&input)?
    };
    output.eval()?;
    output.save_numpy(&args.output)?;
    println!("input: {:?}, output: {:?}", input.shape(), output.shape());
    Ok(())
}
