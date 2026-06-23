use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::Router;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use clap::{Parser, ValueEnum};
use mlx_rs::{Array, Device};
use serde::Serialize;
use tokio::net::TcpListener;

use srs_inference::Renaissance;
use srs_inference::audio::{AudioBuffer, SAMPLE_RATE};
use srs_inference::spectral::SpectralTransform;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ComputeDevice {
    Cpu,
    Gpu,
}

#[derive(Debug, Parser)]
#[command(about = "SRS Enhancement Server - Model-resident HTTP inference service")]
struct Args {
    #[arg(short, long)]
    checkpoint: PathBuf,

    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    #[arg(long, default_value_t = 3000)]
    port: u16,

    #[arg(long, value_enum, default_value_t = ComputeDevice::Gpu)]
    device: ComputeDevice,
}

#[derive(Clone)]
struct AppState {
    worker_tx: Sender<InferenceRequest>,
    output_dir: Option<PathBuf>,
}

struct InferenceRequest {
    input_path: PathBuf,
    output_path: PathBuf,
    response: Sender<Result<InferenceResult, String>>,
}

struct InferenceResult {
    output_path: PathBuf,
    duration: f64,
    rtf: f64,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug)]
struct AppError(String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = serde_json::to_string(&ErrorResponse { error: self.0 }).unwrap_or_default();
        (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}

impl<E: std::fmt::Display> From<E> for AppError {
    fn from(err: E) -> Self {
        AppError(err.to_string())
    }
}

fn start_worker(checkpoint: PathBuf, device: ComputeDevice) -> Sender<InferenceRequest> {
    let (tx, rx) = mpsc::channel::<InferenceRequest>();
    thread::spawn(move || {
        let device = match device {
            ComputeDevice::Cpu => Device::cpu(),
            ComputeDevice::Gpu => Device::gpu(),
        };
        Device::set_default(&device);
        let mut model = match Renaissance::load(&checkpoint) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to load model: {e}");
                return;
            }
        };
        let spectral = SpectralTransform::new();
        eprintln!("Model loaded on {device}");

        while let Ok(req) = rx.recv() {
            let start = Instant::now();
            let result = process(&mut model, &spectral, &req.input_path, &req.output_path);
            let elapsed = start.elapsed();
            match result {
                Ok(audio_duration) => {
                    let rtf = elapsed.as_secs_f64() / audio_duration;
                    let _ = req.response.send(Ok(InferenceResult {
                        output_path: req.output_path,
                        duration: elapsed.as_secs_f64(),
                        rtf,
                    }));
                }
                Err(e) => {
                    let _ = req.response.send(Err(e));
                }
            }
        }
    });
    tx
}

fn process(
    model: &mut Renaissance,
    spectral: &SpectralTransform,
    input_path: &Path,
    output_path: &Path,
) -> Result<f64, String> {
    let decoded = AudioBuffer::load(input_path).map_err(|e| e.to_string())?;
    let mut waveform = decoded.preprocess().map_err(|e| e.to_string())?;
    let normalization_factor = waveform.normalize();
    let audio_duration = waveform.samples.len() as f64 / f64::from(SAMPLE_RATE);

    let input = Array::from_slice(&waveform.samples, &[1, waveform.samples.len() as i32]);
    let input_spectrum = spectral.stft(&input).map_err(|e| e.to_string())?;
    let enhanced_spectrum = model.forward(&input_spectrum).map_err(|e| e.to_string())?;
    enhanced_spectrum.eval().map_err(|e| e.to_string())?;

    let enhanced = spectral
        .istft(&enhanced_spectrum)
        .map_err(|e| e.to_string())?;
    let output_samples: Vec<f32> = enhanced
        .as_slice::<f32>()
        .iter()
        .map(|sample| sample * normalization_factor)
        .collect();

    let output = AudioBuffer {
        samples: output_samples,
        sample_rate: SAMPLE_RATE,
    };
    output
        .save_f32_wav(output_path)
        .map_err(|e| e.to_string())?;

    Ok(audio_duration)
}

async fn index() -> Html<&'static str> {
    Html(include_str!("web/index.html"))
}

async fn health() -> &'static str {
    "ok"
}

async fn enhance_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, AppError> {
    let mut input_path = None;
    let mut original_filename = "enhanced.wav".to_string();

    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" {
            if let Some(filename) = field.file_name() {
                original_filename = filename.to_string();
            }
            let data = field.bytes().await?;
            let id = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let temp = std::env::temp_dir().join(format!("srs_input_{}.wav", id));
            tokio::fs::write(&temp, &data).await?;
            input_path = Some(temp);
        }
    }

    let input_path = input_path.ok_or_else(|| AppError("No file uploaded".to_string()))?;
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let output_path = std::env::temp_dir().join(format!("srs_output_{}.wav", id));

    let (resp_tx, resp_rx) = mpsc::channel();
    let req = InferenceRequest {
        input_path: input_path.clone(),
        output_path: output_path.clone(),
        response: resp_tx,
    };

    state
        .worker_tx
        .send(req)
        .map_err(|e| AppError(e.to_string()))?;

    let result = tokio::task::spawn_blocking(move || resp_rx.recv())
        .await
        .map_err(|e| AppError(e.to_string()))?
        .map_err(|e| AppError(e.to_string()))?
        .map_err(AppError)?;

    let output_data = tokio::fs::read(&result.output_path).await?;

    if let Some(ref dir) = state.output_dir {
        let saved_name = format!("enhanced_{}", original_filename);
        let saved_path = dir.join(&saved_name);
        if let Err(e) = tokio::fs::copy(&result.output_path, &saved_path).await {
            eprintln!("Failed to copy result to {}: {e}", saved_path.display());
        } else {
            eprintln!("Saved result to {}", saved_path.display());
        }
    }

    let _ = tokio::fs::remove_file(&result.output_path).await;
    let _ = tokio::fs::remove_file(&input_path).await;

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "audio/wav".parse().unwrap());
    headers.insert(
        "X-Processing-Time",
        format!("{:.3}", result.duration).parse().unwrap(),
    );
    headers.insert("X-RTF", format!("{:.4}", result.rtf).parse().unwrap());
    headers.insert(
        "Content-Disposition",
        format!("attachment; filename=\"enhanced_{}\"", original_filename)
            .parse()
            .unwrap(),
    );

    Ok((headers, Body::from(output_data)).into_response())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let device = match args.device {
        ComputeDevice::Cpu => Device::cpu(),
        ComputeDevice::Gpu => Device::gpu(),
    };

    println!("Starting SRS Enhancement Server");
    println!("Checkpoint: {}", args.checkpoint.display());
    println!("Device: {device}");
    println!("Address: http://{}:{}", args.host, args.port);

    if let Some(ref dir) = args.output_dir {
        tokio::fs::create_dir_all(dir).await?;
        println!("Output directory: {}", dir.display());
    }

    let worker_tx = start_worker(args.checkpoint, args.device);

    let app = Router::new()
        .route("/", get(index))
        .route("/api/health", get(health))
        .route("/api/enhance", post(enhance_handler))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .with_state(AppState {
            worker_tx,
            output_dir: args.output_dir,
        });

    let listener = TcpListener::bind((args.host, args.port)).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
