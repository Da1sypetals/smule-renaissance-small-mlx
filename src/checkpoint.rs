use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::module::Param;
use mlx_rs::nn::Conv1d;

use crate::{Error, Result};

pub(crate) struct Checkpoint {
    tensors: HashMap<String, Array>,
}

impl Checkpoint {
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self> {
        let (tensors, metadata) = Array::load_safetensors_with_metadata(path)?;
        let layout = metadata.get("conv1d_weight_layout").map(String::as_str);
        if layout != Some("out_kernel_in_per_group") {
            return Err(Error::CheckpointMetadata(format!(
                "conv1d_weight_layout is {layout:?}, expected out_kernel_in_per_group"
            )));
        }
        if metadata.get("dtype").map(String::as_str) != Some("float32") {
            return Err(Error::CheckpointMetadata(
                "converted checkpoint must retain float32 weights".to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    pub(crate) fn tensor(&mut self, name: impl Into<String>, shape: &[i32]) -> Result<Array> {
        let name = name.into();
        let tensor = self
            .tensors
            .remove(&name)
            .ok_or_else(|| Error::MissingTensor(name.clone()))?;
        if tensor.shape() != shape {
            return Err(Error::TensorShape {
                name,
                actual: tensor.shape().to_vec(),
                expected: shape.to_vec(),
            });
        }
        Ok(tensor)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv1d(
        &mut self,
        prefix: &str,
        input_channels: i32,
        output_channels: i32,
        kernel_size: i32,
        bias: bool,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Result<Conv1d> {
        let weight = self.tensor(
            format!("{prefix}.weight"),
            &[output_channels, kernel_size, input_channels / groups],
        )?;
        let bias = if bias {
            Some(self.tensor(format!("{prefix}.bias"), &[output_channels])?)
        } else {
            None
        };
        Ok(Conv1d {
            weight: Param::new(weight),
            bias: Param::new(bias),
            stride: 1,
            padding,
            dilation,
            groups,
        })
    }

    pub(crate) fn finish(self) -> Result<()> {
        if self.tensors.is_empty() {
            return Ok(());
        }
        let mut names: Vec<_> = self.tensors.into_keys().collect();
        names.sort();
        Err(Error::UnexpectedTensors(names))
    }
}
