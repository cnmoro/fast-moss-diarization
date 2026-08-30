//! The bridge from Whisper encoder frames to language-model embeddings.

use candle_core::Tensor;

use crate::error::Result;
use crate::model::linear::{LayerNorm, Linear, Loader};

/// `Linear -> SiLU -> Linear -> LayerNorm`, projecting merged audio frames into
/// the text backbone's residual stream.
pub struct VqAdaptor {
    fc1: Linear,
    fc2: Linear,
    norm: LayerNorm,
    merge_size: usize,
}

impl VqAdaptor {
    pub fn load(
        loader: &Loader,
        input_dim: usize,
        hidden_size: usize,
        merge_size: usize,
        norm_eps: f64,
    ) -> Result<Self> {
        let vb = loader.sub("model.vq_adaptor");
        Ok(Self {
            // Kept dense under int8: this is the single point where audio enters
            // the text stream, it is tiny next to the backbone, and quantising
            // it perturbs every downstream token.
            fc1: vb.linear_dense(input_dim, hidden_size, "layers.0", true)?,
            fc2: vb.linear_dense(hidden_size, hidden_size, "layers.2", true)?,
            norm: LayerNorm::load(&vb, hidden_size, "layers.3", norm_eps)?,
            merge_size,
        })
    }

    /// Concatenate every `merge_size` consecutive frames along the feature axis:
    /// `(b, t, d) -> (b, t / merge_size, d * merge_size)`.
    ///
    /// Frames beyond the last whole group are dropped, matching the reference.
    pub fn time_merge(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, d) = xs.dims3()?;
        let groups = t / self.merge_size;
        let trimmed = if groups * self.merge_size == t {
            xs.clone()
        } else {
            xs.narrow(1, 0, groups * self.merge_size)?
        };
        Ok(trimmed
            .contiguous()?
            .reshape((b, groups, d * self.merge_size))?)
    }

    pub fn forward(&self, merged: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(merged)?;
        let h = self.fc2.forward(&candle_nn::ops::silu(&h)?)?;
        self.norm.forward(&h)
    }

    /// Merge and project in one step.
    pub fn merge_and_project(&self, frames: &Tensor) -> Result<Tensor> {
        let merged = self.time_merge(frames)?;
        self.forward(&merged)
    }

    pub fn merge_size(&self) -> usize {
        self.merge_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn adaptor(merge_size: usize) -> VqAdaptor {
        // Only `time_merge` is exercised here, so the projections can be dummies.
        let dev = Device::Cpu;
        VqAdaptor {
            fc1: Linear::Dense {
                weight: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
                bias: None,
            },
            fc2: Linear::Dense {
                weight: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
                bias: None,
            },
            norm: LayerNorm {
                weight: Tensor::zeros(1, DType::F32, &dev).unwrap(),
                bias: Tensor::zeros(1, DType::F32, &dev).unwrap(),
                eps: 1e-6,
            },
            merge_size,
        }
    }

    #[test]
    fn merging_folds_frames_into_features() {
        let dev = Device::Cpu;
        // One batch, 4 frames, 2 features: [[0,1],[2,3],[4,5],[6,7]]
        let xs = Tensor::from_vec(
            (0..8).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 4, 2),
            &dev,
        )
        .unwrap();
        let merged = adaptor(4).time_merge(&xs).unwrap();
        assert_eq!(merged.dims(), &[1, 1, 8]);
        assert_eq!(
            merged.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            (0..8).map(|v| v as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_full_chunk_merges_1500_frames_into_375_tokens() {
        let dev = Device::Cpu;
        let xs = Tensor::zeros((2, 1500, 16), DType::F32, &dev).unwrap();
        assert_eq!(adaptor(4).time_merge(&xs).unwrap().dims(), &[2, 375, 64]);
    }

    #[test]
    fn trailing_partial_group_is_dropped() {
        let dev = Device::Cpu;
        let xs = Tensor::zeros((1, 7, 2), DType::F32, &dev).unwrap();
        // 7 frames at merge 4 -> one whole group, three frames discarded.
        assert_eq!(adaptor(4).time_merge(&xs).unwrap().dims(), &[1, 1, 8]);
    }
}
