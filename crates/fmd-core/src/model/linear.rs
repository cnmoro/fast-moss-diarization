//! Precision-polymorphic linear layers and the weight loader behind them.

use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;

use crate::error::Result;
use crate::precision::Precision;

/// A `y = x W^T + b` projection, stored either densely or as 8-bit blocks.
pub enum Linear {
    Dense {
        weight: Tensor,
        bias: Option<Tensor>,
    },
    Quantized {
        matmul: QMatMul,
        bias: Option<Tensor>,
    },
}

impl Linear {
    /// Apply the projection to an input whose last dimension is the input width.
    ///
    /// Inputs of any rank are flattened to 2-D first: the quantised kernels only
    /// define a matrix path, and doing it here keeps both branches identical.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims();
        let (flat, restore) = if dims.len() == 2 {
            (xs.clone(), None)
        } else {
            let in_dim = dims[dims.len() - 1];
            let rows: usize = dims[..dims.len() - 1].iter().product();
            (xs.reshape((rows, in_dim))?, Some(dims.to_vec()))
        };

        let out = match self {
            Self::Dense { weight, bias } => {
                let y = flat.matmul(&weight.t()?)?;
                match bias {
                    Some(b) => y.broadcast_add(b)?,
                    None => y,
                }
            }
            Self::Quantized { matmul, bias } => {
                let y = matmul.forward(&flat)?;

                match bias {
                    Some(b) => y.broadcast_add(&b.to_dtype(y.dtype())?)?,
                    None => y,
                }
            }
        };

        match restore {
            None => Ok(out),
            Some(mut shape) => {
                let out_dim = out.dim(out.rank() - 1)?;
                *shape.last_mut().expect("rank >= 2") = out_dim;
                Ok(out.reshape(shape)?)
            }
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }
}

/// Several projections over the same input, merged into one matmul.
///
/// Q/K/V and the MLP's gate/up pairs all read the same activation, so stacking
/// their weights turns three (or two) launches into one and gives the GEMM a
/// wider output to work with. At batch 1 the decode loop is launch-bound, so
/// this is worth more than it looks.
pub struct FusedLinear {
    inner: Linear,
    splits: Vec<usize>,
}

impl FusedLinear {
    /// Apply the merged projection and split the result back apart.
    pub fn forward(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let joined = self.inner.forward(xs)?;
        let last = joined.rank() - 1;
        let mut out = Vec::with_capacity(self.splits.len());
        let mut offset = 0;
        for &width in &self.splits {
            out.push(joined.narrow(last, offset, width)?);
            offset += width;
        }
        Ok(out)
    }

    pub fn splits(&self) -> &[usize] {
        &self.splits
    }
}

/// Loads tensors out of the checkpoint at the right precision.
///
/// Two views over the same mmapped safetensors are kept: `dense` yields the
/// activation dtype for everything that stays in floating point, and `quant_src`
/// yields f32 so quantisation reads the checkpoint's full mantissa rather than a
/// down-converted copy.
pub struct Loader<'a> {
    dense: VarBuilder<'a>,
    quant_src: Option<VarBuilder<'a>>,
    precision: Precision,
    device: Device,
}

impl<'a> Loader<'a> {
    /// # Safety
    /// The safetensors files are memory-mapped; they must not be modified while
    /// the loader or any tensor it produced is alive.
    pub unsafe fn from_safetensors(
        paths: &[std::path::PathBuf],
        precision: Precision,
        device: &Device,
    ) -> Result<Self> {
        let dtype = precision.activation_dtype();
        let dense = VarBuilder::from_mmaped_safetensors(paths, dtype, device)?;
        let quant_src = if precision.is_quantized() {
            Some(VarBuilder::from_mmaped_safetensors(
                paths,
                DType::F32,
                device,
            )?)
        } else {
            None
        };
        Ok(Self {
            dense,
            quant_src,
            precision,
            device: device.clone(),
        })
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.precision.activation_dtype()
    }

    /// Descend into a sub-module namespace.
    pub fn sub(&self, prefix: &str) -> Loader<'a> {
        Loader {
            dense: self.dense.pp(prefix),
            quant_src: self.quant_src.as_ref().map(|vb| vb.pp(prefix)),
            precision: self.precision,
            device: self.device.clone(),
        }
    }

    /// A raw tensor in the activation dtype.
    pub fn get(&self, shape: impl Into<candle_core::Shape>, name: &str) -> Result<Tensor> {
        Ok(self.dense.get(shape, name)?)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.dense.contains_tensor(name)
    }

    /// Load a projection, quantising it when the caller opts in *and* the
    /// engine is running in int8.
    ///
    /// `quantizable` is a policy switch, not a capability one: some layers are
    /// deliberately left dense even in int8 mode (see [`Self::linear`] callers).
    pub fn linear_with(
        &self,
        in_dim: usize,
        out_dim: usize,
        name: &str,
        bias: bool,
        quantizable: bool,
    ) -> Result<Linear> {
        let bias = if bias {
            Some(self.dense.get(out_dim, &format!("{name}.bias"))?)
        } else {
            None
        };
        let weight_name = format!("{name}.weight");

        match (&self.quant_src, quantizable) {
            (Some(src), true) => {
                let w = src.get((out_dim, in_dim), &weight_name)?;
                let qt = QTensor::quantize(&w, GgmlDType::Q8_0)?;
                Ok(Linear::Quantized {
                    matmul: QMatMul::from_arc(Arc::new(qt))?,
                    bias,
                })
            }
            _ => Ok(Linear::Dense {
                weight: self.dense.get((out_dim, in_dim), &weight_name)?,
                bias,
            }),
        }
    }

    /// A projection that follows the engine's default quantisation policy.
    pub fn linear(&self, in_dim: usize, out_dim: usize, name: &str, bias: bool) -> Result<Linear> {
        self.linear_with(in_dim, out_dim, name, bias, true)
    }

    /// Merge several same-input projections into one [`FusedLinear`].
    ///
    /// Weights are concatenated along the output axis, which is safe for Q8_0:
    /// its quantisation blocks run along the *input* axis, so stacking rows
    /// neither splits nor merges a block.
    pub fn fused_linear(
        &self,
        in_dim: usize,
        parts: &[(&str, usize)],
        bias: bool,
        quantizable: bool,
    ) -> Result<FusedLinear> {
        let splits: Vec<usize> = parts.iter().map(|(_, w)| *w).collect();

        let bias = if bias {
            let biases = parts
                .iter()
                .map(|(name, out)| self.dense.get(*out, &format!("{name}.bias")))
                .collect::<candle_core::Result<Vec<_>>>()?;
            Some(Tensor::cat(&biases, 0)?)
        } else {
            None
        };

        let inner = match (&self.quant_src, quantizable) {
            (Some(src), true) => {
                let weights = parts
                    .iter()
                    .map(|(name, out)| src.get((*out, in_dim), &format!("{name}.weight")))
                    .collect::<candle_core::Result<Vec<_>>>()?;
                let joined = Tensor::cat(&weights, 0)?;
                let qt = QTensor::quantize(&joined, GgmlDType::Q8_0)?;
                Linear::Quantized {
                    matmul: QMatMul::from_arc(Arc::new(qt))?,
                    bias,
                }
            }
            _ => {
                let weights = parts
                    .iter()
                    .map(|(name, out)| self.dense.get((*out, in_dim), &format!("{name}.weight")))
                    .collect::<candle_core::Result<Vec<_>>>()?;
                Linear::Dense {
                    weight: Tensor::cat(&weights, 0)?,
                    bias,
                }
            }
        };

        Ok(FusedLinear { inner, splits })
    }

    /// A projection that always stays in floating point.
    pub fn linear_dense(
        &self,
        in_dim: usize,
        out_dim: usize,
        name: &str,
        bias: bool,
    ) -> Result<Linear> {
        self.linear_with(in_dim, out_dim, name, bias, false)
    }
}

/// RMS normalisation with a learned gain (Qwen3's norm).
pub struct RmsNorm {
    pub(crate) weight: Tensor,
    pub(crate) eps: f64,
}

impl RmsNorm {
    pub fn load(loader: &Loader, dim: usize, name: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: loader.get(dim, &format!("{name}.weight"))?,
            eps,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // The fused kernel accumulates the sum of squares in f32 internally, so
        // there is no need to widen the tensor first -- and widening would be
        // expensive: this runs 113 times per generated token.
        Ok(candle_nn::ops::rms_norm(
            &xs.contiguous()?,
            &self.weight,
            self.eps as f32,
        )?)
    }
}

/// Standard layer normalisation with weight and bias (Whisper's norm).
pub struct LayerNorm {
    pub(crate) weight: Tensor,
    pub(crate) bias: Tensor,
    pub(crate) eps: f64,
}

impl LayerNorm {
    pub fn load(loader: &Loader, dim: usize, name: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: loader.get(dim, &format!("{name}.weight"))?,
            bias: loader.get(dim, &format!("{name}.bias"))?,
            eps,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // As with `RmsNorm`, the fused kernel already reduces in f32.
        Ok(candle_nn::ops::layer_norm(
            &xs.contiguous()?,
            &self.weight,
            &self.bias,
            self.eps as f32,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn dense_linear_matches_a_hand_computed_product() -> Result<()> {
        let dev = Device::Cpu;
        // y = x W^T + b with W = [[1,2],[3,4]], b = [10, 20]
        let weight = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &dev)?;
        let bias = Tensor::from_vec(vec![10f32, 20.], 2, &dev)?;
        let lin = Linear::Dense {
            weight,
            bias: Some(bias),
        };
        let x = Tensor::from_vec(vec![1f32, 1.], (1, 2), &dev)?;
        let y = lin.forward(&x)?.to_vec2::<f32>()?;
        assert_eq!(y, vec![vec![13.0, 27.0]]);
        Ok(())
    }

    #[test]
    fn linear_preserves_leading_dimensions() -> Result<()> {
        let dev = Device::Cpu;
        let lin = Linear::Dense {
            weight: Tensor::zeros((7, 3), DType::F32, &dev)?,
            bias: None,
        };
        let x = Tensor::zeros((2, 5, 3), DType::F32, &dev)?;
        assert_eq!(lin.forward(&x)?.dims(), &[2, 5, 7]);
        assert!(!lin.is_quantized());
        Ok(())
    }

    #[test]
    fn layer_norm_standardises_then_rescales() -> Result<()> {
        let dev = Device::Cpu;
        let ln = LayerNorm {
            weight: Tensor::from_vec(vec![1f32; 4], 4, &dev)?,
            bias: Tensor::from_vec(vec![0f32; 4], 4, &dev)?,
            eps: 1e-5,
        };
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 4), &dev)?;
        let y = ln.forward(&x)?.to_vec2::<f32>()?.remove(0);
        let mean: f32 = y.iter().sum::<f32>() / 4.0;
        let var: f32 = y.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4, "mean was {mean}");
        assert!((var - 1.0).abs() < 1e-3, "var was {var}");
        Ok(())
    }
}
