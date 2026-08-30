use candle_core::{DType, Device};

use crate::error::{Error, Result};

/// Numeric precision requested by the caller.
///
/// The first three map straight onto a candle [`DType`]. `Int8` is different in
/// kind: activations still flow in 16-bit, and only the large weight matrices
/// are stored as 8-bit blocks, so it carries a companion "activation dtype".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    F16,
    BF16,
    /// 8-bit block-quantised weights (Q8_0) with 16-bit activations.
    Int8,
}

impl Precision {
    /// Parse the spellings people actually type.
    pub fn parse(name: &str) -> Result<Self> {
        match name.trim().to_ascii_lowercase().replace(['-', '_'], "") {
            s if s == "fp32" || s == "f32" || s == "float32" || s == "full" => Ok(Self::F32),
            s if s == "fp16" || s == "f16" || s == "float16" || s == "half" => Ok(Self::F16),
            s if s == "bf16" || s == "bfloat16" => Ok(Self::BF16),
            s if s == "int8" || s == "i8" || s == "q8" || s == "q80" => Ok(Self::Int8),
            other => Err(Error::UnsupportedDtype(other)),
        }
    }

    /// The dtype activations and non-quantised weights are held in.
    pub fn activation_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
            // Q8_0 dequantises into F16, so keeping activations in F16 avoids a
            // cast on every matmul.
            Self::Int8 => DType::F16,
        }
    }

    pub fn is_quantized(self) -> bool {
        matches!(self, Self::Int8)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "fp32",
            Self::F16 => "fp16",
            Self::BF16 => "bf16",
            Self::Int8 => "int8",
        }
    }

    /// Reject combinations the device cannot execute.
    ///
    /// CPU BF16 support in candle exists but is emulated and ruinously slow, and
    /// the quantised kernels are only worth using where they were tuned.
    pub fn validate_for(self, device: &Device) -> Result<()> {
        if device.is_cpu() && matches!(self, Self::F16 | Self::BF16) {
            return Err(Error::config(format!(
                "{} is not usable on CPU (candle emulates it in software); use fp32",
                self.as_str()
            )));
        }
        Ok(())
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Precision {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

/// Pick the best precision the hardware supports when the caller did not choose.
pub fn default_precision(device: &Device) -> Precision {
    if device.is_cpu() {
        Precision::F32
    } else {
        // BF16 matches the checkpoint's stored dtype, so it is both the fastest
        // and the most faithful default on any Ampere-or-newer GPU.
        Precision::BF16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_spellings() {
        for (input, want) in [
            ("fp32", Precision::F32),
            ("F32", Precision::F32),
            ("float32", Precision::F32),
            ("fp16", Precision::F16),
            ("half", Precision::F16),
            ("bf16", Precision::BF16),
            ("bfloat16", Precision::BF16),
            ("int8", Precision::Int8),
            ("q8_0", Precision::Int8),
        ] {
            assert_eq!(Precision::parse(input).unwrap(), want, "input {input}");
        }
        assert!(Precision::parse("fp8").is_err());
    }

    #[test]
    fn int8_runs_activations_in_f16() {
        assert_eq!(Precision::Int8.activation_dtype(), DType::F16);
        assert!(Precision::Int8.is_quantized());
        assert!(!Precision::BF16.is_quantized());
    }

    #[test]
    fn cpu_rejects_reduced_precision() {
        let cpu = Device::Cpu;
        assert!(Precision::F32.validate_for(&cpu).is_ok());
        assert!(Precision::F16.validate_for(&cpu).is_err());
        assert!(Precision::BF16.validate_for(&cpu).is_err());
    }
}
