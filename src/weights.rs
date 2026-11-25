//! text-to-speech safetensors (tools/convert.py output) into burn tensors. f32 or f16 on disk, f32 here.
use burn::tensor::{Device, Tensor, TensorData};
use safetensors::{tensor::Dtype, SafeTensors};

pub(crate) struct Weights<'a> {
    st: SafeTensors<'a>,
    dev: Device,
}

impl<'a> Weights<'a> {
    pub(crate) fn new(bytes: &'a [u8], dev: &Device) -> Result<Self, String> {
        Ok(Weights { st: SafeTensors::deserialize(bytes).map_err(|e| e.to_string())?, dev: dev.clone() })
    }

    fn raw(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let t = self.st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
        let d = t.data();
        let v: Vec<f32> = match t.dtype() {
            Dtype::F32 => d.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            Dtype::F16 => d.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
            Dtype::BF16 => d.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
            other => return Err(format!("{name}: dtype {other:?}")),
        };
        Ok((v, t.shape().to_vec()))
    }

    pub(crate) fn t<const D: usize>(&self, name: &str) -> Result<Tensor<D>, String> {
        let (v, s) = self.raw(name)?;
        if s.len() != D {
            return Err(format!("{name}: rank {} not {D}", s.len()));
        }
        Ok(Tensor::<D>::from_data(TensorData::new(v, s), &self.dev))
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        self.st.tensor(name).is_ok()
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    #[test]
    fn f16() {
        assert_eq!(super::f16_to_f32(0x3c00), 1.0);
        assert_eq!(super::f16_to_f32(0xc000), -2.0);
        assert_eq!(super::f16_to_f32(0x0000), 0.0);
        assert!((super::f16_to_f32(0x0001) - 5.96e-8).abs() < 1e-9);
    }
}
