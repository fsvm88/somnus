//! One place for the "read a float tensor back to the host as `Vec<f32>`" chain that used to be
//! spelled out ~50 times (`.into_data().convert::<f32>().try_to_vec::<f32>().unwrap()`).

use burn::tensor::Tensor;

/// Host readback of a float tensor.
pub trait TensorF32Ext {
    /// Row-major `Vec<f32>` of the tensor's elements (converting from the stored float dtype).
    ///
    /// Panics only if the backend cannot produce the data (a device/readback failure), which no
    /// caller can recover from mid-step.
    fn to_f32_vec(self) -> Vec<f32>;
}

impl<const D: usize> TensorF32Ext for Tensor<D> {
    fn to_f32_vec(self) -> Vec<f32> {
        self.into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .expect("float tensor readback")
    }
}
