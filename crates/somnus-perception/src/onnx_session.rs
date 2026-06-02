/// ORT-backed binding inference session for the perception pipeline (Phase 1).
///
/// Replaces burn's binding area (GRU + cross-attention) with an ORT call.
/// Burn k-WTA layers still run for duty-cycle tracking; their dense sparse activations
/// are passed into the ORT binding graph. Phase 1.5 adds the encoder session.
use std::path::Path;

use ndarray::{Array2, Array3};
use ort::{session::Session, value::TensorRef};

use crate::{EPISODE_DIM, N_UNITS};

const GRU_HIDDEN: usize = 256;

pub struct OnnxPerceptionSession {
    sess_bind: Session,
    pub gru_hidden: Vec<f32>,
}

impl OnnxPerceptionSession {
    /// Load from an existing `perception_binding.onnx`.
    /// `use_migraphx`: attach MIGraphX EP (AMD GPU).
    pub fn load(bind_path: &Path, use_migraphx: bool) -> anyhow::Result<Self> {
        let sess = if use_migraphx {
            migraphx_session(bind_path)?
        } else {
            cpu_session(bind_path)?
        };
        Ok(Self {
            sess_bind: sess,
            gru_hidden: vec![0.0f32; GRU_HIDDEN],
        })
    }

    /// Run the binding graph.
    ///
    /// `sparse_z`:   5 dense [N_UNITS] vectors from burn k-WTA → dense conversion
    /// `prev_ep`:    previous episode [EPISODE_DIM]; zeros for first call
    ///
    /// Returns new episode [EPISODE_DIM]. Updates `self.gru_hidden`.
    pub fn run_binding(
        &mut self,
        sparse_z: &[Vec<f32>; 5],
        prev_ep: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        // Each ndarray must outlive the TensorRef that borrows it.
        let sp0 = Array2::from_shape_vec((1, N_UNITS), sparse_z[0].clone())?;
        let sp1 = Array2::from_shape_vec((1, N_UNITS), sparse_z[1].clone())?;
        let sp2 = Array2::from_shape_vec((1, N_UNITS), sparse_z[2].clone())?;
        let sp3 = Array2::from_shape_vec((1, N_UNITS), sparse_z[3].clone())?;
        let sp4 = Array2::from_shape_vec((1, N_UNITS), sparse_z[4].clone())?;
        let prev_arr = Array2::from_shape_vec((1, EPISODE_DIM), prev_ep.to_vec())?;
        let gru_arr  = Array3::from_shape_vec((1, 1, GRU_HIDDEN), self.gru_hidden.clone())?;

        let t_sp0  = TensorRef::from_array_view(sp0.view())?;
        let t_sp1  = TensorRef::from_array_view(sp1.view())?;
        let t_sp2  = TensorRef::from_array_view(sp2.view())?;
        let t_sp3  = TensorRef::from_array_view(sp3.view())?;
        let t_sp4  = TensorRef::from_array_view(sp4.view())?;
        let t_prev = TensorRef::from_array_view(prev_arr.view())?;
        let t_gru  = TensorRef::from_array_view(gru_arr.view())?;

        let outputs = self.sess_bind.run(ort::inputs![
            "sp_0"         => t_sp0,
            "sp_1"         => t_sp1,
            "sp_2"         => t_sp2,
            "sp_3"         => t_sp3,
            "sp_4"         => t_sp4,
            "prev_episode" => t_prev,
            "gru_hidden"   => t_gru,
        ])?;

        let (_, ep_data) = outputs["episode"].try_extract_tensor::<f32>()?;
        let episode: Vec<f32> = ep_data.to_vec();

        let (_, h_data) = outputs["gru_hidden_out"].try_extract_tensor::<f32>()?;
        self.gru_hidden = h_data.to_vec();

        Ok(episode)
    }

    pub fn reset_gru(&mut self) {
        self.gru_hidden.fill(0.0);
    }
}

/// MIGraphX EP session via raw C API (matches migraphx_session in crates/somnus/src/main.rs).
fn migraphx_session(path: &Path) -> anyhow::Result<Session> {
    use ort::AsPointer;
    let (intra, inter) = thread_budget();
    let mut builder = Session::builder()?
        .with_intra_threads(intra)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_inter_threads(inter)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let name = std::ffi::CString::new("MIGraphXExecutionProvider").unwrap();
    let status = unsafe {
        (ort::api().SessionOptionsAppendExecutionProvider)(
            builder.ptr_mut(),
            name.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0,
        )
    };
    if !status.0.is_null() {
        unsafe { (ort::api().ReleaseStatus)(status.0) };
    }
    Ok(builder.commit_from_file(path)?)
}

fn cpu_session(path: &Path) -> anyhow::Result<Session> {
    let (intra, inter) = thread_budget();
    Ok(Session::builder()?
        .with_intra_threads(intra)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_inter_threads(inter)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(path)?)
}

fn thread_budget() -> (usize, usize) {
    let n = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    ((n / 2).max(1), (n / 4).max(1))
}
