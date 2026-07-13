/// ORT-backed binding + encoder inference sessions for the perception pipeline.
///
/// Phase 1 (binding): replaces burn's GRU + cross-attention with an ORT call.
/// Phase 1.5 (encoder): replaces burn's 5 encoder matmuls with a single ORT call.
/// Topk selection and duty-cycle updates remain on CPU (Rust) between encoder and binding.
use std::path::Path;

use ndarray::{Array2, Array3};
use ort::{session::Session, value::TensorRef};

use crate::EPISODE_DIM;

const GRU_HIDDEN: usize = 256;
// Encoder input dimensions per region: [speech, env, visual, language]
const ENC_DIMS: [usize; 4] = [384, 512, 384, 896];

pub struct OnnxPerceptionSession {
    sess_bind: Session,
    sess_enc: Option<Session>, // Phase 1.5: encoder session
    pub gru_hidden: Vec<f32>,
}

impl OnnxPerceptionSession {
    /// Load binding session from `perception_binding.onnx`.
    /// `use_migraphx`: attach MIGraphX EP (AMD GPU).
    pub fn load(bind_path: &Path, use_migraphx: bool) -> anyhow::Result<Self> {
        let sess = if use_migraphx {
            migraphx_session(bind_path)?
        } else {
            cpu_session(bind_path)?
        };
        Ok(Self {
            sess_bind: sess,
            sess_enc: None,
            gru_hidden: vec![0.0f32; GRU_HIDDEN],
        })
    }

    /// Attach an encoder session from `perception_encoder.onnx` (Phase 1.5).
    pub fn with_encoder(mut self, enc_path: &Path, use_migraphx: bool) -> anyhow::Result<Self> {
        let sess = if use_migraphx {
            migraphx_session(enc_path)?
        } else {
            cpu_session(enc_path)?
        };
        self.sess_enc = Some(sess);
        Ok(self)
    }

    pub fn has_encoder(&self) -> bool {
        self.sess_enc.is_some()
    }

    /// Run the encoder graph (Phase 1.5).
    ///
    /// Inputs: raw embeddings for each region (zero-padded to correct dim by caller).
    /// Returns 5 pre-topk z vectors [N_UNITS] in region order [speech, env, visual, symbolic, lang].
    pub fn run_encoder(
        &mut self,
        speech: &[f32],
        env: &[f32],
        visual: &[f32],
        lang: &[f32],
    ) -> anyhow::Result<[Vec<f32>; 5]> {
        let sess = self
            .sess_enc
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("encoder session not loaded"))?;

        // Pad or truncate to expected dims
        let speech_v = pad_to(speech, ENC_DIMS[0]);
        let env_v = pad_to(env, ENC_DIMS[1]);
        let visual_v = pad_to(visual, ENC_DIMS[2]);
        let lang_v = pad_to(lang, ENC_DIMS[3]);

        let sp_arr = Array2::from_shape_vec((1, ENC_DIMS[0]), speech_v)?;
        let en_arr = Array2::from_shape_vec((1, ENC_DIMS[1]), env_v)?;
        let vi_arr = Array2::from_shape_vec((1, ENC_DIMS[2]), visual_v)?;
        let la_arr = Array2::from_shape_vec((1, ENC_DIMS[3]), lang_v)?;

        let t_sp = TensorRef::from_array_view(sp_arr.view())?;
        let t_en = TensorRef::from_array_view(en_arr.view())?;
        let t_vi = TensorRef::from_array_view(vi_arr.view())?;
        let t_la = TensorRef::from_array_view(la_arr.view())?;

        let outputs = sess.run(ort::inputs![
            "speech_enc"   => t_sp,
            "env_enc"      => t_en,
            "visual_enc"   => t_vi,
            "language_enc" => t_la,
        ])?;

        let extract = |name: &str| -> anyhow::Result<Vec<f32>> {
            let (_, data) = outputs[name].try_extract_tensor::<f32>()?;
            Ok(data.to_vec())
        };

        Ok([
            extract("z_0")?,
            extract("z_1")?,
            extract("z_2")?,
            extract("z_3")?,
            extract("z_4")?,
        ])
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
        // Shape from each vector's own length, not the baseline N_UNITS constant — the
        // ONNX graph's actual input dims match each region's size at export time, which
        // may be larger than baseline for a grown region (gaps.md §24).
        // Each ndarray must outlive the TensorRef that borrows it.
        let sp0 = Array2::from_shape_vec((1, sparse_z[0].len()), sparse_z[0].clone())?;
        let sp1 = Array2::from_shape_vec((1, sparse_z[1].len()), sparse_z[1].clone())?;
        let sp2 = Array2::from_shape_vec((1, sparse_z[2].len()), sparse_z[2].clone())?;
        let sp3 = Array2::from_shape_vec((1, sparse_z[3].len()), sparse_z[3].clone())?;
        let sp4 = Array2::from_shape_vec((1, sparse_z[4].len()), sparse_z[4].clone())?;
        let prev_arr = Array2::from_shape_vec((1, EPISODE_DIM), prev_ep.to_vec())?;
        let gru_arr = Array3::from_shape_vec((1, 1, GRU_HIDDEN), self.gru_hidden.clone())?;

        let t_sp0 = TensorRef::from_array_view(sp0.view())?;
        let t_sp1 = TensorRef::from_array_view(sp1.view())?;
        let t_sp2 = TensorRef::from_array_view(sp2.view())?;
        let t_sp3 = TensorRef::from_array_view(sp3.view())?;
        let t_sp4 = TensorRef::from_array_view(sp4.view())?;
        let t_prev = TensorRef::from_array_view(prev_arr.view())?;
        let t_gru = TensorRef::from_array_view(gru_arr.view())?;

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

/// Truncate or zero-pad `src` to exactly `n` elements.
fn pad_to(src: &[f32], n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    let copy = src.len().min(n);
    v[..copy].copy_from_slice(&src[..copy]);
    v
}
