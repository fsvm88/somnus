/// ORT-backed binding + encoder inference sessions for the perception pipeline.
///
/// Phase 1 (binding): replaces burn's GRU + cross-attention with an ORT call.
/// Phase 1.5 (encoder): replaces burn's 5 encoder matmuls with a single ORT call.
/// Topk selection and duty-cycle updates remain on CPU (Rust) between encoder and binding.
use std::path::Path;

use ndarray::{Array2, Array3};
use ort::{session::Session, value::TensorRef};

use crate::region::SparseActivation;
use crate::{ENV_DIM, EPISODE_DIM, LANG_HIDDEN};

const GRU_HIDDEN: usize = 256;
// Encoder input dimensions per region: [speech, env, visual, language]. Derived from the
// live constants, not a second hardcoded copy — a stale copy here (896 vs. LANG_HIDDEN=1024
// after ADR 0036) previously caused every encoder call to fail with a language_enc dimension
// mismatch (found 2026-07-13 resuming the growth campaign after gaps.md §24's fix).
const ENC_DIMS: [usize; 4] = [384, ENV_DIM, 384, LANG_HIDDEN];

pub struct OnnxPerceptionSession {
    sess_bind: Session,
    sess_enc: Option<Session>, // Phase 1.5: encoder session
    pub gru_hidden: Vec<f32>,
    /// Per-region active-unit count baked into the binding graph's sparse-gather
    /// inputs at export time ([speech, env, visual, symbolic, language] order).
    /// Read from `weights/weights_manifest.json` (sibling to `bind_path`) at load
    /// time — never recomputed from n_units in this Rust caller either, since
    /// Rust's own `KWtaLayer::k_active` field (not `scaled_k(n_units)`) is what
    /// the exported graph's shape actually reflects.
    k_active: [usize; 5],
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
        let k_active = load_k_active(bind_path)?;
        Ok(Self {
            sess_bind: sess,
            sess_enc: None,
            gru_hidden: vec![0.0f32; GRU_HIDDEN],
            k_active,
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
        let (z, _) = somnus_metrics::timed_core!(somnus_metrics::Metric::OrtRunEncoder, {
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

            [
                extract("z_0")?,
                extract("z_1")?,
                extract("z_2")?,
                extract("z_3")?,
                extract("z_4")?,
            ]
        });
        Ok(z)
    }

    /// Run the binding graph.
    ///
    /// `sparse`:   5 region k-WTA activations, in [speech, env, visual, symbolic,
    ///             language] order. Each is padded/truncated to this session's
    ///             per-region `k_active` (see `SparseActivation::to_exact_k`) to
    ///             match the static shape the binding graph's Gather inputs were
    ///             exported with — callers do not need to pad themselves.
    /// `prev_ep`:  previous episode [EPISODE_DIM]; zeros for first call
    ///
    /// Returns new episode [EPISODE_DIM]. Updates `self.gru_hidden`.
    pub fn run_binding(
        &mut self,
        sparse: &[SparseActivation; 5],
        prev_ep: &[f32],
    ) -> anyhow::Result<Vec<f32>> {
        let (episode, _) = somnus_metrics::timed_core!(somnus_metrics::Metric::OrtRunBinding, {
            let exact: Vec<(Vec<i64>, Vec<f32>)> = (0..5)
                .map(|i| sparse[i].to_exact_k(self.k_active[i]))
                .collect();

            // Each ndarray must outlive the TensorRef that borrows it.
            let idx_arrs: Vec<ndarray::Array1<i64>> = exact
                .iter()
                .map(|(idx, _)| ndarray::Array1::from_vec(idx.clone()))
                .collect();
            let val_arrs: Vec<Array2<f32>> = exact
                .iter()
                .map(|(_, val)| Array2::from_shape_vec((1, val.len()), val.clone()))
                .collect::<Result<_, _>>()?;
            let prev_arr = Array2::from_shape_vec((1, EPISODE_DIM), prev_ep.to_vec())?;
            let gru_arr = Array3::from_shape_vec((1, 1, GRU_HIDDEN), self.gru_hidden.clone())?;

            let t_idx: Vec<TensorRef<'_, i64>> = idx_arrs
                .iter()
                .map(|a| TensorRef::from_array_view(a.view()))
                .collect::<Result<_, _>>()?;
            let t_val: Vec<TensorRef<'_, f32>> = val_arrs
                .iter()
                .map(|a| TensorRef::from_array_view(a.view()))
                .collect::<Result<_, _>>()?;
            let [i0, i1, i2, i3, i4]: [TensorRef<'_, i64>; 5] = t_idx
                .try_into()
                .map_err(|_| anyhow::anyhow!("expected 5 active_idx tensors"))?;
            let [v0, v1, v2, v3, v4]: [TensorRef<'_, f32>; 5] = t_val
                .try_into()
                .map_err(|_| anyhow::anyhow!("expected 5 active_val tensors"))?;
            let t_prev = TensorRef::from_array_view(prev_arr.view())?;
            let t_gru = TensorRef::from_array_view(gru_arr.view())?;

            let outputs = self.sess_bind.run(ort::inputs![
                "active_idx_0" => i0,
                "active_idx_1" => i1,
                "active_idx_2" => i2,
                "active_idx_3" => i3,
                "active_idx_4" => i4,
                "active_val_0" => v0,
                "active_val_1" => v1,
                "active_val_2" => v2,
                "active_val_3" => v3,
                "active_val_4" => v4,
                "prev_episode" => t_prev,
                "gru_hidden"   => t_gru,
            ])?;

            let (_, ep_data) = outputs["episode"].try_extract_tensor::<f32>()?;
            let episode: Vec<f32> = ep_data.to_vec();

            let (_, h_data) = outputs["gru_hidden_out"].try_extract_tensor::<f32>()?;
            self.gru_hidden = h_data.to_vec();

            episode
        });
        Ok(episode)
    }

    pub fn reset_gru(&mut self) {
        self.gru_hidden.fill(0.0);
    }
}

/// Read the per-region `k_active` list written by
/// `PerceptionProcessor::export_weights_for_onnx` into
/// `weights_manifest.json`, from the `weights/` dir sibling to the binding
/// graph's own directory (same lookup pattern as `with_encoder`'s sibling
/// `perception_encoder.onnx` search).
fn load_k_active(bind_path: &Path) -> anyhow::Result<[usize; 5]> {
    let manifest_path = bind_path
        .parent()
        .unwrap_or(Path::new("."))
        .join("weights")
        .join("weights_manifest.json");
    let text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_str(&text)?;
    let k_active: Vec<usize> = serde_json::from_value(
        manifest
            .get("k_active")
            .ok_or_else(|| anyhow::anyhow!("weights_manifest.json missing \"k_active\""))?
            .clone(),
    )?;
    k_active
        .try_into()
        .map_err(|v: Vec<usize>| anyhow::anyhow!("expected 5 k_active entries, got {}", v.len()))
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
