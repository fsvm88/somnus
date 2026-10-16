//! Async wake-reconstruction worker (perf opt, 2026-06).
//!
//! `wake_reconstruct_step` only trains the 5 output heads (gradients flow
//! exclusively through `*_head.forward`), and the wake inference path reads
//! only the flat EMA weights. That makes the blocking backward pass (5–15ms
//! on high-salience episodes) safe to offload: the worker owns a copy of the
//! heads plus its own AdamW and EMA state, streams updated flat EMA weights
//! back non-blockingly, and merges head weights with the main model only at
//! sleep boundaries.
//!
//! Weight-sync protocol with the main thread:
//! - wake: `request_train` (non-blocking, drops when busy) + `try_recv_ema`
//!   polled each loop iteration
//! - sleep entry: `fetch_weights` (blocking) → caller writes them into the
//!   main model so NREM head training continues from the wake-trained weights
//! - wake re-entry: `set_weights` pushes the NREM-trained heads back. Worker
//!   AdamW moments survive but refer to pre-sleep weights — stale by one
//!   sleep cycle, acceptable here.

use crate::tensor_ext::TensorF32Ext;
use std::sync::mpsc;

use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    optim::{AdamWConfig, GradientsParams, ModuleOptimizer},
    tensor::{Device, Tensor, TensorData},
};

use crate::{EmaOutputHeads, EPISODE_DIM, REGION_DIMS};

/// Standalone copy of the 5 output heads. Field order mirrors REGION_DIMS:
/// [AudioSpeech, AudioEnv, Visual, Symbolic, Language].
#[derive(Module, Debug)]
pub(crate) struct WakeHeadsModel {
    speech: Linear,
    env: Linear,
    visual: Linear,
    symbolic: Linear,
    language: Linear,
}

type WakeOptim = ModuleOptimizer;

/// Overwrite a head's weight from a flat [EPISODE_DIM × d_out] vec, preserving
/// the ParamId so the worker's AdamW moments stay attached.
fn set_linear_weight(linear: &mut Linear, flat: &[f32], d_out: usize, device: &Device) {
    let id = linear.weight.id;
    // require_grad is mandatory: Param::initialized inherits the tensor's flag,
    // and from_data tensors don't track gradients — without it the head is
    // silently frozen (no grads → AdamW no-op).
    let t: Tensor<2> =
        Tensor::from_data(TensorData::new(flat.to_vec(), [EPISODE_DIM, d_out]), device)
            .require_grad();
    linear.weight = Param::initialized(id, t);
}

fn heads_from_flat(weights: &[Vec<f32>; 5], device: &Device) -> WakeHeadsModel {
    let mk = |flat: &Vec<f32>, d_out: usize| {
        let mut l = LinearConfig::new(EPISODE_DIM, d_out)
            .with_bias(false)
            .init(device);
        set_linear_weight(&mut l, flat, d_out, device);
        l
    };
    WakeHeadsModel {
        speech: mk(&weights[0], REGION_DIMS[0]),
        env: mk(&weights[1], REGION_DIMS[1]),
        visual: mk(&weights[2], REGION_DIMS[2]),
        symbolic: mk(&weights[3], REGION_DIMS[3]),
        language: mk(&weights[4], REGION_DIMS[4]),
    }
}

/// One training step: per-region MSE → single backward → AdamW → EMA update.
/// Mirrors `PerceptionProcessor::wake_reconstruct_step`.
fn train_step(
    model: WakeHeadsModel,
    optim: &mut WakeOptim,
    ema: &mut EmaOutputHeads,
    episode: &[f32],
    regional: &[Vec<f32>; 5],
    lr: f64,
    device: &Device,
) -> (WakeHeadsModel, Option<f32>) {
    // Runs on this module's dedicated background OS thread — somnus_metrics::METRICS.record()
    // (invoked inside timed_core!) is a plain atomic fetch_add, safe to call from any thread.
    let (result, _) = somnus_metrics::timed_core!(somnus_metrics::Metric::WakeWorkerTrainStep, {
        let ep_t: Tensor<2> = Tensor::from_data(
            TensorData::new(episode.to_vec(), [1, episode.len()]),
            device,
        );
        let mut total_loss: Option<Tensor<1>> = None;
        let mut n = 0usize;

        for (head_idx, input) in regional.iter().enumerate() {
            if input.is_empty() {
                continue;
            }
            let target: Tensor<2> =
                Tensor::from_data(TensorData::new(input.clone(), [1, input.len()]), device);
            let pred = match head_idx {
                0 => model.speech.forward(ep_t.clone()),
                1 => model.env.forward(ep_t.clone()),
                2 => model.visual.forward(ep_t.clone()),
                3 => model.symbolic.forward(ep_t.clone()),
                4 => model.language.forward(ep_t.clone()),
                _ => unreachable!(),
            };
            let loss: Tensor<1> = (pred - target).powi_scalar(2).mean();
            total_loss = Some(match total_loss {
                None => loss,
                Some(acc) => acc + loss,
            });
            n += 1;
        }

        if let Some(loss) = total_loss {
            let loss_val: f32 = loss.clone().into_scalar::<f32>();
            let grads = loss.backward();
            let grads_p = GradientsParams::from_grads(grads, &model);
            let model = optim.step(lr, model, grads_p);
            ema.update_from_weights([
                model.speech.weight.val().inner(),
                model.env.weight.val().inner(),
                model.visual.weight.val().inner(),
                model.symbolic.weight.val().inner(),
                model.language.weight.val().inner(),
            ]);
            (model, Some(loss_val / n.max(1) as f32))
        } else {
            (model, None)
        }
    });
    result
}

fn weights_flat(model: &WakeHeadsModel) -> [Vec<f32>; 5] {
    [
        &model.speech,
        &model.env,
        &model.visual,
        &model.symbolic,
        &model.language,
    ]
    .map(|h| h.weight.val().inner().to_f32_vec())
}

enum Msg {
    Train {
        episode: Vec<f32>,
        regional: Box<[Vec<f32>; 5]>,
        lr: f64,
    },
    SetWeights(Box<[Vec<f32>; 5]>),
    FetchWeights,
}

enum Resp {
    Ema { flat: Box<[Vec<f32>; 5]>, loss: f32 },
    Weights(Box<[Vec<f32>; 5]>),
}

/// Handle to the wake-reconstruction worker thread. Same usage pattern as
/// `GainWorker` in the somnus crate: non-blocking sends, polled receives.
pub struct WakeReconWorker {
    tx: mpsc::SyncSender<Msg>,
    rx: mpsc::Receiver<Resp>,
}

impl WakeReconWorker {
    /// Spawn with the current model head weights and a clone of the current
    /// EMA state (so EMA continuity is preserved across the handoff).
    pub(crate) fn spawn(weights: [Vec<f32>; 5], mut ema: EmaOutputHeads) -> Self {
        // 16-deep: episodes arrive in bursts in file mode; a 4-deep queue
        // dropped ~10–15% of train requests in the 2026-06-12 A/B run.
        let (tx, req_rx) = mpsc::sync_channel::<Msg>(16);
        let (resp_tx, rx) = mpsc::sync_channel::<Resp>(4);

        std::thread::Builder::new()
            .name("wake-recon-worker".into())
            .spawn(move || {
                let device = Device::flex().autodiff();
                let mut model = Some(heads_from_flat(&weights, &device));
                let mut optim = AdamWConfig::new().init();
                while let Ok(msg) = req_rx.recv() {
                    match msg {
                        Msg::Train {
                            episode,
                            regional,
                            lr,
                        } => {
                            let (m, loss) = train_step(
                                model.take().unwrap(),
                                &mut optim,
                                &mut ema,
                                &episode,
                                &regional,
                                lr,
                                &device,
                            );
                            model = Some(m);
                            if let Some(loss) = loss {
                                // Drop the update if main hasn't polled yet; the
                                // next one supersedes it anyway.
                                let _ = resp_tx.try_send(Resp::Ema {
                                    flat: Box::new(ema.flat_clone()),
                                    loss,
                                });
                            }
                        }
                        Msg::SetWeights(w) => {
                            let mut m = model.take().unwrap();
                            set_linear_weight(&mut m.speech, &w[0], REGION_DIMS[0], &device);
                            set_linear_weight(&mut m.env, &w[1], REGION_DIMS[1], &device);
                            set_linear_weight(&mut m.visual, &w[2], REGION_DIMS[2], &device);
                            set_linear_weight(&mut m.symbolic, &w[3], REGION_DIMS[3], &device);
                            set_linear_weight(&mut m.language, &w[4], REGION_DIMS[4], &device);
                            model = Some(m);
                        }
                        Msg::FetchWeights => {
                            // Blocking send: delivery must be guaranteed for the
                            // sleep-entry handshake.
                            let flat = weights_flat(model.as_ref().unwrap());
                            if resp_tx.send(Resp::Weights(Box::new(flat))).is_err() {
                                break;
                            }
                        }
                    }
                }
            })
            .expect("failed to spawn wake-recon-worker thread");

        Self { tx, rx }
    }

    /// Non-blocking: queue a training step. Returns false (request dropped)
    /// when the worker is backlogged.
    pub fn request_train(&self, episode: Vec<f32>, regional: [Vec<f32>; 5], lr: f64) -> bool {
        self.tx
            .try_send(Msg::Train {
                episode,
                regional: Box::new(regional),
                lr,
            })
            .is_ok()
    }

    /// Non-blocking: latest EMA weight update, if one is ready.
    /// Returns (flat EMA weights, mean MSE of that step).
    pub fn try_recv_ema(&self) -> Option<([Vec<f32>; 5], f32)> {
        match self.rx.try_recv() {
            Ok(Resp::Ema { flat, loss }) => Some((*flat, loss)),
            // Weights responses only exist inside fetch_weights' recv loop.
            _ => None,
        }
    }

    /// Blocking sleep-entry handshake: drain queued work, return the worker's
    /// current head weights. EMA updates that arrive while draining are passed
    /// to `on_ema` so they aren't lost. Returns None if the worker died.
    pub fn fetch_weights(
        &self,
        mut on_ema: impl FnMut([Vec<f32>; 5], f32),
    ) -> Option<[Vec<f32>; 5]> {
        self.tx.send(Msg::FetchWeights).ok()?;
        while let Ok(resp) = self.rx.recv() {
            match resp {
                Resp::Ema { flat, loss } => on_ema(*flat, loss),
                Resp::Weights(w) => return Some(*w),
            }
        }
        None
    }

    /// Push NREM-trained head weights to the worker at wake re-entry.
    /// Blocking send — the queue is empty right after sleep.
    pub fn set_weights(&self, weights: [Vec<f32>; 5]) {
        let _ = self.tx.send(Msg::SetWeights(Box::new(weights)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PerceptionProcessor;

    /// Full protocol round-trip: train → EMA stream → sleep-entry weight fetch
    /// → wake re-entry weight push. Message ordering on the request channel
    /// makes the EMA-before-Weights sequence deterministic.
    #[test]
    fn round_trip_train_and_weight_sync() {
        let p = PerceptionProcessor::new(Device::flex());
        let initial = p.head_weights_flat();
        let worker = p.create_wake_recon_worker();

        let episode = vec![0.1f32; EPISODE_DIM];
        let regional: [Vec<f32>; 5] = std::array::from_fn(|i| vec![0.2f32; REGION_DIMS[i]]);
        assert!(worker.request_train(episode, regional, 1e-3));

        let mut ema_seen = 0;
        let weights = worker
            .fetch_weights(|flat, loss| {
                assert!(loss.is_finite());
                for (f, d) in flat.iter().zip(REGION_DIMS) {
                    assert_eq!(f.len(), EPISODE_DIM * d);
                }
                ema_seen += 1;
            })
            .expect("worker should be alive");
        assert_eq!(ema_seen, 1, "exactly one EMA update from the train step");
        for (w, d) in weights.iter().zip(REGION_DIMS) {
            assert_eq!(w.len(), EPISODE_DIM * d);
        }

        // The train step must actually move the weights. Guards against the
        // require_grad regression: a from_data tensor without .require_grad()
        // produces no gradients and AdamW silently no-ops.
        let moved = initial
            .iter()
            .zip(weights.iter())
            .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-9));
        assert!(moved, "head weights unchanged after a train step");

        worker.set_weights(weights.clone());
        // Worker must still be responsive after a weight push, and a second
        // train step must keep learning (weights pushed via SetWeights must
        // still require grad).
        let episode2 = vec![0.3f32; EPISODE_DIM];
        let regional2: [Vec<f32>; 5] = std::array::from_fn(|i| vec![0.5f32; REGION_DIMS[i]]);
        assert!(worker.request_train(episode2, regional2, 1e-3));
        let weights2 = worker.fetch_weights(|_, _| {}).expect("worker alive");
        let moved2 = weights2
            .iter()
            .zip(weights.iter())
            .any(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-9));
        assert!(moved2, "weights frozen after SetWeights push");
    }
}
