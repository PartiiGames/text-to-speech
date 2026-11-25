//! Real-time, on-device text to speech: Piper VITS voices on [burn](https://burn.dev), native CPU. First sentence of
//! audio in 65 ms on 4 desktop CPU threads; sentence streaming with cancellation. wasm and wgpu/WebGPU paths build but
//! are untested.
//!
//! Pure Rust on burn 0.22.1 (CPU = burn-flex; the GPU path is feature `gpu`), no ONNX Runtime. Async throughout (no
//! blocking readbacks, as wasm needs).
//!
//! The crate never bundles or downloads weights: you pass the bytes of a voice converted with `tools/convert.py`
//! (a `model.safetensors` and its `voice.json`), checked against their sha256 by the caller (see the `speak` example).
//!
//! ```no_run
//! use text_to_speech::{Speaker, SynthOpts};
//! use burn::tensor::Device;
//! # async fn run() -> Result<(), String> {
//! let voice_json = std::fs::read_to_string("voice.json").map_err(|e| e.to_string())?;
//! let weights = std::fs::read("model.safetensors").map_err(|e| e.to_string())?;
//! let sp = Speaker::load(&Device::flex(), &voice_json, &weights)?;
//! let ids = sp.phoneme_ids("həlˈoʊ!");            // espeak IPA -> ids (or ids straight from a G2P)
//! let pcm = sp.synth(&ids, &sp.default_opts()).await?; // mono f32 PCM at sp.sample_rate() (22050 Hz)
//! # Ok(()) }
//! ```
//! - **One readback mid-call**: the predicted durations (the output length must be known to expand the latent).
//! - **Sentence streaming**: [`speak_sentences`] synthesizes sentence by sentence and hands each to the player, so
//!   playback starts after sentence 1; a [`Cancel`] drops the rest.
//! - **Text in** (feature `phonemes`): `Speaker::speak_text` splits English text into sentences and phonemizes them
//!   with the pure-Rust text-to-phonemes crate; without the feature the crate takes phoneme ids (any G2P).
//! - **Deterministic**: noise comes from a seeded generator in the crate ([`SynthOpts::seed`]), the same on every
//!   backend; [`SynthOpts::deterministic`] turns the noise off (the parity setting).
//! - **Op counter**: [`Synth::ops`] / [`ops_total`], tensor ops issued by this crate (before burn's fusion).
mod model;
mod stream;
mod weights;

pub use model::{Arch, Audio, Inference, ResSpec, Speaker, Synth, SynthOpts, VoiceConfig};
pub use stream::{speak_sentences, Cancel};

use std::sync::atomic::{AtomicU64, Ordering};
static OPS: AtomicU64 = AtomicU64::new(0);
pub(crate) fn op(n: u64) {
    OPS.fetch_add(n, Ordering::Relaxed);
}
/// Tensor ops this crate has issued since the process started (all calls, all threads).
pub fn ops_total() -> u64 {
    OPS.load(Ordering::Relaxed)
}
