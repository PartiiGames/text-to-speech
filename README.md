Text to speech (Piper VITS voices) on burn, by Partii (https://partii.games), the place to make games together.

# text-to-speech

Real-time, on-device text to speech: Piper VITS voices on burn, native CPU. First sentence of audio in 65 ms on 4
desktop CPU threads; sentence streaming with cancellation. wasm and wgpu/WebGPU paths build but are untested.

Pure Rust on burn 0.22.1 (CPU = burn-flex; the GPU path is feature `gpu`), no C/C++ and no ONNX Runtime. It is Lindii's on-device voice in Partii, and it builds on its own against stock burn
from crates.io (no Partii-internal deps). License: MIT OR Apache-2.0 (LICENSE-MIT, LICENSE-APACHE); credits and the
voice's license in NOTICE.

First voice: `en_US-hifi_tts_92-medium` (rhasspy/piper-voices; weights CC-BY-4.0, Hi-Fi TTS data CC-BY-4.0,
openslr.org/109, trained from scratch; source onnx sha256 `14d19c09…5ff91`, 63.2 MB, 15.65 M params). **Weights are
never bundled**: the crate takes bytes; the caller fetches and checks them (the example does it by sha256).

## Run it
```sh
# voice_dir: model.safetensors + voice.json; missing files are fetched from
# https://api.partii.games/v1/ai/text-to-speech/<sha256>/<file> (SPEAK_VOICE_HOST=https://api-staging.partii.games)
cargo run --release --example speak -- voice_dir "həlˈoʊ! aɪm lˈɪndi." hello.wav
```
The example checks both files against the sha256 pinned in `examples/speak.rs` and refuses anything else. Upstream
source (and the fallback if the host is down): the Piper voice at the Hugging Face revision pinned in NOTICE,
converted with `tools/convert.py` (byte-identical output, same sha256).

## API
```rust
let sp = Speaker::load(&device, &voice_json, &safetensors)?;    // tools/convert.py output
let pcm = sp.synth(&phoneme_ids, &sp.default_opts()).await?;     // f32 PCM, sp.sample_rate() = 22050
let r = sp.synth_detail(&ids, &SynthOpts::deterministic(), false).await?; // + durations, ops, readbacks
speak_sentences(&sp, &sentences, &opts, &cancel, // feature `phonemes`: English text in (sentences split, phonemized by the text-to-phonemes crate, then spoken)
sp.speak_text("Hi! I'm Lindii. Let's make a game.", &opts, &cancel, |i, pcm| player.queue(i, pcm)).await?;
|i, pcm| player.queue(i, pcm)).await?;
```
`SynthOpts { noise_scale, length_scale, noise_w, seed }`; `default_opts()` = the voice json's inference values
(0.667 / 1.0 / 0.8). Noise comes from our own seeded generator, so a seed gives the same audio on every backend.
`phoneme_ids(&str)` maps espeak IPA to ids in Piper's layout (`^ _ p1 _ … pn _ $`) for a G2P that outputs IPA.

## The graph (the voice's ONNX export, mirrored module by module)
Inputs `input [1, T] i64`, `input_lengths [1]`, `scales [3]` = (noise_scale, length_scale, noise_w); output
`[1, 1, 1, samples]`. Single speaker (no `sid` input, no emb_g / cond layers); 256 symbols. opset 15, torch 2.7.
1. **Text encoder** `enc_p`: embedding [256, 192] × √192 → 6 × {multi-head attention (2 heads × 96, 1×1 conv
   q/k/v/o, relative-position keys and values in a ±4 window, `emb_rel_k/v [1, 9, 96]`), residual + LayerNorm(ch),
   FFN conv k3 192→768→192 with ReLU, residual + LayerNorm} → `proj` 1×1 → m_p, logs_p (192 each).
2. **Stochastic duration predictor** `dp`, reversed: pre 1×1 → DDSConv (3 layers, depthwise k3, dilations 1/3/9,
   LayerNorm + exact GELU) → proj; noise z [1, 2, T] × noise_w through Flip, ConvFlow 7, Flip, ConvFlow 5, Flip,
   ConvFlow 3, Flip, ElementwiseAffine 0 (ConvFlow 1 is the "useless vflow" VITS drops at inference, so the export
   has no weights for it). Each ConvFlow: pre 1→192, DDSConv conditioned on the predictor's features, proj → 29 =
   10 widths + 10 heights + 9 derivatives of an inverse rational-quadratic spline (tail bound 5, linear tails).
   logw = z[0]; w = exp(logw) × length_scale; durations = ceil(w).
3. **Expand**: m_p/logs_p repeated per frame (the monotonic path); z_p = m + noise × exp(logs) × noise_scale.
4. **Flow**, reversed: 4 × {Flip, residual coupling (mean only): pre 1×1 96→192, WaveNet (4 layers, k5, dilation 1,
   tanh·sigmoid gates, res/skip 1×1), post 192→96, x1 -= m}. Weight norm was folded at export (`onnx::Conv_*`).
5. **Generator** (HiFi-GAN, resblock type 2): conv_pre k7 192→256 → 3 × {LeakyReLU 0.1, ConvTranspose (×8 k16,
   ×8 k16, ×4 k8), mean of 3 resblocks (k3 d1,2; k5 d2,6; k7 d3,12)} → LeakyReLU 0.01 → conv_post k7 → tanh.
   Hop = 256 samples per frame (11.6 ms).

`tools/convert.py` keeps the PyTorch names; the folded ones are mapped from their consumer node (Conv node path →
`flow.flows.N.enc.*.weight`, `sid` → `enc_p.emb.weight`, `/dp/flows.0/Exp_output_0` → `dp.flows.0.exp_neg_logs`),
and the dilations, strides and paddings go into the voice json's `partii_arch` (defaults = this voice).

## Converting a voice
`tools/convert.py` (Python, needs the `onnx`, `numpy` and `safetensors` packages; we run it on the farm only, never
in the product): `python -I tools/convert.py voice.onnx voice.onnx.json model.safetensors voice.json`. The test tools
`tools/parity.py` (onnxruntime, the reference) and `tools/phonemize.py` (piper-phonemize / espeak-ng) are
test-only too; none of them is a crate dependency.

## Farm
`bash platform/ai/addons/text-to-speech/farm.sh` (from the repo root; `SPEAK_STEPS` picks steps): download + sha256 check,
convert, phonemize (piper-tts / espeak-ng, **test-only**), onnxruntime reference (**test-only**), cargo test, bench
build, parity, speed at 1 and 4 threads, wasm32 lib build. Weights stay out of git (job cache).

## Numbers (x86_64 Linux build machine, burn-flex CPU)
**Parity** vs onnxruntime on the original .onnx, noise_scale = noise_w = 0, 3 inputs (41 / 99 / 325 phoneme ids):

| input | durations | frames | w max-abs | encoder SNR | flow-out SNR | audio max-abs | audio SNR |
|---|---|---|---|---|---|---|---|
| short (41) | equal | 121 | 2.6e-5 | 125.2 dB | 117.1 dB | 1.3e-5 | 95.8 dB |
| parity (99) | equal | 268 | 4.2e-5 | 126.5 dB | 114.0 dB | 6.8e-5 | 88.3 dB |
| long (325) | equal | 680 | 1.1e-4 | 127.6 dB | 117.4 dB | 2.4e-5 | 97.8 dB |

The nearest pre-ceil duration was 0.0024 frames from an integer, so the ceil had margin.

**Speed** (the voice's default noise; median of 3; RTF = synth time / audio time):

| input | audio | 1 thread | RTF | 4 threads | RTF |
|---|---|---|---|---|---|
| "Hello! I'm Lindii." | 1.49 s | 264 ms | 0.18 | 109 ms | 0.07 |
| "Hello! I'm Lindii. Let's make a game together." | 3.30 s | 341 ms | 0.10 | 171 ms | 0.05 |
| 25-word sentence | 8.36 s | 966 ms | 0.12 | 462 ms | 0.06 |

**Ear check** (text -> text-to-phonemes -> this crate, vs Piper's own espeak-ng pipeline on the same voice; 30
game-chat lines, 306 words, both with the voice's default noise; transcribed by Whisper base.en): WER 2.9 % ours
(9 edits) vs 3.3 % Piper (10 edits). Most errors are shared (names: "Maya and Leo" heard as "Meyer and Leah"; "3:45
PM" as "3.45pm"). One run, so the difference is within noise: our G2P costs no measurable intelligibility here.

**speak_text first audio** on a 3-sentence reply ("Sure, let's do it! First we put a castle on the hill. Then I'll
add a friendly dragon who guards the gate."), time to the first sentence's PCM, median of 5 warm runs: 139 ms at
1 thread, 65 ms at 4 threads (whole reply 568 / 267 ms; the first call in a process, building the G2P tables,
165 / 72 ms).

With sentence streaming, Lindii's first audio = the first sentence's synth: about 0.1-0.35 s for a short opener on
these cores (load 20 ms). The wasm32 lib builds, and the crate builds and tests on its own against crates.io burn.

## Streaming (design)
Target: Lindii's voice within about 1 s of her reply text arriving.
- **Unit = sentence.** VITS makes a whole utterance at once (the durations, then every frame), so the chat's text
  stream is cut at sentence ends (`. ! ?`, and at a comma after ~12 words for run-ons), each sentence goes through G2P
  and `synth`, and its PCM is queued to the player. Playback starts when sentence 1 is done; sentence n+1 is made
  while n plays. On CPU the RTF is far below 1, so the queue never runs dry after the first sentence.
- **First audio** = chat time to the first sentence end + G2P + `synth(sentence 1)`. To keep it short, Lindii's
  replies should open with a short sentence ("Sure!", "Okay, here's the plan."); a very long first sentence can be
  split at its first comma.
- **Interruption** = `Cancel::set()`: no new sentence starts; the player drops its queue at once (that half is the
  player's). The sentence in flight finishes in the background and is discarded (≤ one sentence of compute).
- **Seeds**: sentence i uses seed + i, so a reply sounds the same when replayed.
- **Later, if sentence 1 is too slow**: the generator is ~all of the compute and is convolutional, so it can run in
  frame chunks with a few frames of overlap (receptive field), giving the first 0.3 s of audio before the sentence is
  done. Not needed at the CPU numbers below.

## Not done yet
- GPU (wgpu) parity and speed: the path compiles (feature `gpu`) but isn't measured. One mid-call readback (the
  durations) is inherent; the spline is all tensor ops (no extra readbacks).
- Multi-speaker voices (emb_g, cond layers) and resblock type 1 (Piper "high" voices).
- The browser entry (wasm-bindgen) and the Tauri plugin's `partii.ai.speak`.
