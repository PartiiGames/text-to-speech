//! Piper's VITS (the inference half of SynthesizerTrn.infer) on burn, mirroring the voice's ONNX export:
//! text encoder (relative-position transformer) -> stochastic duration predictor (reversed flows with a rational-
//! quadratic spline) -> durations (the one mid-call readback) -> expand -> flow (residual coupling, reversed) ->
//! HiFi-GAN generator. Tensors are batch 1, channels first ([1, C, T]); there is no padding mask (one utterance).
use crate::op;
use crate::weights::Weights;
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{activation, module, Device, Int, Tensor, TensorData};
use std::collections::HashMap;

/// The numbers shapes can't give (tools/convert.py reads them from the ONNX node attributes into the voice json's
/// "partii_arch"). Defaults: Piper "medium" (en_US-hifi_tts_92-medium).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct Arch {
    pub hidden: usize,
    pub window: usize,
    pub heads: usize,
    pub enc_layers: usize,
    pub ffn_kernel: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernels: Vec<usize>,
    pub upsample_pads: Vec<usize>,
    pub resblocks: Vec<ResSpec>,
    /// flow.flows.N indices of the residual coupling layers (a Flip follows each)
    pub flows: Vec<usize>,
    pub wn_layers: usize,
    pub wn_kernel: usize,
    pub wn_dilations: Vec<usize>,
    /// dp.flows.N indices of the ConvFlows used at inference, in the reversed (inference) order
    pub dp_flows: Vec<usize>,
    pub dds_dilations: Vec<usize>,
    pub dp_bins: usize,
    pub tail_bound: f64,
}

/// One generator resblock (type 2): kernel size and the dilation of each conv.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ResSpec {
    pub kernel: usize,
    pub dilations: Vec<usize>,
}

impl Default for Arch {
    fn default() -> Self {
        Arch {
            hidden: 192,
            window: 4,
            heads: 2,
            enc_layers: 6,
            ffn_kernel: 3,
            upsample_rates: vec![8, 8, 4],
            upsample_kernels: vec![16, 16, 8],
            upsample_pads: vec![4, 4, 2],
            resblocks: vec![ResSpec { kernel: 3, dilations: vec![1, 2] }, ResSpec { kernel: 5, dilations: vec![2, 6] }, ResSpec { kernel: 7, dilations: vec![3, 12] }],
            flows: vec![0, 2, 4, 6],
            wn_layers: 4,
            wn_kernel: 5,
            wn_dilations: vec![1, 1, 1, 1],
            dp_flows: vec![7, 5, 3],
            dds_dilations: vec![1, 3, 9],
            dp_bins: 10,
            tail_bound: 5.0,
        }
    }
}

/// The voice's .onnx.json (Piper's format), plus "partii_arch" when tools/convert.py wrote it.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VoiceConfig {
    pub audio: Audio,
    pub inference: Inference,
    #[serde(default)]
    pub num_speakers: usize,
    pub phoneme_id_map: HashMap<String, Vec<i64>>,
    #[serde(default)]
    pub partii_arch: Arch,
}

/// The voice json's `audio` section.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Audio {
    pub sample_rate: u32,
}

/// The voice json's `inference` section: the voice's default noise and speed.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub struct Inference {
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_w: f32,
}

/// Per-call synthesis settings ([`Speaker::default_opts`] gives the voice's own).
#[derive(Debug, Clone, Copy)]
pub struct SynthOpts {
    /// Spread of the latent around the text's mean (0 = deterministic). Piper default for this voice: 0.667.
    pub noise_scale: f32,
    /// > 1 slower speech, < 1 faster.
    pub length_scale: f32,
    /// Spread of the duration predictor's noise (0 = deterministic). Piper default: 0.8.
    pub noise_w: f32,
    /// Seed for both noises (our own generator: the same audio on every backend for a seed).
    pub seed: u64,
}

impl SynthOpts {
    /// noise_scale = noise_w = 0: the parity setting (same durations and audio as onnxruntime with scales [0, ls, 0]).
    pub fn deterministic() -> Self {
        SynthOpts { noise_scale: 0.0, length_scale: 1.0, noise_w: 0.0, seed: 0 }
    }
}

/// One synthesis, with the numbers.
#[derive(Debug, Clone)]
pub struct Synth {
    /// PCM at the voice's sample rate, in [-1, 1].
    pub pcm: Vec<f32>,
    /// Frames per phoneme id (ceil of the predicted durations; 1 frame = hop samples).
    pub durations: Vec<u32>,
    /// Pre-ceil durations (debug/parity).
    pub w: Vec<f32>,
    /// Tensor ops this call issued (before fusion).
    pub ops: u64,
    /// Device-to-host readbacks this call made (2 without debug: the durations and the audio).
    pub readbacks: u32,
    /// Debug stages (stats = encoder output [2*hidden, T], z = flow output [hidden, frames]), when asked for.
    pub stages: Vec<(&'static str, Vec<f32>)>,
}

struct Conv {
    w: Tensor<3>,
    w1: Option<Tensor<2>>, // 1x1, groups 1: the [out, in] matrix (a matmul instead of conv1d)
    b: Option<Tensor<1>>,
    pad: usize,
    dil: usize,
    groups: usize,
}

impl Conv {
    fn load(w: &Weights, p: &str, pad: usize, dil: usize, groups: usize) -> Result<Conv, String> {
        let wt = w.t::<3>(&format!("{p}.weight"))?;
        let [o, i, k] = wt.dims();
        let b = if w.has(&format!("{p}.bias")) { Some(w.t::<1>(&format!("{p}.bias"))?) } else { None };
        let w1 = (k == 1 && groups == 1 && pad == 0).then(|| wt.clone().reshape([o, i]));
        Ok(Conv { w: wt, w1, b, pad, dil, groups })
    }
    fn f(&self, x: Tensor<3>) -> Tensor<3> {
        op(1);
        if let Some(w1) = &self.w1 {
            let [_, c, t] = x.dims();
            let y = w1.clone().matmul(x.reshape([c, t]));
            let y = match &self.b {
                Some(b) => {
                    op(1);
                    y + b.clone().unsqueeze_dim::<2>(1)
                }
                None => y,
            };
            let o = y.dims()[0];
            return y.reshape([1, o, t]);
        }
        module::conv1d(x, self.w.clone(), self.b.clone(), ConvOptions::new([1], [self.pad], [self.dil], self.groups))
    }
}

/// VITS LayerNorm: over channels of [1, C, T].
struct Ln {
    g: Tensor<1>,
    b: Tensor<1>,
}

impl Ln {
    fn load(w: &Weights, p: &str) -> Result<Ln, String> {
        Ok(Ln { g: w.t::<1>(&format!("{p}.gamma"))?, b: w.t::<1>(&format!("{p}.beta"))? })
    }
    fn f(&self, x: Tensor<3>) -> Tensor<3> {
        op(1);
        module::layer_norm(x.swap_dims(1, 2), self.g.clone(), Some(self.b.clone()), 1e-5).swap_dims(1, 2)
    }
}

/// Relative-position index tensors for one length (the windowed embeddings as gathers, built on the CPU).
struct RelIdx {
    k_idx: Tensor<4, Int>, // [1, h, t, t]: j - i + W, clamped
    k_band: Tensor<4>,     // 1 where |j - i| <= W
    v_idx: Tensor<4, Int>, // [1, h, t, 2W+1]: i + r - W, clamped
    v_band: Tensor<4>,     // 1 where 0 <= i + r - W < t
}

impl RelIdx {
    fn new(t: usize, heads: usize, win: usize, dev: &Device) -> RelIdx {
        let w = win as i64;
        let n = 2 * win + 1;
        let (mut ki, mut kb, mut vi, mut vb) = (Vec::with_capacity(t * t), Vec::with_capacity(t * t), Vec::with_capacity(t * n), Vec::with_capacity(t * n));
        for i in 0..t as i64 {
            for j in 0..t as i64 {
                let d = j - i;
                ki.push((d + w).clamp(0, 2 * w));
                kb.push(if d.abs() <= w { 1f32 } else { 0.0 });
            }
            for r in 0..n as i64 {
                let j = i + r - w;
                vi.push(j.clamp(0, t as i64 - 1));
                vb.push(if j >= 0 && j < t as i64 { 1f32 } else { 0.0 });
            }
        }
        let rep_i = |v: Vec<i64>, c: usize| Tensor::<4, Int>::from_data(TensorData::new(v.repeat(heads), [1, heads, t, c]), dev);
        let rep_f = |v: Vec<f32>, c: usize| Tensor::<4>::from_data(TensorData::new(v.repeat(heads), [1, heads, t, c]), dev);
        RelIdx { k_idx: rep_i(ki, t), k_band: rep_f(kb, t), v_idx: rep_i(vi, n), v_band: rep_f(vb, n) }
    }
}

struct EncLayer {
    q: Conv,
    k: Conv,
    v: Conv,
    o: Conv,
    rel_k: Tensor<4>, // [1, h, kc, 2W+1]
    rel_v: Tensor<4>, // [1, h, 2W+1, kc]
    ln1: Ln,
    ffn1: Conv,
    ffn2: Conv,
    ln2: Ln,
}

impl EncLayer {
    fn attn(&self, x: Tensor<3>, heads: usize, rel: &RelIdx) -> Tensor<3> {
        let [_, c, t] = x.dims();
        let kc = c / heads;
        let split = |y: Tensor<3>| y.reshape([1, heads, kc, t]).swap_dims(2, 3); // [1, h, t, kc]
        op(1);
        let q = split(self.q.f(x.clone())).div_scalar((kc as f64).sqrt());
        let k = split(self.k.f(x.clone()));
        let v = split(self.v.f(x));
        op(9);
        let scores = q.clone().matmul(k.swap_dims(2, 3)); // [1, h, t, t]
        let local = q.matmul(self.rel_k.clone()).gather(3, rel.k_idx.clone()) * rel.k_band.clone();
        let p = activation::softmax(scores + local, 3);
        let out = p.clone().matmul(v) + (p.gather(3, rel.v_idx.clone()) * rel.v_band.clone()).matmul(self.rel_v.clone());
        self.o.f(out.swap_dims(2, 3).reshape([1, c, t]))
    }
}

/// DDSConv: dilated depth-separable convs (the duration predictor's body).
struct Dds {
    sep: Vec<Conv>,
    c11: Vec<Conv>,
    n1: Vec<Ln>,
    n2: Vec<Ln>,
}

impl Dds {
    fn load(w: &Weights, p: &str, ch: usize, dils: &[usize]) -> Result<Dds, String> {
        let mut d = Dds { sep: vec![], c11: vec![], n1: vec![], n2: vec![] };
        for (i, &dl) in dils.iter().enumerate() {
            d.sep.push(Conv::load(w, &format!("{p}.convs_sep.{i}"), dl, dl, ch)?); // kernel 3: pad = dilation
            d.c11.push(Conv::load(w, &format!("{p}.convs_1x1.{i}"), 0, 1, 1)?);
            d.n1.push(Ln::load(w, &format!("{p}.norms_1.{i}"))?);
            d.n2.push(Ln::load(w, &format!("{p}.norms_2.{i}"))?);
        }
        Ok(d)
    }
    fn f(&self, mut x: Tensor<3>) -> Tensor<3> {
        for i in 0..self.sep.len() {
            op(3);
            let y = activation::gelu(self.n1[i].f(self.sep[i].f(x.clone())));
            let y = activation::gelu(self.n2[i].f(self.c11[i].f(y)));
            x = x + y;
        }
        x
    }
}

struct ConvFlow {
    pre: Conv,
    dds: Dds,
    proj: Conv,
}

struct Coupling {
    pre: Conv,
    inl: Vec<Conv>,
    res: Vec<Conv>,
    post: Conv,
}

struct Gen {
    pre: Conv,
    ups: Vec<(Tensor<3>, Tensor<1>, usize, usize)>, // weight [in, out, k], bias, stride, pad
    res: Vec<Vec<Conv>>,
    post: Conv,
}

/// A loaded voice. Cheap to share by reference; `synth` takes `&self`.
pub struct Speaker {
    cfg: VoiceConfig,
    dev: Device,
    emb: Tensor<2>,
    enc: Vec<EncLayer>,
    proj: Conv,
    dp_pre: Conv,
    dp_dds: Dds,
    dp_proj: Conv,
    dp_flows: Vec<ConvFlow>,
    dp_m: Tensor<3>,       // [1, 2, 1]
    dp_exp_nl: Tensor<3>,  // exp(-logs) [1, 2, 1]
    flows: Vec<Coupling>,  // flow.flows order (forward); run reversed
    generator: Gen,
    hop: usize,
    #[cfg(feature = "phonemes")]
    ids_map: text_to_phonemes::PhonemeIdMap,
    #[cfg(feature = "phonemes")]
    g2p: std::sync::OnceLock<text_to_phonemes::G2p>,
}

impl Speaker {
    /// Builds the voice on `dev` from its voice json (Piper's `.onnx.json`, plus `partii_arch` from
    /// `tools/convert.py`; without it the Piper "medium" layout is assumed) and its safetensors bytes (f32, f16 or bf16).
    pub fn load(dev: &Device, config_json: &str, safetensors: &[u8]) -> Result<Speaker, String> {
        let cfg: VoiceConfig = serde_json::from_str(config_json).map_err(|e| format!("voice json: {e}"))?;
        if cfg.num_speakers > 1 {
            return Err("multi-speaker voices: not yet (emb_g / cond layers)".into());
        }
        let a = cfg.partii_arch.clone();
        let w = Weights::new(safetensors, dev)?;
        let h = a.hidden;
        let kc = h / a.heads;
        let fp = a.ffn_kernel / 2;
        let mut enc = Vec::new();
        for i in 0..a.enc_layers {
            let p = format!("enc_p.encoder.attn_layers.{i}");
            let f = format!("enc_p.encoder.ffn_layers.{i}");
            let rk = w.t::<3>(&format!("{p}.emb_rel_k"))?.squeeze_dim::<2>(0); // [2W+1, kc]
            let rv = w.t::<3>(&format!("{p}.emb_rel_v"))?.squeeze_dim::<2>(0);
            let n = rk.dims()[0];
            enc.push(EncLayer {
                q: Conv::load(&w, &format!("{p}.conv_q"), 0, 1, 1)?,
                k: Conv::load(&w, &format!("{p}.conv_k"), 0, 1, 1)?,
                v: Conv::load(&w, &format!("{p}.conv_v"), 0, 1, 1)?,
                o: Conv::load(&w, &format!("{p}.conv_o"), 0, 1, 1)?,
                rel_k: rk.transpose().reshape([1, 1, kc, n]).repeat_dim(1, a.heads),
                rel_v: rv.reshape([1, 1, n, kc]).repeat_dim(1, a.heads),
                ln1: Ln::load(&w, &format!("enc_p.encoder.norm_layers_1.{i}"))?,
                ffn1: Conv::load(&w, &format!("{f}.conv_1"), fp, 1, 1)?,
                ffn2: Conv::load(&w, &format!("{f}.conv_2"), fp, 1, 1)?,
                ln2: Ln::load(&w, &format!("enc_p.encoder.norm_layers_2.{i}"))?,
            });
        }
        let mut dp_flows = Vec::new();
        for &fi in &a.dp_flows {
            let p = format!("dp.flows.{fi}");
            dp_flows.push(ConvFlow { pre: Conv::load(&w, &format!("{p}.pre"), 0, 1, 1)?, dds: Dds::load(&w, &format!("{p}.convs"), h, &a.dds_dilations)?, proj: Conv::load(&w, &format!("{p}.proj"), 0, 1, 1)? });
        }
        let mut flows = Vec::new();
        for &fi in &a.flows {
            let p = format!("flow.flows.{fi}");
            let mut inl = Vec::new();
            let mut res = Vec::new();
            for i in 0..a.wn_layers {
                let d = a.wn_dilations[i];
                inl.push(Conv::load(&w, &format!("{p}.enc.in_layers.{i}"), (a.wn_kernel * d - d) / 2, d, 1)?);
                res.push(Conv::load(&w, &format!("{p}.enc.res_skip_layers.{i}"), 0, 1, 1)?);
            }
            flows.push(Coupling { pre: Conv::load(&w, &format!("{p}.pre"), 0, 1, 1)?, inl, res, post: Conv::load(&w, &format!("{p}.post"), 0, 1, 1)? });
        }
        let mut ups = Vec::new();
        let mut res = Vec::new();
        for i in 0..a.upsample_rates.len() {
            ups.push((w.t::<3>(&format!("dec.ups.{i}.weight"))?, w.t::<1>(&format!("dec.ups.{i}.bias"))?, a.upsample_rates[i], a.upsample_pads[i]));
            for (j, r) in a.resblocks.iter().enumerate() {
                let p = format!("dec.resblocks.{}", i * a.resblocks.len() + j);
                let mut cs = Vec::new();
                for (c, &d) in r.dilations.iter().enumerate() {
                    cs.push(Conv::load(&w, &format!("{p}.convs.{c}"), (r.kernel * d - d) / 2, d, 1)?);
                }
                res.push(cs);
            }
        }
        let post_k = w.t::<3>("dec.conv_post.weight")?.dims()[2];
        let pre_k = w.t::<3>("dec.conv_pre.weight")?.dims()[2];
        let generator = Gen { pre: Conv::load(&w, "dec.conv_pre", pre_k / 2, 1, 1)?, ups, res, post: Conv::load(&w, "dec.conv_post", post_k / 2, 1, 1)? };
        Ok(Speaker {
            emb: w.t::<2>("enc_p.emb.weight")?,
            enc,
            proj: Conv::load(&w, "enc_p.proj", 0, 1, 1)?,
            dp_pre: Conv::load(&w, "dp.pre", 0, 1, 1)?,
            dp_dds: Dds::load(&w, "dp.convs", h, &a.dds_dilations)?,
            dp_proj: Conv::load(&w, "dp.proj", 0, 1, 1)?,
            dp_flows,
            dp_m: w.t::<2>("dp.flows.0.m")?.unsqueeze_dim::<3>(0),
            dp_exp_nl: w.t::<2>("dp.flows.0.exp_neg_logs")?.unsqueeze_dim::<3>(0),
            flows,
            generator,
            hop: a.upsample_rates.iter().product(),
            #[cfg(feature = "phonemes")]
            ids_map: text_to_phonemes::phoneme_id_map_from_config(config_json)?,
            #[cfg(feature = "phonemes")]
            g2p: std::sync::OnceLock::new(),
            dev: dev.clone(),
            cfg,
        })
    }

    /// The voice json this speaker was loaded from.
    pub fn config(&self) -> &VoiceConfig {
        &self.cfg
    }

    /// Output samples per duration frame (the product of the generator's upsample rates; 256 for this voice).
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// PCM sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.cfg.audio.sample_rate
    }

    /// The voice's own inference defaults (from its .onnx.json), seed 0.
    pub fn default_opts(&self) -> SynthOpts {
        let i = self.cfg.inference;
        SynthOpts { noise_scale: i.noise_scale, length_scale: i.length_scale, noise_w: i.noise_w, seed: 0 }
    }

    /// Phonemes (espeak IPA, one char each, as Piper's phonemizer gives them) -> ids in Piper's layout:
    /// ^ _ p1 _ p2 _ ... pn _ $. Unknown phonemes are skipped (as Piper does).
    pub fn phoneme_ids(&self, phonemes: &str) -> Vec<i64> {
        let m = &self.cfg.phoneme_id_map;
        let get = |k: &str| m.get(k).cloned().unwrap_or_default();
        let pad = get("_");
        let mut ids = get("^");
        ids.extend(&pad);
        for c in phonemes.chars() {
            if let Some(v) = m.get(c.encode_utf8(&mut [0; 4]) as &str) {
                ids.extend(v);
                ids.extend(&pad);
            }
        }
        ids.extend(get("$"));
        ids
    }

    /// English text -> one phoneme id list per sentence (feature `phonemes`: the text-to-phonemes G2P, espeak-ng
    /// style en-us IPA, then this voice's `phoneme_id_map`). The G2P's tables are built on first use.
    #[cfg(feature = "phonemes")]
    pub fn text_ids(&self, text: &str) -> Vec<Vec<i64>> {
        let g = self.g2p.get_or_init(text_to_phonemes::G2p::new);
        g.sentences(text).iter().filter(|s| !s.trim().is_empty()).map(|s| text_to_phonemes::to_ids(s, &self.ids_map)).collect()
    }

    /// Builds the G2P's tables now (they are otherwise built on the first `text_ids` / `speak_text`).
    #[cfg(feature = "phonemes")]
    pub fn warm_g2p(&self) {
        self.g2p.get_or_init(text_to_phonemes::G2p::new);
    }

    /// Phoneme ids -> PCM at `sample_rate()`.
    pub async fn synth(&self, phoneme_ids: &[i64], opts: &SynthOpts) -> Result<Vec<f32>, String> {
        Ok(self.synth_detail(phoneme_ids, opts, false).await?.pcm)
    }

    /// As [`Speaker::synth`], with durations and counters; `debug` also reads back the encoder output and the
    /// flow output (two extra readbacks) into [`Synth::stages`].
    pub async fn synth_detail(&self, ids: &[i64], opts: &SynthOpts, debug: bool) -> Result<Synth, String> {
        let ops0 = crate::ops_total();
        let t = ids.len();
        if t == 0 {
            return Err("no phoneme ids".into());
        }
        let n_sym = self.emb.dims()[0] as i64;
        if let Some(bad) = ids.iter().find(|&&i| i < 0 || i >= n_sym) {
            return Err(format!("phoneme id {bad} out of range (0..{n_sym})"));
        }
        let a = &self.cfg.partii_arch;
        let h = a.hidden;
        let dev = &self.dev;
        let mut rng = Rng::new(opts.seed);
        let mut stages = Vec::new();
        let mut readbacks = 0u32;

        // text encoder
        let idx = Tensor::<1, Int>::from_data(TensorData::new(ids.to_vec(), [t]), dev);
        op(3);
        let mut x = self.emb.clone().select(0, idx).mul_scalar((h as f64).sqrt()).transpose().reshape([1, h, t]);
        let rel = RelIdx::new(t, a.heads, a.window, dev);
        for l in &self.enc {
            op(4);
            let y = l.attn(x.clone(), a.heads, &rel);
            x = l.ln1.f(x + y);
            let y = l.ffn2.f(activation::relu(l.ffn1.f(x.clone())));
            x = l.ln2.f(x + y);
        }
        let stats = self.proj.f(x.clone()); // [1, 2h, t]
        let m_p = stats.clone().narrow(1, 0, h);
        let logs_p = stats.clone().narrow(1, h, h);

        // stochastic duration predictor, reversed
        op(2);
        let g = self.dp_proj.f(self.dp_dds.f(self.dp_pre.f(x)));
        let mut z = if opts.noise_w == 0.0 { Tensor::<3>::zeros([1, 2, t], dev) } else { self.noise([1, 2, t], opts.noise_w, &mut rng) };
        for f in &self.dp_flows {
            op(1);
            z = z.flip([1]);
            z = self.conv_flow_rev(f, z, g.clone(), a);
        }
        op(4);
        z = (z.flip([1]) - self.dp_m.clone()) * self.dp_exp_nl.clone();
        let w = z.narrow(1, 0, 1).exp().mul_scalar(opts.length_scale as f64); // [1, 1, t]
        readbacks += 1;
        let wv: Vec<f32> = read(w).await?;
        let durations: Vec<u32> = wv.iter().map(|v| v.ceil().max(0.0) as u32).collect();
        let frames_idx: Vec<i64> = durations.iter().enumerate().flat_map(|(i, &d)| std::iter::repeat_n(i as i64, d as usize)).collect();
        let ty = frames_idx.len().max(1);

        // expand the text's mean/log-std to frames; sample the latent
        op(2);
        let (m, logs) = if frames_idx.is_empty() {
            (Tensor::<3>::zeros([1, h, 1], dev), Tensor::<3>::zeros([1, h, 1], dev))
        } else {
            let fi = Tensor::<1, Int>::from_data(TensorData::new(frames_idx, [ty]), dev);
            (m_p.select(2, fi.clone()), logs_p.select(2, fi))
        };
        let mut z = if opts.noise_scale == 0.0 {
            m
        } else {
            op(3);
            m + self.noise([1, h, ty], opts.noise_scale, &mut rng) * logs.exp()
        };

        // flow, reversed
        let half = h / 2;
        for c in self.flows.iter().rev() {
            op(4);
            z = z.flip([1]);
            let x0 = z.clone().narrow(1, 0, half);
            let x1 = z.narrow(1, half, half);
            let mut hh = c.pre.f(x0.clone());
            let mut skip: Option<Tensor<3>> = None;
            let n = c.inl.len();
            for i in 0..n {
                op(3);
                let xin = c.inl[i].f(hh.clone());
                let acts = xin.clone().narrow(1, 0, h).tanh() * activation::sigmoid(xin.narrow(1, h, h));
                let rs = c.res[i].f(acts);
                let s = if i < n - 1 {
                    hh = hh + rs.clone().narrow(1, 0, h);
                    rs.narrow(1, h, h)
                } else {
                    rs
                };
                skip = Some(match skip {
                    Some(o) => o + s,
                    None => s,
                });
            }
            let mm = c.post.f(skip.expect("wn_layers >= 1"));
            z = Tensor::cat(vec![x0, x1 - mm], 1);
        }

        if debug {
            readbacks += 2;
            stages.push(("stats", read(stats).await?));
            stages.push(("z", read(z.clone()).await?));
        }

        // generator
        let gn = &self.generator;
        let mut x = gn.pre.f(z);
        let nk = a.resblocks.len();
        for (i, (uw, ub, s, p)) in gn.ups.iter().enumerate() {
            op(2);
            x = activation::leaky_relu(x, 0.1);
            x = module::conv_transpose1d(x, uw.clone(), Some(ub.clone()), ConvTransposeOptions::new([*s], [*p], [0], [1], 1));
            let mut xs: Option<Tensor<3>> = None;
            for j in 0..nk {
                let mut r = x.clone();
                for c in &gn.res[i * nk + j] {
                    op(2);
                    r = c.f(activation::leaky_relu(r.clone(), 0.1)) + r;
                }
                xs = Some(match xs {
                    Some(o) => {
                        op(1);
                        o + r
                    }
                    None => r,
                });
            }
            op(1);
            x = xs.expect("resblocks").div_scalar(nk as f64);
        }
        op(2);
        let audio = gn.post.f(activation::leaky_relu(x, 0.01)).tanh();
        readbacks += 1;
        let pcm = read(audio).await?;
        Ok(Synth { pcm, durations, w: wv, ops: crate::ops_total() - ops0, readbacks, stages })
    }

    fn noise(&self, shape: [usize; 3], scale: f32, rng: &mut Rng) -> Tensor<3> {
        let n = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| rng.normal() * scale).collect();
        Tensor::<3>::from_data(TensorData::new(v, shape), &self.dev)
    }

    fn conv_flow_rev(&self, f: &ConvFlow, z: Tensor<3>, g: Tensor<3>, a: &Arch) -> Tensor<3> {
        let t = z.dims()[2];
        let k = a.dp_bins;
        let x0 = z.clone().narrow(1, 0, 1);
        let x1 = z.narrow(1, 1, 1);
        op(2);
        let hh = f.proj.f(f.dds.f(f.pre.f(x0.clone()) + g)); // [1, 3k-1, t]
        let hh = hh.reshape([3 * k - 1, t]).transpose(); // [t, 3k-1]
        let s = 1.0 / (a.hidden as f64).sqrt();
        let uw = hh.clone().narrow(1, 0, k).mul_scalar(s);
        let uh = hh.clone().narrow(1, k, k).mul_scalar(s);
        let ud = hh.narrow(1, 2 * k, k - 1);
        let y1 = rq_spline_inverse(x1.reshape([t, 1]), uw, uh, ud, a.tail_bound);
        Tensor::cat(vec![x0, y1.reshape([1, 1, t])], 1)
    }
}

/// Inverse of the unconstrained rational-quadratic spline with linear tails (VITS transforms.py, min bin width /
/// height / derivative 1e-3). x: [t, 1]; uw, uh: [t, K]; ud: [t, K-1]. Outside [-B, B] the identity.
pub(crate) fn rq_spline_inverse(x: Tensor<2>, uw: Tensor<2>, uh: Tensor<2>, ud: Tensor<2>, bound: f64) -> Tensor<2> {
    const MIN: f64 = 1e-3;
    let [t, k] = uw.dims();
    let dev = x.device();
    op(40);
    let edge = |v: f64| Tensor::<2>::zeros([t, 1], &dev).add_scalar(v);
    let knots = |u: Tensor<2>| {
        let w = activation::softmax(u, 1).mul_scalar(1.0 - MIN * k as f64).add_scalar(MIN);
        let c = w.cumsum(1).mul_scalar(2.0 * bound).add_scalar(-bound); // knots 1..=K
        let c = Tensor::cat(vec![edge(-bound), c.narrow(1, 0, k - 1), edge(bound)], 1); // [t, K+1], ends pinned
        let widths = c.clone().narrow(1, 1, k) - c.clone().narrow(1, 0, k);
        (c, widths)
    };
    let (cw, widths) = knots(uw);
    let (ch, heights) = knots(uh);
    let c0 = ((1.0 - MIN).exp() - 1.0).ln();
    let d = activation::softplus(Tensor::cat(vec![edge(c0), ud, edge(c0)], 1), 1.0).add_scalar(MIN); // [t, K+1]
    // searchsorted over the height knots (last + 1e-6), as transforms.py
    let ch_eps = Tensor::cat(vec![ch.clone().narrow(1, 0, k), ch.clone().narrow(1, k, 1).add_scalar(1e-6)], 1);
    let bin = x.clone().repeat_dim(1, k + 1).greater_equal(ch_eps).int().sum_dim(1).sub_scalar(1).clamp(0i64, k as i64 - 1); // [t, 1]
    let gat = |v: Tensor<2>| v.gather(1, bin.clone());
    let delta = heights.clone() / widths.clone();
    let (icw, ibw, ich, idl, id0, id1, ih) = (gat(cw), gat(widths), gat(ch), gat(delta), gat(d.clone().narrow(1, 0, k)), gat(d.narrow(1, 1, k)), gat(heights));
    let xr = x.clone() - ich;
    let s = id0.clone() + id1 - idl.clone().mul_scalar(2.0);
    let qa = xr.clone() * s.clone() + ih.clone() * (idl.clone() - id0.clone());
    let qb = ih * id0 - xr.clone() * s;
    let qc = idl.neg() * xr;
    let disc = qb.clone().square() - qa.mul_scalar(4.0) * qc.clone();
    let root = qc.mul_scalar(2.0) / (qb.neg() - disc.sqrt());
    let y = root * ibw + icw;
    let inside = x.clone().greater_equal_elem(-bound).bool_and(x.clone().lower_equal_elem(bound));
    x.mask_where(inside, y)
}

/// SplitMix64 + Box-Muller: the same noise on every backend for a seed.
struct Rng(u64, Option<f32>);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15, None)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f32 {
        if let Some(v) = self.1.take() {
            return v;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f64::consts::TAU * u2;
        self.1 = Some((r * th.sin()) as f32);
        (r * th.cos()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spline_identity_when_flat() {
        // uniform bins and unit derivatives (ud = the tail constant) make the spline the identity
        let dev = Device::flex();
        let (t, k) = (7, 10);
        let c0 = ((1.0f64 - 1e-3).exp() - 1.0).ln() as f32;
        let xs = vec![-6.0f32, -4.9, -1.0, 0.0, 0.3, 4.99, 6.0];
        let x = Tensor::<2>::from_data(TensorData::new(xs.clone(), [t, 1]), &dev);
        let z = Tensor::<2>::zeros([t, k], &dev);
        let ud = Tensor::<2>::zeros([t, k - 1], &dev).add_scalar(c0);
        let y: Vec<f32> = rq_spline_inverse(x, z.clone(), z, ud, 5.0).into_data().iter().collect();
        for (a, b) in xs.iter().zip(&y) {
            assert!((a - b).abs() < 1e-4, "{a} -> {b}");
        }
    }

    #[test]
    fn rel_idx() {
        let r = RelIdx::new(6, 1, 4, &Device::flex());
        let ki: Vec<i64> = r.k_idx.into_data().iter().collect();
        let kb: Vec<f32> = r.k_band.into_data().iter().collect();
        // row 0: j - 0 + 4 = 4..8, then clamped 8 (j = 5 is outside the window)
        assert_eq!(&ki[0..6], &[4, 5, 6, 7, 8, 8]);
        assert_eq!(&kb[0..6], &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
        let vb: Vec<f32> = r.v_band.into_data().iter().collect();
        assert_eq!(&vb[0..9], &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn normal_noise() {
        let mut r = Rng::new(1);
        let v: Vec<f32> = (0..20000).map(|_| r.normal()).collect();
        let m = v.iter().sum::<f32>() / v.len() as f32;
        let s = (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32).sqrt();
        assert!(m.abs() < 0.03 && (s - 1.0).abs() < 0.03, "{m} {s}");
    }
}

/// Async readback (no blocking on wasm), converted to f32 whatever the backend's dtype.
async fn read<const D: usize>(t: Tensor<D>) -> Result<Vec<f32>, String> {
    Ok(t.into_data_async().await.map_err(|e| format!("{e:?}"))?.iter::<f32>().collect())
}
