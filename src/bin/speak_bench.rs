//! Native bench + parity dumps for text-to-speech. Prints a timestamped line per step (the farm's progress signal).
//!   speak_bench <voice_dir> <ids.json> <cpu|wgpu> <out_dir> parity
//!       deterministic synth of every item -> <out_dir>/burn_<name>.{audio,w,stats,z}.f32 (tools/parity.py cmp)
//!   speak_bench <voice_dir> <ids.json> <cpu|wgpu> <out_dir> speed <label> [reps]
//!       the voice's default noise; per item: ms, audio s, RTF, and the first sentence's time (sentence streaming)
//!   speak_bench <voice_dir> <lines.txt> <cpu|wgpu> <out_dir> text      (feature phonemes)
//!       each line through speak_text (our G2P + VITS, the voice's default noise, seed 7) -> <out_dir>/ours_NN.wav
//!       and <out_dir>/ours.json (per line: text, IPA ids, ms, audio s)
//!   speak_bench <voice_dir> <reply.txt> <cpu|wgpu> <out_dir> latency <label>   (feature phonemes)
//!       first-audio time of speak_text on a multi-sentence reply (time to the first sentence's PCM), cold and warm
//! voice_dir holds model.safetensors + voice.json (tools/convert.py). ids.json: tools/phonemize.py.
use burn::tensor::Device;
use text_to_speech::{Speaker, SynthOpts};
use std::time::Instant;

#[derive(serde::Deserialize)]
struct Item {
    name: String,
    text: String,
    ids: Vec<i64>,
    #[serde(default)]
    sentences: Vec<Vec<i64>>,
}
#[derive(serde::Deserialize)]
struct Ids {
    items: Vec<Item>,
}

fn log(m: &str) {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    println!("[{t}] {m}");
}

fn write_f32(p: &str, v: &[f32]) {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(p, b).expect(p);
}

fn write_wav(p: &str, pcm: &[f32], sr: u32) {
    let n = pcm.len() as u32;
    let mut b = Vec::with_capacity(44 + 2 * pcm.len());
    b.extend(b"RIFF");
    b.extend((36 + 2 * n).to_le_bytes());
    b.extend(b"WAVEfmt ");
    b.extend(16u32.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend(sr.to_le_bytes());
    b.extend((sr * 2).to_le_bytes());
    b.extend(2u16.to_le_bytes());
    b.extend(16u16.to_le_bytes());
    b.extend(b"data");
    b.extend((2 * n).to_le_bytes());
    for s in pcm {
        b.extend(((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(p, b).expect(p);
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, ids_path, be, out, mode) = (&a[1], &a[2], a[3].as_str(), &a[4], a[5].as_str());
    let dev: Device = match be {
        "cpu" => Device::flex(),
        #[cfg(feature = "gpu")]
        "wgpu" => Device::from(burn::backend::wgpu::WgpuDevice::default()),
        other => panic!("backend {other} not built in"),
    };
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());
    let t = Instant::now();
    let cfg = std::fs::read_to_string(format!("{dir}/voice.json")).expect("voice.json");
    let st = std::fs::read(format!("{dir}/model.safetensors")).expect("model.safetensors");
    let sp = Speaker::load(&dev, &cfg, &st).expect("load");
    let sr = sp.sample_rate();
    log(&format!("loaded in {:.0} ms ({be}, threads {threads}, {sr} Hz, hop {})", ms(t), sp.hop()));
    #[cfg(feature = "phonemes")]
    if mode == "text" || mode == "latency" {
        return text_modes(&sp, ids_path, out, mode, a.get(6).cloned().unwrap_or_else(|| format!("{be}-t{threads}")));
    }
    let ids: Ids = serde_json::from_str(&std::fs::read_to_string(ids_path).expect("ids.json")).expect("ids json");
    std::fs::create_dir_all(out).ok();

    if mode == "parity" {
        for it in &ids.items {
            let t = Instant::now();
            let r = pollster::block_on(sp.synth_detail(&it.ids, &SynthOpts::deterministic(), true)).expect("synth");
            log(&format!("parity {}: {} ids -> {} frames, {} samples in {:.0} ms (ops {}, readbacks {})", it.name, it.ids.len(), r.durations.iter().sum::<u32>(), r.pcm.len(), ms(t), r.ops, r.readbacks));
            let p = |k: &str| format!("{out}/burn_{}.{k}.f32", it.name);
            write_f32(&p("audio"), &r.pcm);
            write_f32(&p("w"), &r.w);
            for (k, v) in &r.stages {
                write_f32(&p(k), v);
            }
            write_wav(&format!("{out}/burn_{}_det.wav", it.name), &r.pcm, sr);
        }
        return;
    }

    let label = a.get(6).cloned().unwrap_or_else(|| format!("{be}-t{threads}"));
    let reps: usize = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(3);
    let opts = SynthOpts { seed: 7, ..sp.default_opts() };
    let t = Instant::now();
    let _ = pollster::block_on(sp.synth(&ids.items[0].ids, &opts)).expect("warm-up");
    log(&format!("{label}: warm-up {:.0} ms", ms(t)));
    let mut rows = Vec::new();
    for it in &ids.items {
        let mut times = Vec::new();
        let mut pcm = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            pcm = pollster::block_on(sp.synth(&it.ids, &opts)).expect("synth");
            times.push(ms(t));
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sec = pcm.len() as f64 / sr as f64;
        let med = times[times.len() / 2];
        // sentence streaming: the first sentence alone is what the player waits for
        let (first_ms, first_sec) = match it.sentences.first() {
            Some(s0) if it.sentences.len() > 1 => {
                let mut best = f64::MAX;
                let mut n = 0;
                for _ in 0..reps {
                    let t = Instant::now();
                    n = pollster::block_on(sp.synth(s0, &opts)).expect("synth").len();
                    best = best.min(ms(t));
                }
                (best, n as f64 / sr as f64)
            }
            _ => (times[0], sec),
        };
        log(&format!("{label} {}: {} ids, {:.2} s audio, median {:.0} ms (min {:.0}), rtf {:.3}; first sentence {:.0} ms for {:.2} s | {}", it.name, it.ids.len(), sec, med, times[0], med / 1e3 / sec, first_ms, first_sec, it.text));
        write_wav(&format!("{out}/{label}_{}.wav", it.name), &pcm, sr);
        rows.push(serde_json::json!({"name": it.name, "ids": it.ids.len(), "sentences": it.sentences.len(), "audio_s": sec, "ms_median": med, "ms_min": times[0], "rtf": med / 1e3 / sec, "first_sentence_ms": first_ms, "first_sentence_audio_s": first_sec}));
    }
    let j = serde_json::json!({"label": label, "backend": be, "threads": threads, "rows": rows});
    std::fs::write(format!("{out}/speed_{label}.json"), serde_json::to_string_pretty(&j).unwrap()).unwrap();
}

#[cfg(feature = "phonemes")]
fn text_modes(sp: &Speaker, input: &str, out: &str, mode: &str, label: String) {
    use text_to_speech::Cancel;
    let sr = sp.sample_rate();
    let opts = SynthOpts { seed: 7, ..sp.default_opts() };
    std::fs::create_dir_all(out).ok();
    let text = std::fs::read_to_string(input).expect("input");
    if mode == "text" {
        let t = Instant::now();
        sp.warm_g2p();
        log(&format!("g2p tables built in {:.0} ms", ms(t)));
        let g2p = text_to_phonemes::G2p::new();
        let mut rows = Vec::new();
        for (n, line) in text.lines().map(str::trim).filter(|l| !l.is_empty()).enumerate() {
            let t = Instant::now();
            let mut pcm = Vec::new();
            let k = pollster::block_on(sp.speak_text(line, &opts, &Cancel::new(), |_, p| {
                pcm.extend(p);
                pcm.extend(std::iter::repeat_n(0.0f32, sr as usize / 5)); // 200 ms between sentences
            })).expect("speak_text");
            let took = ms(t);
            let sec = pcm.len() as f64 / sr as f64;
            write_wav(&format!("{out}/ours_{:02}.wav", n + 1), &pcm, sr);
            log(&format!("text {:02}: {k} sentence(s), {sec:.2} s audio in {took:.0} ms | {line}", n + 1));
            rows.push(serde_json::json!({"n": n + 1, "text": line, "sentences": k, "ipa": g2p.sentences(line), "ids": sp.text_ids(line), "ms": took, "audio_s": sec}));
        }
        std::fs::write(format!("{out}/ours.json"), serde_json::to_string_pretty(&rows).unwrap()).unwrap();
        return;
    }
    // latency: cold = the first call in the process (G2P tables built inside it), then warm reps
    let reply = text.trim();
    let mut first = Vec::new();
    let mut total = Vec::new();
    let mut n_sent = 0;
    for rep in 0..6 {
        let t = Instant::now();
        let mut t_first = None;
        n_sent = pollster::block_on(sp.speak_text(reply, &opts, &Cancel::new(), |i, _| {
            if i == 0 {
                t_first = Some(ms(t));
            }
        })).expect("speak_text");
        let (f, all) = (t_first.unwrap_or(f64::NAN), ms(t));
        log(&format!("{label} latency rep {rep}{}: first sentence PCM at {f:.0} ms, all {n_sent} sentences at {all:.0} ms", if rep == 0 { " (cold)" } else { "" }));
        if rep > 0 {
            first.push(f);
            total.push(all);
        }
        if rep == 0 {
            let j = serde_json::json!({"cold_first_ms": f, "cold_total_ms": all});
            std::fs::write(format!("{out}/latency_{label}_cold.json"), j.to_string()).unwrap();
        }
    }
    first.sort_by(|a, b| a.partial_cmp(b).unwrap());
    total.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let j = serde_json::json!({"label": label, "reply": reply, "sentences": n_sent, "first_ms_median": first[first.len() / 2], "first_ms_min": first[0], "total_ms_median": total[total.len() / 2]});
    log(&format!("{label} latency: {j}"));
    std::fs::write(format!("{out}/latency_{label}.json"), serde_json::to_string_pretty(&j).unwrap()).unwrap();
}
