//! Speak a line to a WAV file, with the weights fetched and checked by sha256.
//!
//!   cargo run --release --example speak -- <voice_dir> "həlˈoʊ! aɪm lˈɪndi." out.wav
//!
//! `<voice_dir>` holds `voice.json` + `model.safetensors` (tools/convert.py output). A missing file is fetched with
//! `curl` from Partii's model host, `https://api.partii.games/v1/ai/text-to-speech/<sha256>/<file>`
//! (`SPEAK_VOICE_HOST=https://api-staging.partii.games` for staging). Every file must match its pinned sha256 below,
//! wherever it came from; nothing is loaded otherwise. The crate itself never fetches anything. Upstream source (the
//! fallback): the Piper .onnx at the pinned Hugging Face revision in NOTICE, converted with tools/convert.py.
use text_to_speech::Speaker;
use sha2::{Digest, Sha256};
use std::path::Path;

/// en_US-hifi_tts_92-medium, converted by tools/convert.py (NOTICE has the source files' hashes).
const FILES: [(&str, &str); 2] = [
    ("voice.json", "7b917e9613a059540a370a1231dddb3e92adb1b92c440671f853dc798981d557"),
    ("model.safetensors", "5b560a49924b7a80065638c4404eee086ee13c26dd6166907c580f863dad693e"),
];

fn fetch_checked(dir: &Path, name: &str, sha: &str) -> Result<Vec<u8>, String> {
    let p = dir.join(name);
    if !p.exists() {
        let host = std::env::var("SPEAK_VOICE_HOST").unwrap_or_else(|_| "https://api.partii.games".into());
        let url = format!("{host}/v1/ai/text-to-speech/{sha}/{name}");
        eprintln!("fetching {url}");
        let part = dir.join(format!("{name}.part"));
        let ok = std::process::Command::new("curl").args(["-fsSL", "--compressed", "--retry", "3", "-o"]).arg(&part).arg(&url).status().map_err(|e| format!("curl: {e}"))?;
        if !ok.success() {
            return Err(format!("fetch {url} failed"));
        }
        std::fs::rename(&part, &p).map_err(|e| e.to_string())?;
    }
    let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let got: String = Sha256::digest(&b).iter().map(|x| format!("{x:02x}")).collect();
    if got != sha {
        return Err(format!("{name}: sha256 {got}, want {sha}"));
    }
    Ok(b)
}

fn write_wav(p: &str, pcm: &[f32], sr: u32) -> std::io::Result<()> {
    let n = pcm.len() as u32;
    let mut b = Vec::with_capacity(44 + 2 * pcm.len());
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + 2 * n).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&sr.to_le_bytes());
    b.extend_from_slice(&(sr * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(2 * n).to_le_bytes());
    for s in pcm {
        b.extend(((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(p, b)
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 {
        return Err("usage: speak <voice_dir> <espeak IPA phonemes> <out.wav>".into());
    }
    let dir = Path::new(&a[1]);
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let json = fetch_checked(dir, FILES[0].0, FILES[0].1)?;
    let weights = fetch_checked(dir, FILES[1].0, FILES[1].1)?;
    let json = String::from_utf8(json).map_err(|e| e.to_string())?;
    let sp = Speaker::load(&burn::tensor::Device::flex(), &json, &weights)?;
    let ids = sp.phoneme_ids(&a[2]);
    let t = std::time::Instant::now();
    let pcm = pollster::block_on(sp.synth(&ids, &sp.default_opts()))?;
    let secs = pcm.len() as f64 / sp.sample_rate() as f64;
    println!("{} ids -> {secs:.2} s of audio in {:.0} ms", ids.len(), t.elapsed().as_secs_f64() * 1e3);
    write_wav(&a[3], &pcm, sp.sample_rate()).map_err(|e| e.to_string())
}
