//! Sentence-by-sentence synthesis for Lindii's replies: the first sentence plays while the rest are made.
//! Interruption (the player talks, the game cuts in) = set the `Cancel`: the sentence in flight finishes, the remaining
//! ones are dropped. A VITS call has no partial output, so a sentence is the unit of latency (README: streaming).
use crate::model::{Speaker, SynthOpts};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Cancel {
        Cancel::default()
    }
    pub fn set(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Synthesizes `sentences` (phoneme ids, one list per sentence) in order, calling `on_audio(i, pcm)` as each is done.
/// Seeds advance per sentence (seed + i) so a reply is reproducible. Returns how many sentences were spoken; stops
/// early (Ok) when `cancel` is set, checked before each sentence.
pub async fn speak_sentences<F: FnMut(usize, Vec<f32>)>(sp: &Speaker, sentences: &[Vec<i64>], opts: &SynthOpts, cancel: &Cancel, mut on_audio: F) -> Result<usize, String> {
    for (i, ids) in sentences.iter().enumerate() {
        if cancel.is_set() {
            return Ok(i);
        }
        let o = SynthOpts { seed: opts.seed.wrapping_add(i as u64), ..*opts };
        let pcm = sp.synth(ids, &o).await?;
        on_audio(i, pcm);
    }
    Ok(sentences.len())
}

#[cfg(feature = "phonemes")]
impl Speaker {
    /// English text in, audio out sentence by sentence (feature `phonemes`): splits `text` into sentences,
    /// phonemizes them with the text-to-phonemes G2P and synthesizes each, calling `on_audio(i, pcm)` as each is done
    /// (play sentence 0 while 1 is made). Same seeds and cancellation as [`speak_sentences`]; returns how many
    /// sentences were spoken.
    pub async fn speak_text<F: FnMut(usize, Vec<f32>)>(&self, text: &str, opts: &SynthOpts, cancel: &Cancel, on_audio: F) -> Result<usize, String> {
        let sentences = self.text_ids(text);
        speak_sentences(self, &sentences, opts, cancel, on_audio).await
    }
}
