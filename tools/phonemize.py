# Farm-only, test-only: the bench/parity sentences as Piper phoneme ids. Never a crate dependency; Partii's own G2P
# replaces this. Uses piper-phonemize (rhasspy, MIT; it runs espeak-ng) when its wheel installs, else hand-written IPA
# in espeak en-us style. Never piper-tts >= 1.3 (OHF-Voice/piper1-gpl, GPL).
# Usage: python -I phonemize.py <voice.onnx> <voice.onnx.json> <out ids.json>
import json
import sys

SENTENCES = [
    ("short", "Hello! I'm Lindii."),
    ("parity", "Hello! I'm Lindii. Let's make a game together."),
    ("long", "Today we're going to build a brand new game together, with a castle on a hill, a friendly dragon, "
             "and a secret door that only opens when you sing to it."),
]
FALLBACK = {
    "short": "həlˈoʊ! aɪm lˈɪndi.",
    "parity": "həlˈoʊ! aɪm lˈɪndi. lˈɛts mˈeɪk ɐ ɡˈeɪm təɡˈɛðɚ.",
    "long": "tədˈeɪ wɪɹ ɡˌoʊɪŋ tə bˈɪld ɐ bɹˈænd nˈuː ɡˈeɪm təɡˈɛðɚ, wɪð ɐ kˈæsəl ˌɑːn ɐ hˈɪl, ɐ fɹˈɛndli dɹˈæɡən, "
            "ænd ɐ sˈiːkɹət dˈoːɹ ðæt ˈoʊnli ˈoʊpənz wɛn juː sˈɪŋ tʊ ɪt.",
}
cfg = json.load(open(sys.argv[2]))
pmap = cfg["phoneme_id_map"]


def to_ids(phonemes):  # Piper's layout: ^ _ p1 _ p2 _ ... pn _ $
    ids = list(pmap["^"]) + list(pmap["_"])
    for p in phonemes:
        if p in pmap:
            ids += list(pmap[p]) + list(pmap["_"])
    return ids + list(pmap["$"])


out = []
src = "piper-phonemize"
try:
    from piper_phonemize import phonemize_espeak
    for name, text in SENTENCES:
        sents = phonemize_espeak(text, cfg["espeak"]["voice"])  # one phoneme list per sentence
        flat = []
        for i, s in enumerate(sents):
            flat += ([" "] if i else []) + list(s)
        out.append({"name": name, "text": text, "phonemes": "".join(flat), "ids": to_ids(flat),
                    "sentences": [to_ids(list(s)) for s in sents]})
except Exception as e:  # noqa: BLE001
    print("piper-phonemize unavailable, using the fallback IPA:", e)
    src = "fallback"
    out = []
    for name, text in SENTENCES:
        ph = FALLBACK[name]
        out.append({"name": name, "text": text, "phonemes": ph, "ids": to_ids(list(ph)), "sentences": []})
for o in out:
    print(o["name"], len(o["ids"]), "ids |", o["phonemes"])
json.dump({"source": src, "items": out}, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
