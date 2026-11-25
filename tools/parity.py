# Farm-only, test-only: the voice's .onnx through onnxruntime (the reference) vs text-to-speech's dumps.
#   python -I parity.py ref <voice.onnx> <ids.json> <dir>   -> <dir>/ref_<name>.{audio,w,stats,z}.f32 (+ .json shapes)
#   python -I parity.py cmp <ids.json> <dir>                -> parity numbers per stage (burn_<name>.* written by speak_bench)
# Deterministic: scales = [noise_scale 0, length_scale 1, noise_w 0].
import json
import sys

import numpy as np


def stage_names(g):
    names = {}
    for n in g.node:
        if n.name == "/enc_p/proj/Conv":
            names["stats"] = n.output[0]
        if n.op_type == "Ceil":
            names["w"] = n.input[0]
        if n.name == "/dec/conv_pre/Conv":
            names["z"] = n.input[0]
    return names


def ref(onnx_path, ids_path, d):
    import onnx
    import onnxruntime as ort
    m = onnx.load(onnx_path)
    names = stage_names(m.graph)
    for k, v in names.items():
        m.graph.output.append(onnx.helper.make_empty_tensor_value_info(v))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    outs = ["output"] + list(names.values())
    for it in json.load(open(ids_path))["items"]:
        x = np.array([it["ids"]], dtype=np.int64)
        r = s.run(outs, {"input": x, "input_lengths": np.array([x.shape[1]], dtype=np.int64),
                         "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32)})
        shapes = {}
        for k, v in zip(["audio"] + list(names.keys()), r):
            v = np.asarray(v, dtype=np.float32)
            v.tofile(f"{d}/ref_{it['name']}.{k}.f32")
            shapes[k] = list(v.shape)
        json.dump(shapes, open(f"{d}/ref_{it['name']}.json", "w"))
        print("ref", it["name"], shapes, "durations", int(np.ceil(r[outs.index(names['w'])]).sum()))


def snr(ref, x):
    e = ref - x
    return 10 * np.log10((ref ** 2).sum() / max((e ** 2).sum(), 1e-30))


def cmp(ids_path, d):
    rows = []
    for it in json.load(open(ids_path))["items"]:
        n = it["name"]
        rw = np.fromfile(f"{d}/ref_{n}.w.f32", dtype=np.float32).ravel()
        bw = np.fromfile(f"{d}/burn_{n}.w.f32", dtype=np.float32).ravel()
        rd, bd = np.ceil(rw).astype(int), np.ceil(bw).astype(int)
        row = {"name": n, "ids": len(it["ids"]), "dur_equal": bool((rd == bd).all()), "dur_mismatches": int((rd != bd).sum()),
               "frames_ref": int(rd.sum()), "frames_burn": int(bd.sum()), "w_maxabs": float(np.abs(rw - bw).max()),
               "w_min_dist_to_int": float(np.min(np.abs(rw - np.round(rw))))}
        for k in ["stats", "z", "audio"]:
            r = np.fromfile(f"{d}/ref_{n}.{k}.f32", dtype=np.float32).ravel()
            b = np.fromfile(f"{d}/burn_{n}.{k}.f32", dtype=np.float32).ravel()
            if r.size != b.size:
                row[k] = f"size {r.size} vs {b.size}"
                continue
            row[k] = {"maxabs": float(np.abs(r - b).max()), "snr_db": round(float(snr(r, b)), 2), "ref_peak": float(np.abs(r).max())}
        rows.append(row)
        print(json.dumps(row))
    json.dump(rows, open(f"{d}/parity.json", "w"), indent=1)


if sys.argv[1] == "ref":
    ref(*sys.argv[2:5])
else:
    cmp(*sys.argv[2:4])
