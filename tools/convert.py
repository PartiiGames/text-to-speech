# Farm-only tool: Piper VITS voice (.onnx + .onnx.json) -> text-to-speech weights (safetensors) + voice json.
# Usage: python -I convert.py <voice.onnx> <voice.onnx.json> <out.safetensors> <out.voice.json>
#
# Named initializers keep their PyTorch module names (enc_p.*, dp.*, flow.*, dec.*). The rest are mapped from the node
# that consumes them (torch.onnx names nodes by module path):
#   onnx::Conv_NNNN (the flow's weight-normed WN convs, folded at export) -> <module path>.weight, from its Conv node
#   sid (the text embedding; the exporter's name)                      -> enc_p.emb.weight, from /enc_p/emb/Gather
#   /dp/flows.0/Exp_output_0 (exp(-logs) of the duration predictor's ElementwiseAffine, folded) -> dp.flows.0.exp_neg_logs
# The architecture numbers that shapes can't give (dilations, strides, paddings) are read from the node attributes and
# written to the voice json under "partii_arch".
import json
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

onnx_path, cfg_path, out_st, out_cfg = sys.argv[1:5]
m = onnx.load(onnx_path)
g = m.graph
init = {i.name: numpy_helper.to_array(i) for i in g.initializer}


def attrs(n):
    return {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}


def modpath(node):  # "/flow/flows.6/enc/in_layers.0/Conv" -> "flow.flows.6.enc.in_layers.0"
    return ".".join(node.name.strip("/").split("/")[:-1])


out = {}
for k, v in init.items():
    if "." in k and not k.startswith("/") and not k.startswith("onnx::"):
        out[k] = v
for n in g.node:
    for j, x in enumerate(n.input):
        if x not in init:
            continue
        if x.startswith("onnx::Conv_") and n.op_type == "Conv":
            out[modpath(n) + (".weight" if j == 1 else ".bias")] = init[x]
        elif n.op_type == "Gather" and n.name == "/enc_p/emb/Gather" and j == 0:
            out["enc_p.emb.weight"] = init[x]
        elif x == "/dp/flows.0/Exp_output_0":
            assert n.op_type == "Mul" and n.name.startswith("/dp/flows.0/"), n.name
            out["dp.flows.0.exp_neg_logs"] = init[x]
mapped_src = set()
for n in g.node:
    for x in n.input:
        if x in init and (x.startswith("onnx::Conv_") or x == "sid" or x == "/dp/flows.0/Exp_output_0"):
            mapped_src.add(x)
left = [k for k, v in init.items() if v.dtype == np.float32 and v.size > 16 and k not in out and k not in mapped_src]
assert not left, f"unmapped float initializers: {left}"

# architecture from the graph
arch = {}
conv = {modpath(n): attrs(n) for n in g.node if n.op_type in ("Conv", "ConvTranspose")}
rk = out["enc_p.encoder.attn_layers.0.emb_rel_k"]
arch["hidden"] = int(out["enc_p.emb.weight"].shape[1])
arch["window"] = int((rk.shape[1] - 1) // 2)
arch["heads"] = arch["hidden"] // int(rk.shape[2])
arch["enc_layers"] = len([k for k in out if k.startswith("enc_p.encoder.attn_layers.") and k.endswith("conv_q.weight")])
arch["ffn_kernel"] = int(out["enc_p.encoder.ffn_layers.0.conv_1.weight"].shape[2])
ups = sorted({int(k.split(".")[2]) for k in out if k.startswith("dec.ups.")})
arch["upsample_rates"] = [int(conv[f"dec.ups.{i}"]["strides"][0]) for i in ups]
arch["upsample_kernels"] = [int(conv[f"dec.ups.{i}"]["kernel_shape"][0]) for i in ups]
arch["upsample_pads"] = [int(conv[f"dec.ups.{i}"]["pads"][0]) for i in ups]
nres = len({int(k.split(".")[2]) for k in out if k.startswith("dec.resblocks.")})
per = nres // len(ups)
rb = []
for r in range(per):
    convs = sorted({int(k.split(".")[4]) for k in out if k.startswith(f"dec.resblocks.{r}.convs.")})
    # resblock "2" has convs.N; resblock "1" would have convs1/convs2 (not this voice)
    assert convs, "resblock type 1 not supported yet"
    rb.append({"kernel": int(conv[f"dec.resblocks.{r}.convs.0"]["kernel_shape"][0]),
               "dilations": [int(conv[f"dec.resblocks.{r}.convs.{c}"]["dilations"][0]) for c in convs]})
arch["resblocks"] = rb
fl = sorted({int(k.split(".")[2]) for k in out if k.startswith("flow.flows.") and k.endswith(".pre.weight")})
arch["flows"] = fl
arch["wn_layers"] = len([k for k in out if k.startswith(f"flow.flows.{fl[0]}.enc.in_layers.") and k.endswith(".weight")])
arch["wn_kernel"] = int(out[f"flow.flows.{fl[0]}.enc.in_layers.0.weight"].shape[2])
arch["wn_dilations"] = [int(conv[f"flow.flows.{fl[0]}.enc.in_layers.{i}"]["dilations"][0]) for i in range(arch["wn_layers"])]
arch["dp_flows"] = sorted({int(k.split(".")[2]) for k in out if k.startswith("dp.flows.") and k.endswith(".pre.weight")}, reverse=True)
arch["dds_dilations"] = [int(conv[f"dp.convs.convs_sep.{i}"]["dilations"][0]) for i in range(3)]
arch["dp_bins"] = int((out[f"dp.flows.{arch['dp_flows'][0]}.proj.weight"].shape[0] + 1) // 3)
arch["tail_bound"] = 5.0
print("arch", json.dumps(arch))

out = {k: np.ascontiguousarray(v.astype(np.float32)) for k, v in out.items()}
save_file(out, out_st)
cfg = json.load(open(cfg_path))
cfg["partii_arch"] = arch
json.dump(cfg, open(out_cfg, "w"), ensure_ascii=False, indent=1)
print("tensors", len(out), "params", sum(v.size for v in out.values()))
