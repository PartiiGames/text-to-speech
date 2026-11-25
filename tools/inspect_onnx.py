# Farm-only, test-only: dump the Piper voice's ONNX graph (initializers, nodes with module paths and attributes).
# Usage: python inspect_onnx.py <voice.onnx> <out.txt>
import sys
import onnx
from onnx import numpy_helper

m = onnx.load(sys.argv[1])
g = m.graph
init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
consts = {}
for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                consts[n.output[0]] = numpy_helper.to_array(a.t)
with open(sys.argv[2], "w") as f:
    f.write(f"opset {[(o.domain, o.version) for o in m.opset_import]} producer {m.producer_name} {m.producer_version}\n")
    for i in g.input:
        f.write(f"INPUT {i.name} {i.type}\n")
    for o in g.output:
        f.write(f"OUTPUT {o.name} {o.type}\n")
    tot = 0
    for k, v in init.items():
        tot += v.size
        f.write(f"INIT {k} {list(v.shape)} {v.dtype}\n")
    f.write(f"INIT total params {tot}\n")
    for n in g.node:
        if n.op_type == "Constant":
            v = consts[n.output[0]]
            s = v.tolist() if v.size <= 8 else f"<{list(v.shape)} {v.dtype}>"
            f.write(f"NODE Constant {n.name} -> {n.output[0]} = {s}\n")
            continue
        ins = []
        for x in n.input:
            if x in init:
                ins.append(f"{x}{list(init[x].shape)}")
            elif x in consts and consts[x].size <= 8:
                ins.append(f"{x}={consts[x].tolist()}")
            else:
                ins.append(x)
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        attrs = {k: (v if not isinstance(v, (onnx.TensorProto, onnx.GraphProto)) else "<t>") for k, v in attrs.items()}
        f.write(f"NODE {n.op_type} {n.name} [{', '.join(ins)}] -> {list(n.output)} {attrs}\n")
print("nodes", len(g.node), "initializers", len(init))
