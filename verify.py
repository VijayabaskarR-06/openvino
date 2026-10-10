import itertools
import re
import sys

import numpy as np
import openvino as ov
import openvino.opset13 as ops

rng = np.random.default_rng(2026)
core = ov.Core()


def const(v):
    return ops.constant(np.asarray(v, np.float32))


LINKS = {
    "direct": ([1, 3, 3, 3], lambda n: n),
    "transpose": ([1, 3, 3, 3], lambda n: ops.transpose(n, ops.constant(np.array([0, 2, 1, 3], np.int64)))),
    "reshape": ([1, 3, 3, 1], lambda n: ops.reshape(n, ops.constant(np.array([3, 3, 1, 1], np.int64)), False)),
}


def ranges(per_channel, same_range):
    if per_channel:
        il = np.array([-0.83, -1.21, -2.57], np.float32).reshape(1, 3, 1, 1)
        ih = np.array([1.0, 1.41, 2.93], np.float32).reshape(1, 3, 1, 1)
        ol, oh = (il, ih) if same_range else (il * 0.61, ih * 1.73)
    else:
        il, ih = np.float32(-0.83), np.float32(1.17)
        ol, oh = (il, ih) if same_range else (np.float32(-0.37), np.float32(2.91))
    return il, ih, ol, oh


def build(link, per_channel, same_range, levels):
    shape, chain = LINKS[link]
    r = ranges(per_channel, same_range)
    p = ops.parameter(shape, ov.Type.f32)
    fq1 = ops.fake_quantize(p, *map(const, r), levels)
    fq1.set_friendly_name("first_fq")
    fq2 = ops.fake_quantize(chain(fq1), *map(const, r), levels)
    fq2.set_friendly_name("second_fq")
    return ov.Model([ops.abs(fq2)], [p]), shape


def fq_count(compiled):
    names = []
    for op in compiled.get_runtime_model().get_ordered_ops():
        names += op.get_rt_info()["originalLayersNames"].astype(str).split(",")
    return sum(1 for n in set(names) if n in ("first_fq", "second_fq") or re.fullmatch(r"FakeQuantize_\d+", n))


def reference(model, x):
    out = ov.Tensor(ov.Type.f32, model.output(0).get_partial_shape().to_shape())
    model.evaluate([out], [ov.Tensor(x)])
    return out.data.copy()


print(f"OpenVINO {ov.get_version()}")
print(f"{'link':10}{'ranges':12}{'output range':14}{'levels':>6}   {'FQ2 in CPU graph':18}{'max diff vs reference':>22}   verdict")
bad = 0
for link, per_channel, same_range, levels in itertools.product(LINKS, [False, True], [True, False], [3, 16, 256]):
    model, shape = build(link, per_channel, same_range, levels)
    compiled = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    kept = fq_count(compiled) >= 2
    diff = 0.0
    for _ in range(50):
        x = rng.uniform(-4, 4, shape).astype(np.float32)
        diff = max(diff, float(np.abs(compiled(x)[0] - reference(model, x)).max()))
    ok = diff < 1e-4
    bad += not ok
    print(f"{link:10}{'per_channel' if per_channel else 'per_tensor':12}{'same' if same_range else 'different':14}{levels:>6}   "
          f"{'kept' if kept else 'removed':18}{diff:>22.6f}   {'OK' if ok else 'WRONG'}")
print(f"wrong results: {bad} of 36")
sys.exit(1 if bad else 0)
