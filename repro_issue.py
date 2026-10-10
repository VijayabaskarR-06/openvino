

import numpy as np
import openvino as ov
import openvino.opset13 as ops

def c(v):
    return ops.constant(np.asarray(v, np.float32))

def fq(node, il, ih, ol, oh, levels):
    return ops.fake_quantize(node, c(il), c(ih), c(ol), c(oh), levels)

def fq_ref(x, il, ih, ol, oh, levels):
    il, ih, ol, oh = (np.asarray(v, np.float32) for v in (il, ih, ol, oh))
    q = np.round((x - il) / (ih - il) * (levels - 1)) / (levels - 1) * (oh - ol) + ol
    return np.where(x <= np.minimum(il, ih), ol, np.where(x > np.maximum(il, ih), oh, q))

core = ov.Core()

p = ops.parameter([4], ov.Type.f32)
args = (0.0, 2.0, -1.0, 1.0, 3)
model = ov.Model([ops.abs(fq(fq(p, *args), *args))], [p])
x = np.array([0, 1, 1.4, 2], np.float32)
print("OpenVINO", ov.get_version())
print("case 1 expected", np.abs(fq_ref(fq_ref(x, *args), *args)))
print("case 1 actual  ", core.compile_model(model, "CPU")(x)[0])

p = ops.parameter([1, 2, 2, 1], ov.Type.f32)
low = np.zeros((1, 2, 1, 1), np.float32)
high = np.array([1, 2], np.float32).reshape(1, 2, 1, 1)
args = (low, high, low, high, 3)
shape = ops.constant(np.array([2, 2, 1, 1], np.int64))
model = ov.Model([ops.abs(fq(ops.reshape(fq(p, *args), shape, False), *args))], [p])
x = np.array([0.4, 0.6, 1.9, 1.1], np.float32).reshape(1, 2, 2, 1)
print("case 2 expected", np.abs(fq_ref(fq_ref(x, *args).reshape(2, 2, 1, 1), *args)).ravel())
print("case 2 actual  ", core.compile_model(model, "CPU")(x)[0].ravel())

