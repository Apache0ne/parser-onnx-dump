import onnxruntime as ort
import numpy as np

MODEL = "fashn_parser_fp32.onnx"

sess = ort.InferenceSession(MODEL, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("Providers:", sess.get_providers())

# SegFormer input: NCHW float32
x = np.random.rand(1, 3, 512, 512).astype(np.float32)

y = sess.run(None, {"pixel_values": x})[0]
print("Output shape:", y.shape)
print("dtype:", y.dtype, "min/max:", float(y.min()), float(y.max()))