import onnxruntime as ort
import numpy as np
import time

MODEL = "fashn_parser_fp16.onnx"

print("Creating session...")
sess = ort.InferenceSession(
    MODEL,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

print("Providers:", sess.get_providers())

# FP16 input
x = np.random.rand(1, 3, 512, 512).astype(np.float16)

print("Running FP16 inference...")
t0 = time.time()
y = sess.run(None, {"pixel_values": x})[0]
t1 = time.time()

print("Done.")
print("Output shape:", y.shape)
print("dtype:", y.dtype)
print("Time (s):", round(t1 - t0, 4))
print("Min / Max:", float(y.min()), float(y.max()))