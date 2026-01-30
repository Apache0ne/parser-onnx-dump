import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
from transformers import SegformerImageProcessor
import cv2

# ---------------- CONFIG ----------------
ENGINE_PATH = "fashn_parser_fp16.trt"
IMAGE_PATH = "test_image.jpg"
OUT_IMAGE = "trt_fp16_output.png"
# ---------------------------------------

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
processor = SegformerImageProcessor.from_pretrained("fashn-ai/fashn-human-parser")

def preprocess(img):
    inputs = processor(images=img, return_tensors="np")
    return inputs["pixel_values"].astype(np.float16)

def logits_to_mask(logits):
    return np.argmax(logits, axis=1)[0].astype(np.uint8)

def colorize(mask):
    return cv2.applyColorMap(mask * 10, cv2.COLORMAP_TURBO)

print("🔍 Loading FP16 TensorRT engine...")
with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Load image
img = Image.open(IMAGE_PATH).convert("RGB")
x = preprocess(img)

# Dynamic shape
context.set_input_shape("pixel_values", x.shape)

# CUDA stream
stream = cuda.Stream()

# Allocate buffers
d_input = cuda.mem_alloc(int(x.nbytes))

# SegFormer output is downscaled by 4
out_shape = (1, 18, x.shape[2] // 4, x.shape[3] // 4)
d_output = cuda.mem_alloc(int(np.prod(out_shape) * np.float16().nbytes))

# TensorRT 10 API
context.set_tensor_address("pixel_values", int(d_input))
context.set_tensor_address("logits", int(d_output))

# Copy input
cuda.memcpy_htod_async(d_input, x, stream)

print("🚀 Running FP16 TensorRT inference...")
t0 = time.time()
context.execute_async_v3(stream_handle=stream.handle)
stream.synchronize()
latency_ms = (time.time() - t0) * 1000

# Copy output
output = np.empty(out_shape, dtype=np.float16)
cuda.memcpy_dtoh_async(output, d_output, stream)
stream.synchronize()

# Postprocess
mask = logits_to_mask(output)
mask = cv2.resize(mask, img.size, interpolation=cv2.INTER_NEAREST)
colored = colorize(mask)

Image.fromarray(colored).save(OUT_IMAGE)

print("✅ Inference complete")
print("Output shape:", output.shape)
print("Output dtype:", output.dtype)
print(f"Latency (ms): {latency_ms:.2f}")
print(f"Saved output → {OUT_IMAGE}")