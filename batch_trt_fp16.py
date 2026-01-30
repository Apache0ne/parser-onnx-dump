import os
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
INPUT_DIR = "calib_images"
OUTPUT_DIR = "output_fp16"
FONT_SCALE = 0.7
FONT_THICKNESS = 2
# ---------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
processor = SegformerImageProcessor.from_pretrained("fashn-ai/fashn-human-parser")

def preprocess(img):
    inputs = processor(images=img, return_tensors="np")
    return inputs["pixel_values"].astype(np.float16)

def logits_to_mask(logits):
    return np.argmax(logits, axis=1)[0].astype(np.uint8)

def colorize(mask):
    return cv2.applyColorMap(mask * 10, cv2.COLORMAP_TURBO)

def overlay_time(img, ms):
    text = f"{ms:.2f} ms"
    cv2.putText(
        img,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return img

# -------- Load TensorRT engine once --------
print("🔍 Loading TensorRT FP16 engine...")
with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
stream = cuda.Stream()

print(f"📂 Processing images from: {INPUT_DIR}")

for name in os.listdir(INPUT_DIR):
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue

    img_path = os.path.join(INPUT_DIR, name)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    x = preprocess(img)
    context.set_input_shape("pixel_values", x.shape)

    # Allocate buffers
    d_input = cuda.mem_alloc(int(x.nbytes))
    out_shape = (1, 18, x.shape[2] // 4, x.shape[3] // 4)
    d_output = cuda.mem_alloc(int(np.prod(out_shape) * np.float16().nbytes))

    context.set_tensor_address("pixel_values", int(d_input))
    context.set_tensor_address("logits", int(d_output))

    cuda.memcpy_htod_async(d_input, x, stream)

    # Inference
    t0 = time.time()
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    latency_ms = (time.time() - t0) * 1000

    output = np.empty(out_shape, dtype=np.float16)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    mask = logits_to_mask(output)
    mask = cv2.resize(mask, img.size, interpolation=cv2.INTER_NEAREST)
    colored = colorize(mask)

    blended = cv2.addWeighted(img_np, 0.45, colored, 0.55, 0)
    blended = overlay_time(blended, latency_ms)

    out_path = os.path.join(OUTPUT_DIR, name)
    Image.fromarray(blended).save(out_path)

    print(f"✅ {name} — {latency_ms:.2f} ms")

print("\n🎉 Batch processing complete.")
