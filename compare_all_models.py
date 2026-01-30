import time
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
from transformers import SegformerImageProcessor
import cv2

# ---------------- CONFIG ----------------
IMAGE_PATH = "test_image.jpg"
FP32_ONNX = "fashn_parser_fp32.onnx"
FP16_ONNX = "fashn_parser_fp16.onnx"
INT8_ENGINE = "fashn_parser_int8.trt"
OUT_IMAGE = "comparison_grid.png"

TARGET_SIZE = (512, 512)
NUM_CLASSES = 18
# ----------------------------------------

processor = SegformerImageProcessor.from_pretrained(
    "fashn-ai/fashn-human-parser"
)

# ---------------- HELPERS ----------------
def preprocess(img):
    inputs = processor(images=img, return_tensors="np")
    return inputs["pixel_values"]

def logits_to_mask(logits):
    return np.argmax(logits[0], axis=0).astype(np.uint8)

def resize_mask(mask, size):
    return np.array(
        Image.fromarray(mask).resize(size, Image.NEAREST)
    )

def colorize(mask):
    return cv2.applyColorMap(mask * 10, cv2.COLORMAP_TURBO)

# ---------------- ONNX ----------------
def run_onnx(path, x):
    sess = ort.InferenceSession(
        path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    t0 = time.time()
    y = sess.run(None, {"pixel_values": x})[0]
    return y, (time.time() - t0) * 1000

# ---------------- TENSORRT INT8 ----------------
def run_trt(engine_path, x):
    logger = trt.Logger(trt.Logger.INFO)

    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_input_shape("pixel_values", x.shape)

    stream = cuda.Stream()

    x = x.astype(np.float16)
    d_input = cuda.mem_alloc(int(x.nbytes))

    out_shape = (1, NUM_CLASSES, x.shape[2] // 4, x.shape[3] // 4)
    d_output = cuda.mem_alloc(
        int(np.prod(out_shape) * np.float16().nbytes)
    )

    context.set_tensor_address("pixel_values", int(d_input))
    context.set_tensor_address("logits", int(d_output))

    cuda.memcpy_htod_async(d_input, x, stream)

    t0 = time.time()
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    latency = (time.time() - t0) * 1000

    output = np.empty(out_shape, dtype=np.float16)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    return output, latency

# ---------------- MAIN ----------------
img = Image.open(IMAGE_PATH).convert("RGB").resize(TARGET_SIZE)
img_np = np.array(img)  # ✅ DEFINE EARLY

x = preprocess(img)

fp32_logits, t_fp32 = run_onnx(FP32_ONNX, x.astype(np.float32))
fp16_logits, t_fp16 = run_onnx(FP16_ONNX, x.astype(np.float16))
int8_logits, t_int8 = run_trt(INT8_ENGINE, x)

fp32_mask = resize_mask(logits_to_mask(fp32_logits), TARGET_SIZE)
fp16_mask = resize_mask(logits_to_mask(fp16_logits), TARGET_SIZE)
int8_mask = resize_mask(logits_to_mask(int8_logits), TARGET_SIZE)

fp32_vis = colorize(fp32_mask)
fp16_vis = colorize(fp16_mask)
int8_vis = colorize(int8_mask)

# Grid layout:
# [ Input | FP32 ]
# [ FP16  | INT8 ]
top = np.hstack([img_np, fp32_vis])
bottom = np.hstack([fp16_vis, int8_vis])
grid = np.vstack([top, bottom])

Image.fromarray(grid).save(OUT_IMAGE)

print("\n=== Latency (ms) ===")
print(f"FP32 ONNX: {t_fp32:.2f}")
print(f"FP16 ONNX: {t_fp16:.2f}")
print(f"INT8 TRT : {t_int8:.2f}")
print(f"\nSaved → {OUT_IMAGE}")
