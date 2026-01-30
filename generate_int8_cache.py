import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor

ONNX_PATH = "fashn_parser_fp16.onnx"
CACHE_FILE = "segformer_int8.cache"
CALIB_DIR = "calib_images"

BATCH = 1
HEIGHT = 512
WIDTH = 512

processor = SegformerImageProcessor.from_pretrained("fashn-ai/fashn-human-parser")

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self):
        super().__init__()
        self.files = [os.path.join(CALIB_DIR, f) for f in os.listdir(CALIB_DIR)]
        self.index = 0
        self.device_input = cuda.mem_alloc(BATCH * 3 * HEIGHT * WIDTH * np.float32().nbytes)

    def get_batch_size(self):
        return BATCH

    def get_batch(self, names):
        if self.index >= len(self.files):
            return None

        img = Image.open(self.files[self.index]).convert("RGB")
        inputs = processor(images=img, return_tensors="np")
        data = inputs["pixel_values"].astype(np.float32)

        cuda.memcpy_htod(self.device_input, data)
        self.index += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(CACHE_FILE, "wb") as f:
            f.write(cache)


print("Building INT8 calibration cache...")

with trt.Builder(TRT_LOGGER) as builder, \
     builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
     trt.OnnxParser(network, TRT_LOGGER) as parser:

    with open(ONNX_PATH, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = Calibrator()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    profile = builder.create_optimization_profile()
    profile.set_shape("pixel_values",
                      (1, 3, 256, 256),
                      (1, 3, 512, 512),
                      (1, 3, 1024, 1024))
    config.add_optimization_profile(profile)

    builder.build_serialized_network(network, config)

print("Calibration cache written:", CACHE_FILE)