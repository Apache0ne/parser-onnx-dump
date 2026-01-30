import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import numpy as np

MODEL_ID = "fashn-ai/fashn-human-parser"
OUT_PATH = "fashn_parser_fp16.onnx"

device = "cuda"

# Load model directly in FP16
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to(device)
model.eval()

processor = SegformerImageProcessor.from_pretrained(MODEL_ID)

# Dummy image (PIL -> processor -> tensor)
dummy_image = Image.fromarray(
    np.zeros((512, 512, 3), dtype=np.uint8)
)

inputs = processor(images=dummy_image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device).half()

with torch.no_grad():
    torch.onnx.export(
        model,
        (pixel_values,),
        OUT_PATH,
        opset_version=18,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )

print("Exported:", OUT_PATH)
