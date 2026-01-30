import tensorrt as trt

ONNX_PATH = "fashn_parser_fp16.onnx"
ENGINE_PATH = "fashn_parser_fp16.trt"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

print("🔧 Building TensorRT FP16 engine...")

with trt.Builder(TRT_LOGGER) as builder, \
     builder.create_network(
         1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
     ) as network, \
     trt.OnnxParser(network, TRT_LOGGER) as parser:

    # Parse ONNX
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ Failed to parse ONNX:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    # Builder config
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # Dynamic shape profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "pixel_values",
        (1, 3, 256, 256),
        (1, 3, 512, 512),
        (1, 3, 1024, 1024),
    )
    config.add_optimization_profile(profile)

    # Build engine
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("❌ Engine build failed")

    with open(ENGINE_PATH, "wb") as f:
        f.write(engine_bytes)

print(f"✅ FP16 TensorRT engine saved → {ENGINE_PATH}")