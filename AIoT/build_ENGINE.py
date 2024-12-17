import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(TRT_LOGGER)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
# Set cache
cache = config.create_timing_cache(b"")
config.set_timing_cache(cache, ignore_mismatch=False)

path_onnx_model = "/home/ivsr/tensorRT/model.onnx"
config.max_workspace_size = 1 << 20
flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flag)
print(network.num_inputs)
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(path_onnx_model, "rb") as f:
    if not parser.parse(f.read()):
        print(f"ERROR: Failed to parse the ONNX file {path_onnx_model}")
        for error in range(parser.num_errors):
            print(parser.get_error(error))


inputs = [network.get_input(i) for i in range(network.num_inputs)]
outputs = [network.get_output(i) for i in range(network.num_outputs)]

for input in inputs:
    print(f"Model {input.name} shape: {input.shape} {input.dtype}")
for output in outputs:
    print(f"Model {output.name} shape: {output.shape} {output.dtype}")
    
# Define
channels = 3
width = 224
height = 224
input_model = [channels, height, width]
max_batch_size = 4

shape_input_model = [max_batch_size] + input_model

if max_batch_size > 1:
    # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles
    # To explict batch, set min, opt and max shape
    # This help to TensorRT to search better optimizations
    profile = builder.create_optimization_profile()
    min_shape = [1] + shape_input_model[-3:]
    opt_shape = [int(max_batch_size/2)] + shape_input_model[-3:]
    max_shape = shape_input_model
    for input in inputs:
        profile.set_shape(input.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

config.get_calibration_profile()

######  Option ###########
half = False
int8 = False
if half:
    config.set_flag(trt.BuilderFlag.FP16)
elif int8:
    config.set_flag(trt.BuilderFlag.INT8)

strip_weights = False
if strip_weights:
    config.set_flag(trt.BuilderFlag.STRIP_PLAN)

######################

engine_bytes = builder.build_serialized_network(network, config)
engine_path = "/home/ivsr/tensorRT/model.engine"
with open(engine_path, "wb") as f:
    f.write(engine_bytes)

