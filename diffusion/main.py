import logging
import os.path
import cuda
import torch
from diffusers.utils import CONFIG_NAME
import pynvml

import onnxruntime as ort
import json
from onnx import TensorProto
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from onnxruntime.transformers.io_binding_helper import TypeHelper
from diffusers import FluxPipeline, StableDiffusion3Pipeline, SD3Transformer2DModel, FluxTransformer2DModel, AutoencoderKL


def torch_to_onnx_type(torch_dtype):
    if torch_dtype == torch.float32:
        return TensorProto.FLOAT
    elif torch_dtype == torch.float16:
        return TensorProto.FLOAT16
    elif torch_dtype == torch.bfloat16:
        return TensorProto.BFLOAT16
    elif torch_dtype == torch.int8:
        return TensorProto.int8
    elif torch_dtype == torch.int32:
        return TensorProto.INT32
    elif torch_dtype == torch.int64:
        return TensorProto.INT64
    else:
        raise TypeError(f"Unsupported dtype: {torch_dtype}")


class OrtWrapper(ModelMixin, ConfigMixin):
    config_name = CONFIG_NAME

    def __init__(self, onnx_path, execution_provider, session_options=ort.SessionOptions(), provider_options={}):
        super().__init__()

        session_options.add_session_config_entry("session.use_env_allocators", "1")
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.stream = torch.cuda.current_stream()
        
        # Create CUDA events for timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        provider_options["user_compute_stream"] = str(self.stream.cuda_stream)
        self.session = ort.InferenceSession(onnx_path,
                                            session_options=session_options,
                                            providers=execution_provider,
                                            provider_options=[provider_options, ])
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.input_shapes = {input_key.name: input_key.shape for idx, input_key in
                             enumerate(self.session.get_inputs())}
        self.input_dtypes = {input_key.name: input_key.type for input_key in self.session.get_inputs()}

        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        self.output_shapes = {output_key.name: output_key.shape for idx, output_key in
                              enumerate(self.session.get_outputs())}
        self.fixed_output_shapes = True
        for name, shape in self.output_shapes.items():
            for dim in shape:
                if isinstance(dim, str):
                    self.fixed_output_shapes = False

        self.output_dtypes = {output_key.name: output_key.type for output_key in self.session.get_outputs()}
        # this is needed to identify the data type of the model
        for name, dtype in self.input_dtypes.items():
            self.register_buffer(
                name,
                torch.zeros(1, device="cuda", dtype=TypeHelper.ort_type_to_torch_type(dtype)),
                persistent=False
            )

    def decode(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, run_options=ort.RunOptions(), **kwargs):
        # Record start time
        self.start_event.record(stream=self.stream)
        
        binding = self.session.io_binding()
        args_idx = 0
        device = None
        for name in self.input_names.keys():
            value = kwargs.get(name, None)
            if value is None:
                value = args[args_idx]
            expected_torch_dtype = TypeHelper.ort_type_to_torch_type(self.input_dtypes[name])

            value = value.to(dtype=expected_torch_dtype).contiguous()
            binding.bind_input(
                name=name,
                device_type='cuda',
                device_id=0,
                element_type=torch_to_onnx_type(value.dtype),
                shape=tuple(value.shape),
                buffer_ptr=value.data_ptr(),
            )
            if (device is not None and device != value.device):
                logging.warning("Inputs are not on the same device. This may lead to errors")
            device = value.device
        torch_outputs = {}
        for name in self.output_names:
            if self.fixed_output_shapes:
                torch_dtype = TypeHelper.ort_type_to_torch_type(self.output_dtypes[name])
                torch_outputs[name] = torch.empty(self.output_shapes[name],
                                                  device=device,
                                                  dtype=torch_dtype)
                binding.bind_output(
                    name=name,
                    device_type='cuda',
                    device_id=0,
                    element_type=torch_to_onnx_type(torch_dtype),
                    shape=torch_outputs[name].shape,
                    buffer_ptr=torch_outputs[name].data_ptr(),
                )
            else:
                binding.bind_output(
                    name=name,
                    device_type='cuda'
                )
        self.session.run_with_iobinding(binding)
        if not self.fixed_output_shapes:
            outputs = binding.get_outputs()
            for name, idx in self.output_names.items():
                ort_value = outputs[idx]
                shape = ort_value.shape()
                torch_outputs[name] = torch.empty(shape,
                                                  device=device,
                                                  dtype=TypeHelper.ort_type_to_torch_type(ort_value.data_type()))
                raise NotImplementedError() # maybe use dl_pack ?
                # https://github.com/microsoft/onnxruntime/blob/7e2408880e963bcfdd2b898c7b6464506545cec2/onnxruntime/python/onnxruntime_pybind_ortvalue.cc#L425
                cuda.bindings.runtime.cudaMemcpyAsync(
                    torch_outputs[name].data_ptr(),
                    ort_value.data_ptr(), self.stream.cuda_stream)
                    
        # Record end time and synchronize
        self.end_event.record(stream=self.stream)
        self.end_event.synchronize()
        
        # Calculate elapsed time in milliseconds
        elapsed_time = self.start_event.elapsed_time(self.end_event)
        print(f"Forward pass took {elapsed_time:.2f} ms")
        
        return list(torch_outputs.values())

class TransfomerOrt(OrtWrapper, FluxTransformer2DModel):
    pass

class VaeOrt(OrtWrapper, AutoencoderKL):
    pass


#model_id = os.path.abspath("N:/hf/flux.1-dev")
model_id = os.path.abspath("C:/flux.1-dev")

def get_vae():
    vae_onnx_ctx = "vae_ctx.onnx"
    config_path = os.path.join(model_id,"vae")
    with open(os.path.join(config_path, CONFIG_NAME), "r") as f:
        json_config = json.load(f)
    ort_vae_model = VaeOrt.from_config(json_config,
                                               onnx_path=vae_onnx_ctx,
                                               execution_provider=["NvTensorRTRTXExecutionProvider"])
    return ort_vae_model


def get_transformer():
    transformer_onnx_ctx = "transformer_ctx.onnx"
    config_path = os.path.join(model_id,"transformer")
    with open(os.path.join(config_path, CONFIG_NAME), "r") as f:
        json_config = json.load(f)
    return TransfomerOrt.from_config(json_config,
                                  onnx_path=transformer_onnx_ctx,
                                  execution_provider=["NvTensorRTRTXExecutionProvider"])

def print_gpu_memory_stats(handle):
    print("\nPyTorch GPU Memory Statistics:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print(f"Max Cached: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
    
    # Get memory info
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    print("\nSystem GPU Memory Statistics:")
    print(f"Total: {memory_info.total / 1024**2:.2f} MB")
    print(f"Used: {memory_info.used / 1024**2:.2f} MB")
    print(f"Free: {memory_info.free / 1024**2:.2f} MB")
    
    # Get utilization info
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"GPU Utilization: {utilization.gpu}%")
    print(f"Memory Utilization: {utilization.memory}%")
    


if __name__ == "__main__":
    
    # Initialize NVML
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0) 


    print_gpu_memory_stats(handle)
    pipe = FluxPipeline.from_pretrained(model_id,
                                        torch_dtype=torch.bfloat16,
                                        vae=get_vae(),
                                        transformer=get_transformer()
                                        ).to("cuda:0")

    # Create CUDA events for timing the entire pipeline
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    prompt = "A cat holding a sign that says hello world"

    for i in range(10):
        # Record start time on default stream
        start_event.record(stream=torch.cuda.default_stream())
        

        out = pipe(
            prompt=prompt,
            guidance_scale=0.,
            height=1024,
            width=1024,
            num_inference_steps=40,
        ).images[0]

        # Record end time on default stream and synchronize
        end_event.record(stream=torch.cuda.default_stream())
        end_event.synchronize()

        # Calculate and print total pipeline time
        total_time = start_event.elapsed_time(end_event)
        print(f"Total pipeline execution took {total_time:.2f} ms")

        # Print GPU memory statistics
        print_gpu_memory_stats(handle)

    out.save("image.png")



    # pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
    #                                                 torch_dtype=torch.bfloat16)
    # pipe = pipe.to("cuda")
    #
    # out = pipe(
    #     "A cat holding a sign that says hello world",
    #     negative_prompt="",
    #     num_inference_steps=28,
    #     guidance_scale=7.0,
    # ).images[0]
    # out.save("image.png")

    # Shutdown NVML
    pynvml.nvmlShutdown()