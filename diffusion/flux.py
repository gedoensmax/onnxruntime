#!/usr/bin/env python3
import onnxruntime as ort
from onnx import TensorProto
import torch
import argparse
import os
import time
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Any

class FluxTextToImagePipeline:
    """
    Custom pipeline for running Flux ONNX models for text-to-image generation.
    Uses PyTorch tensors directly on device with ONNX Runtime IO Bindings.
    Handles dynamic shapes across models and uses IO bindings for performance.
    """
    def __init__(
            self,
            text_encoder_path: str,
            transformer_path: str,
            vae_decoder_path: str,
            device: str = "cuda",
            gpu_id: int = 0,
            fp16: bool = False,
            verbose: bool = False,
            encoder_type: str = "clip",  # or "t5"
    ):
        """
        Initialize the Flux text-to-image pipeline.

        Args:
            text_encoder_path: Path to the text encoder ONNX model (CLIP or T5)
            transformer_path: Path to the transformer ONNX model
            vae_decoder_path: Path to the VAE decoder ONNX model
            device: Device to run inference on ('cuda' or 'cpu')
            gpu_id: GPU ID to use if device is 'cuda'
            fp16: Whether to use FP16 precision
            verbose: Whether to print detailed information
            encoder_type: Type of text encoder to use ('clip' or 't5')
        """
        self.device = device
        self.device_torch = torch.device(device if device != "cuda" else f"cuda:{gpu_id}")
        self.gpu_id = gpu_id
        self.fp16 = fp16
        self.verbose = verbose
        self.encoder_type = encoder_type

        # Set up data type mappings using ONNX TensorProto
        self._setup_dtype_mapping()

        # Model paths
        self.text_encoder_path = Path(text_encoder_path)
        self.transformer_path = Path(transformer_path)
        self.vae_decoder_path = Path(vae_decoder_path)

        # Default parameters for the pipeline
        self.latent_height = 16
        self.latent_width = 16
        self.latent_channels = 16
        self.max_length = 512  # Based on the model specifications

        # Model parameters
        self.height = 512
        self.width = 512

        # Initialize ORT sessions
        self._setup_sessions()

        # Create IO bindings for each session
        self._create_io_bindings()

        # Extract model I/O information
        self._extract_model_info()

        if self.verbose:
            self._print_model_info()

    def _setup_dtype_mapping(self):
        """Set up mapping between PyTorch, NumPy, and ONNX TensorProto data types."""
        # Add bfloat16 support
        # Note: PyTorch's bfloat16 requires PyTorch 1.10+ and CUDA 11.0+
        self.torch_to_onnx_dtype = {
            torch.float32: TensorProto.FLOAT,
            torch.float16: TensorProto.FLOAT16,
            torch.bfloat16: TensorProto.BFLOAT16,
            torch.float64: TensorProto.DOUBLE,
            torch.int32: TensorProto.INT32,
            torch.int64: TensorProto.INT64,
            torch.uint8: TensorProto.UINT8,
            torch.int8: TensorProto.INT8,
            torch.bool: TensorProto.BOOL,
        }

        # ONNX TensorProto to PyTorch mapping
        self.onnx_to_torch_dtype = {
            TensorProto.FLOAT: torch.float32,
            TensorProto.FLOAT16: torch.float16,
            TensorProto.BFLOAT16: torch.bfloat16,
            TensorProto.DOUBLE: torch.float64,
            TensorProto.INT32: torch.int32,
            TensorProto.INT64: torch.int64,
            TensorProto.UINT8: torch.uint8,
            TensorProto.INT8: torch.int8,
            TensorProto.BOOL: torch.bool,
        }

        # ONNX string type to ONNX TensorProto mapping
        self.onnx_str_to_tensorproto = {
            'tensor(float)': TensorProto.FLOAT,
            'tensor(float16)': TensorProto.FLOAT16,
            'tensor(bfloat16)': TensorProto.BFLOAT16,
            'tensor(double)': TensorProto.DOUBLE,
            'tensor(int64)': TensorProto.INT64,
            'tensor(int32)': TensorProto.INT32,
            'tensor(int8)': TensorProto.INT8,
            'tensor(uint8)': TensorProto.UINT8,
            'tensor(bool)': TensorProto.BOOL,
        }

        # ONNX TensorProto to NumPy mapping (needed for ORT)
        self.onnx_to_numpy_dtype = {
            TensorProto.FLOAT: np.float32,
            TensorProto.FLOAT16: np.float16,
            # Note: NumPy doesn't directly support bfloat16, use float32 as closest analog
            TensorProto.BFLOAT16: np.float32,
            TensorProto.DOUBLE: np.float64,
            TensorProto.INT32: np.int32,
            TensorProto.INT64: np.int64,
            TensorProto.UINT8: np.uint8,
            TensorProto.INT8: np.int8,
            TensorProto.BOOL: np.bool_,
        }

    def _torch_dtype_to_onnx(self, torch_dtype):
        """Convert PyTorch dtype to ONNX TensorProto dtype."""
        return self.torch_to_onnx_dtype.get(torch_dtype, TensorProto.FLOAT)

    def _onnx_dtype_to_torch(self, onnx_dtype):
        """Convert ONNX TensorProto dtype to PyTorch dtype."""
        if isinstance(onnx_dtype, str):
            # Convert string representation first
            onnx_dtype = self.onnx_str_to_tensorproto.get(onnx_dtype, TensorProto.FLOAT)
        return self.onnx_to_torch_dtype.get(onnx_dtype, torch.float32)

    def _onnx_dtype_to_numpy(self, onnx_dtype):
        """Convert ONNX TensorProto dtype to NumPy dtype."""
        if isinstance(onnx_dtype, str):
            # Convert string representation first
            onnx_dtype = self.onnx_str_to_tensorproto.get(onnx_dtype, TensorProto.FLOAT)
        #return self.onnx_to_numpy_dtype.get(onnx_dtype, np.float32)
        return onnx_dtype

    def _get_provider_options(self) -> List[Union[str, Tuple[str, Dict[str, Any]]]]:
        """Get appropriate provider options based on device and settings."""
        if self.device.lower() == "cuda":
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': self.gpu_id,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                    'gpu_mem_limit': 0,  # No limit
                })
            ]
            providers = ['NvTensorRTRTXExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        return providers

    def _setup_sessions(self):
        """Set up ONNX Runtime sessions for all models."""
        providers = self._get_provider_options()

        # Create shared session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.enable_cpu_mem_arena = False
        session_options.enable_mem_pattern = False

        if self.verbose:
            session_options.enable_profiling = True
            session_options.log_severity_level = 0  # Verbose logging

        # Create text encoder session
        if self.verbose:
            print(f"Loading text encoder from {self.text_encoder_path}")
        self.text_encoder = ort.InferenceSession(
            str(self.text_encoder_path),
            providers=providers,
            sess_options=session_options,
        )

        # Create transformer session
        if self.verbose:
            print(f"Loading transformer from {self.transformer_path}")
        self.transformer = ort.InferenceSession(
            str(self.transformer_path),
            providers=providers,
            sess_options=session_options,
        )

        # Create VAE decoder session
        if self.verbose:
            print(f"Loading VAE decoder from {self.vae_decoder_path}")
        self.vae_decoder = ort.InferenceSession(
            str(self.vae_decoder_path),
            providers=providers,
            sess_options=session_options,
        )

    def _create_io_bindings(self):
        """Create IO bindings for each session."""
        self.text_encoder_bindings = self.text_encoder.io_binding()
        self.transformer_bindings = self.transformer.io_binding()
        self.vae_decoder_bindings = self.vae_decoder.io_binding()

    def _extract_model_info(self):
        """Extract and store model input/output information."""
        # Get text encoder input/output information
        self.text_encoder_inputs = self.text_encoder.get_inputs()
        self.text_encoder_input_names = [input.name for input in self.text_encoder_inputs]
        self.text_encoder_outputs = self.text_encoder.get_outputs()
        self.text_encoder_output_names = [output.name for output in self.text_encoder_outputs]

        # Get transformer input/output information
        self.transformer_inputs = self.transformer.get_inputs()
        self.transformer_input_names = [input.name for input in self.transformer_inputs]
        self.transformer_outputs = self.transformer.get_outputs()
        self.transformer_output_names = [output.name for output in self.transformer_outputs]

        # Get VAE decoder input/output information
        self.vae_decoder_inputs = self.vae_decoder.get_inputs()
        self.vae_decoder_input_names = [input.name for input in self.vae_decoder_inputs]
        self.vae_decoder_outputs = self.vae_decoder.get_outputs()
        self.vae_decoder_output_names = [output.name for output in self.vae_decoder_outputs]

        if self.verbose:
            self._print_model_info()

    def _print_model_info(self):
        """Print information about the models."""
        print("\nText Encoder Model: ")
        print("  Inputs:")
        for inp in self.text_encoder_inputs:
            print(f"    - {inp.name}: shape={inp.shape}, type={inp.type}")
        print("  Outputs:")
        for out in self.text_encoder_outputs:
            print(f"    - {out.name}: shape={out.shape}, type={out.type}")

        print("\nTransformer Model: ")
        print("  Inputs:")
        for inp in self.transformer_inputs:
            print(f"    - {inp.name}: shape={inp.shape}, type={inp.type}")
        print("  Outputs:")
        for out in self.transformer_outputs:
            print(f"    - {out.name}: shape={out.shape}, type={out.type}")

        print("\nVAE Decoder Model: ")
        print("  Inputs:")
        for inp in self.vae_decoder_inputs:
            print(f"    - {inp.name}: shape={inp.shape}, type={inp.type}")
        print("  Outputs:")
        for out in self.vae_decoder_outputs:
            print(f"    - {out.name}: shape={out.shape}, type={out.type}")

    def _resolve_dynamic_shape(self, shape, batch_size=1, height=None, width=None):
        """Resolve dynamic dimensions in shape specifications."""
        if height is None:
            height = self.latent_height
        if width is None:
            width = self.latent_width

        # Handle string dimensions from ONNX models
        if isinstance(shape, (list, tuple)):
            resolved_shape = []
            for dim in shape:
                # If dimension is a string or contains dynamic expressions
                if dim is None or dim == -1 or (isinstance(dim, str) and 'B' in dim):
                    resolved_shape.append(batch_size)
                elif isinstance(dim, str) and 'latent_dim' in dim:
                    resolved_shape.append(self.latent_height * self.latent_width)
                elif isinstance(dim, str) and 'H' in dim:
                    resolved_shape.append(height)
                elif isinstance(dim, str) and 'W' in dim:
                    resolved_shape.append(width)
                # Handle complex expressions like 'floor(2.0*floor(2.0*floor(2.0*H)))'
                elif isinstance(dim, str) and 'floor' in dim and 'H' in dim:
                    # Approximate - multiply by 8 (3 levels of doubling)
                    resolved_shape.append(height * 8)
                elif isinstance(dim, str) and 'floor' in dim and 'W' in dim:
                    resolved_shape.append(width * 8)
                elif isinstance(dim, str) and 'floor' in dim and 'B' in dim:
                    resolved_shape.append(batch_size)
                else:
                    resolved_shape.append(dim)
            return resolved_shape
        return shape

    def tokenize_text(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize text prompts for the encoder."""
        batch_size = len(prompts)

        # Create a simple tokenizer - actual implementation would depend on your model
        if self.encoder_type == "clip":
            return self.tokenize_clip(prompts)
        else:  # t5
            return self.tokenize_t5(prompts)

    def tokenize_clip(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Simple CLIP-style tokenization."""
        batch_size = len(prompts)

        # For this example, we'll just use a simple character-based encoding
        # In a real implementation, you'd use the proper tokenizer for your model
        input_ids = torch.zeros((batch_size, self.max_length), dtype=torch.int32, device=self.device_torch)

        for i, prompt in enumerate(prompts):
            # Simple character-based tokenization (replace with proper tokenizer)
            chars = [ord(c) % 512 for c in prompt[:self.max_length]]
            chars = chars + [0] * (self.max_length - len(chars))  # Pad to max_length
            input_ids[i, :len(chars)] = torch.tensor(chars, dtype=torch.int32, device=self.device_torch)

        return {"input_ids": input_ids}

    def tokenize_t5(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Simple T5-style tokenization."""
        # Similar to CLIP but with T5 conventions
        return self.tokenize_clip(prompts)  # Simplified for this example

    def encode_text(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode text prompts to embeddings using text encoder."""
        if self.verbose:
            print(f"Tokenizing {len(prompts)} text prompt(s)...")

        # Tokenize prompts
        tokenized = self.tokenize_text(prompts)
        batch_size = tokenized["input_ids"].shape[0]

        try:
            # Clear existing bindings
            self.text_encoder_bindings.clear_binding_inputs()
            self.text_encoder_bindings.clear_binding_outputs()

            # Bind input tensors
            for i, input_info in enumerate(self.text_encoder_inputs):
                input_name = input_info.name

                if input_name in tokenized:
                    # Use provided tensor
                    tensor = tokenized[input_name]
                else:
                    # Create default tensor based on expected shape and type
                    shape = self._resolve_dynamic_shape(input_info.shape, batch_size)
                    dtype = self._onnx_dtype_to_torch(input_info.type)
                    tensor = torch.zeros(shape, dtype=dtype, device=self.device_torch)

                # Get tensor device information
                device_type = tensor.device.type
                device_id = 0 if device_type == 'cpu' else self.gpu_id

                # Bind input
                self.text_encoder_bindings.bind_input(
                    input_name,
                    device_type,
                    device_id,
                    self._onnx_dtype_to_numpy(input_info.type),
                    list(tensor.shape),
                    tensor.data_ptr()
                )

            # Bind output tensors
            output_tensors = {}
            for output_info in self.text_encoder_outputs:
                # Resolve dynamic shape
                shape = self._resolve_dynamic_shape(output_info.shape, batch_size)
                dtype = self._onnx_dtype_to_torch(output_info.type)

                # Create output tensor
                output_tensor = torch.empty(tuple(shape), dtype=dtype, device=self.device_torch)
                output_tensors[output_info.name] = output_tensor

                # Bind output
                self.text_encoder_bindings.bind_output(
                    output_info.name,
                    output_tensor.device.type,
                    0 if output_tensor.device.type == 'cpu' else self.gpu_id,
                    self._onnx_dtype_to_numpy(output_info.type),
                    list(output_tensor.shape),
                    output_tensor.data_ptr()
                )

            # Run inference with bindings
            self.text_encoder.run_with_iobinding(self.text_encoder_bindings)

            # Return output tensors
            return output_tensors

        except Exception as e:
            if self.verbose:
                print(f"IO Binding failed for text encoder: {str(e)}")
                print("Falling back to regular inference...")

            # Fall back to regular inference for compatibility
            inputs = {name: tensor.cpu().numpy() for name, tensor in tokenized.items()}
            outputs = self.text_encoder.run(None, inputs)

            # Convert outputs back to PyTorch tensors
            result = {}
            for i, name in enumerate(self.text_encoder_output_names):
                output_tensor = torch.from_numpy(outputs[i]).to(self.device_torch)
                result[name] = output_tensor

            return result

    def generate_latents(
            self,
            text_embeddings: Dict[str, torch.Tensor],
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate latent space representation using the transformer model.

        Args:
            text_embeddings: Text embeddings from the text encoder
            num_inference_steps: Number of diffusion steps
            guidance_scale: Scale for classifier-free guidance
            seed: Random seed for reproducibility

        Returns:
            Latent representation tensor
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed(seed)

        # Determine batch size from text embeddings
        batch_size = 1
        for name, tensor in text_embeddings.items():
            if tensor.dim() > 0:
                batch_size = tensor.shape[0]
                if "uncond" in name or batch_size % 2 == 0:
                    # If using classifier-free guidance, actual batch size is half
                    batch_size = batch_size // 2
                break

        # Initialize latents
        latent_dim = self.latent_height * self.latent_width
        latent_channels = 64  # From the model spec: hidden_states shape=['B', 'latent_dim', 64]

        # Initialize hidden states with random noise
        hidden_states = torch.randn(
            (batch_size, latent_dim, latent_channels),
            dtype=torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float32,
            device=self.device_torch
        )

        # Create identity matrices for positional embeddings
        #img_ids = torch.eye(3, dtype=torch.float32, device=self.device_torch).repeat(latent_dim, 1)
        #txt_ids = torch.eye(3, dtype=torch.float32, device=self.device_torch).repeat(self.max_length, 1)
        img_ids = torch.eye(latent_dim, 3, dtype=torch.float32, device=self.device_torch)
        txt_ids = torch.eye(self.max_length, 3, dtype=torch.float32, device=self.device_torch)

        # Prepare guidance - set to guidance_scale tensor
        guidance_tensor = torch.ones((batch_size,), dtype=torch.float32, device=self.device_torch) * guidance_scale

        # Prepare timestep tensors - simplified scheduler
        timestep_values = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device_torch)[:-1]

        # Prepare transformer inputs
        transformer_inputs = {}

        # Run diffusion process
        for step, t in enumerate(timestep_values):
            # Prepare timestep tensor
            timestep = torch.ones((batch_size,), device=self.device_torch, dtype=torch.bfloat16) * t

            # Prepare all inputs for this step
            transformer_inputs = {
                'hidden_states': hidden_states,
                'encoder_hidden_states': text_embeddings.get('text_embeddings', next(iter(text_embeddings.values()))),
                'pooled_projections': torch.zeros((batch_size, 768), dtype=torch.bfloat16, device=self.device_torch),
                'timestep': timestep,
                'img_ids': img_ids,
                'txt_ids': txt_ids,
                'guidance': guidance_tensor
            }

            try:
                # Clear existing bindings
                self.transformer_bindings.clear_binding_inputs()
                self.transformer_bindings.clear_binding_outputs()

                # Bind input tensors
                for input_info in self.transformer_inputs:
                    input_name = input_info.name

                    if input_name in transformer_inputs:
                        tensor = transformer_inputs[input_name]
                    else:
                        # Skip if not in our inputs
                        continue

                    # Get tensor device information
                    device_type = tensor.device.type
                    device_id = 0 if device_type == 'cpu' else self.gpu_id

                    # Bind input
                    self.transformer_bindings.bind_input(
                        input_name,
                        device_type,
                        device_id,
                        self._onnx_dtype_to_numpy(input_info.type),
                        list(tensor.shape),
                        tensor.data_ptr()
                    )
                    print("Input name : ", input_name, ", shape : ", tensor.shape)

                # Prepare output bindings - we know it has a dynamic shape
                output_name = self.transformer_output_names[0]  # 'latent'
                output_shape = [batch_size, latent_dim, latent_channels]  # Resolved shape

                # Create output tensor
                output_dtype = self._onnx_dtype_to_torch(self.transformer_outputs[0].type)
                output_tensor = torch.empty(output_shape, dtype=output_dtype, device=self.device_torch)

                # Bind output
                self.transformer_bindings.bind_output(
                    output_name,
                    output_tensor.device.type,
                    0 if output_tensor.device.type == 'cpu' else self.gpu_id,
                    self._onnx_dtype_to_numpy(self.transformer_outputs[0].type),
                    output_shape,
                    output_tensor.data_ptr()
                )
                print("\nOutput name : ", output_name, ", shape : ", output_shape)

                # Run inference with bindings
                self.transformer.run_with_iobinding(self.transformer_bindings)

                # Update hidden states with output tensor
                hidden_states = output_tensor

            except Exception as e:
                if self.verbose and step == 0:
                    print(f"IO Binding failed for transformer: {str(e)}")
                    print("Falling back to regular inference...")

                # Convert inputs to numpy arrays
                feed_dict = {name: tensor.cpu().numpy() for name, tensor in transformer_inputs.items()}

                # Run the transformer
                outputs = self.transformer.run(None, feed_dict)

                # Update hidden states
                hidden_states = torch.from_numpy(outputs[0]).to(self.device_torch)

            if self.verbose and (step == 0 or step == num_inference_steps - 1 or (step + 1) % 10 == 0):
                print(f"  Diffusion step {step + 1}/{num_inference_steps}")

        # Reshape latents to expected shape for VAE: [B, C, H, W]
        final_latents = hidden_states.reshape(batch_size, self.latent_height, self.latent_width, latent_channels)
        final_latents = final_latents.permute(0, 3, 1, 2)  # [B, C, H, W]

        return final_latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to images using VAE decoder.

        Args:
            latents: Latent representation from transformer

        Returns:
            Decoded images tensor
        """
        batch_size = latents.shape[0]

        try:
            # Clear existing bindings
            self.vae_decoder_bindings.clear_binding_inputs()
            self.vae_decoder_bindings.clear_binding_outputs()

            # Bind latent input
            input_name = self.vae_decoder_input_names[0]  # 'latent'

            # Get tensor device information
            device_type = latents.device.type
            device_id = 0 if device_type == 'cpu' else self.gpu_id

            # Bind input
            self.vae_decoder_bindings.bind_input(
                input_name,
                device_type,
                device_id,
                self._onnx_dtype_to_numpy(self.vae_decoder_inputs[0].type),
                list(latents.shape),
                latents.data_ptr()
            )

            # Compute expected output shape from VAE decoder (with floor() operations)
            # Example: floor(2.0*floor(2.0*floor(2.0*H))) = H*8
            output_height = self.latent_height * 8
            output_width = self.latent_width * 8
            output_shape = [batch_size, 3, output_height, output_width]

            # Create output tensor
            output_dtype = self._onnx_dtype_to_torch(self.vae_decoder_outputs[0].type)
            output_tensor = torch.empty(output_shape, dtype=output_dtype, device=self.device_torch)

            # Bind output
            self.vae_decoder_bindings.bind_output(
                self.vae_decoder_output_names[0],
                output_tensor.device.type,
                0 if output_tensor.device.type == 'cpu' else self.gpu_id,
                self._onnx_dtype_to_numpy(self.vae_decoder_outputs[0].type),
                output_shape,
                output_tensor.data_ptr()
            )

            # Run inference with bindings
            self.vae_decoder.run_with_iobinding(self.vae_decoder_bindings)

            # Process output to images (0-1 range)
            images = output_tensor

        except Exception as e:
            if self.verbose:
                print(f"IO Binding failed for VAE decoder: {str(e)}")
                print("Falling back to regular inference...")

            # Convert latents to numpy arrays
            latents_np = latents.cpu().numpy()

            # Run the VAE decoder
            outputs = self.vae_decoder.run(None, {self.vae_decoder_input_names[0]: latents_np})

            # Convert outputs back to PyTorch tensors
            images = torch.from_numpy(outputs[0]).to(self.device_torch)

        # Normalize pixel values to 0-1 range
        if images.min() < 0 or images.max() > 1:
            images = (images + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
            images = torch.clamp(images, 0.0, 1.0)

        return images

    def tensor_to_pil(self, images: torch.Tensor) -> List[Image.Image]:
        """
        Convert tensor image to PIL Image.

        Args:
            images: Tensor containing image data

        Returns:
            List of PIL Image objects
        """
        # Convert to uint8
        images = (images * 255).round().to(torch.uint8)

        # Move to CPU
        images = images.cpu()

        # Convert to PIL Images
        pil_images = []
        for i in range(images.shape[0]):
            # Handle both RGB and RGBA formats
            if images.shape[1] == 4:  # RGBA
                pil_image = Image.fromarray(images[i].permute(1, 2, 0).numpy(), mode="RGBA")
            else:  # RGB
                pil_image = Image.fromarray(images[i].permute(1, 2, 0).numpy(), mode="RGB")
            pil_images.append(pil_image)

        return pil_images

    def __call__(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: Union[str, List[str]] = "",
            height: int = 512,
            width: int = 512,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            seed: Optional[int] = None,
            batch_size: int = 1,
    ) -> List[Image.Image]:
        """
        Generate image from text prompt.

        Args:
            prompt: Text prompt or list of prompts for image generation
            negative_prompt: Negative text prompt or list of prompts for guidance
            height: Output image height
            width: Output image width
            num_inference_steps: Number of diffusion steps
            guidance_scale: Scale for classifier-free guidance
            seed: Random seed for reproducibility
            batch_size: Number of images to generate in a single batch

        Returns:
            List of PIL Image objects
        """
        # Update dimensions
        self.height = height
        self.width = width

        # Calculate latent dimensions (typically height/8 and width/8)
        self.latent_height = height // 32
        self.latent_width = width // 32

        # Start timing
        start_time = time.time()

        # Handle prompt input formats
        if isinstance(prompt, str):
            prompts = [prompt] * batch_size
        elif isinstance(prompt, list):
            prompts = prompt
            batch_size = len(prompts)
        else:
            raise ValueError("Prompt must be a string or list of strings")

        # Handle negative prompt
        if isinstance(negative_prompt, str):
            negative_prompts = [negative_prompt] * batch_size if negative_prompt else []
        elif isinstance(negative_prompt, list):
            negative_prompts = negative_prompt
        else:
            raise ValueError("Negative prompt must be a string or list of strings")

        # Ensure negative prompts match batch size if provided
        if negative_prompts and len(negative_prompts) != batch_size:
            if len(negative_prompts) == 1:
                negative_prompts = negative_prompts * batch_size
            else:
                raise ValueError(f"Number of negative prompts ({len(negative_prompts)}) doesn't match batch size ({batch_size})")

        if self.verbose:
            print(f"Processing batch of {batch_size} prompt(s)")

        # Encode text prompts
        if self.verbose:
            print(f"Encoding prompts...")
        text_embeddings = self.encode_text(prompts)

        # Handle negative prompts if provided
        if negative_prompts:
            if self.verbose:
                print(f"Encoding negative prompts...")
            uncond_embeddings = self.encode_text(negative_prompts)

            # Concatenate embeddings for classifier-free guidance
            for key in text_embeddings:
                if key in uncond_embeddings:
                    text_embeddings[key] = torch.cat([uncond_embeddings[key], text_embeddings[key]], dim=0)

        # Generate latents
        if self.verbose:
            print("Generating latents...")
        latents = self.generate_latents(
            text_embeddings=text_embeddings,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )

        # Decode latents to images
        if self.verbose:
            print("Decoding latents to images...")
        images = self.decode_latents(latents)

        # Convert to PIL images
        pil_images = self.tensor_to_pil(images)

        # End timing
        end_time = time.time()
        if self.verbose:
            print(f"Total generation time: {end_time - start_time:.2f} seconds")
            print(f"Average time per image: {(end_time - start_time) / batch_size:.2f} seconds")

        return pil_images

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Run Flux text-to-image ONNX models with multi-batch support and IO Bindings")
    parser.add_argument("--text-encoder", required=True, help="Path to the text encoder ONNX model (CLIP or T5)")
    parser.add_argument("--transformer", required=True, help="Path to the transformer ONNX model")
    parser.add_argument("--vae-decoder", required=True, help="Path to the VAE decoder ONNX model")
    parser.add_argument("--encoder-type", choices=["clip", "t5"], default="t5", help="Type of text encoder")
    parser.add_argument("--prompt", type=str, action="append", help="Text prompt(s) for image generation (can be specified multiple times)")
    parser.add_argument("--prompt-file", type=str, help="Path to file containing prompts (one per line)")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative text prompt for guidance")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for generated images")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--height", type=int, default=512, help="Output image height")
    parser.add_argument("--width", type=int, default=512, help="Output image width")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run inference on")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use if device is 'cuda'")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")

    args = parser.parse_args()

    # Validate model paths
    for model_path in [args.text_encoder, args.transformer, args.vae_decoder]:
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return 1

    # Collect prompts
    prompts = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, 'r') as f:
            file_prompts = [line.strip() for line in f.readlines() if line.strip()]
            prompts.extend(file_prompts)

    if not prompts:
        prompts = ["A beautiful landscape with mountains and a lake"]
        print(f"No prompts provided, using default: '{prompts[0]}'")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate number of batches and organize prompts
    total_prompts = len(prompts)
    batch_size = min(args.batch_size, total_prompts)
    num_batches = (total_prompts + batch_size - 1) // batch_size  # Ceiling division

    # Create pipeline
    pipeline = FluxTextToImagePipeline(
        text_encoder_path=args.text_encoder,
        transformer_path=args.transformer,
        vae_decoder_path=args.vae_decoder,
        device=args.device,
        gpu_id=args.gpu_id,
        fp16=args.fp16,
        verbose=args.verbose,
        encoder_type=args.encoder_type
    )

    # Process all prompts in batches
    all_images = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_prompts)
        batch_prompts = prompts[start_idx:end_idx]

        print(f"\nProcessing batch {batch_idx+1}/{num_batches} with {len(batch_prompts)} prompt(s)")

        # Generate images for this batch
        batch_images = pipeline(
            prompt=batch_prompts,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            batch_size=len(batch_prompts)
        )

        all_images.extend(batch_images)

    # Save all generated images
    for i, image in enumerate(all_images):
        prompt_text = prompts[i] if i < len(prompts) else f"prompt_{i}"
        # Create a safe filename from the prompt
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt_text[:50])
        filename = f"{i+1:04d}_{safe_prompt}.png"
        output_path = os.path.join(args.output_dir, filename)
        image.save(output_path)
        print(f"Image {i+1} saved to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
