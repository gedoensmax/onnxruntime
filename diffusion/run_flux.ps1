#$env:CUDA_VISIBLE_DEVICES = "0"
#$ErrorActionPreference = "Stop"

$env:PATH += ";C:\TensorRT-RTX-1.0.0.0\lib"

# python flux.py `
# --text-encoder  C:\flux-onnx-optimum-fp16-exported\text_encoder\model.onnx `
# --transformer   C:\flux-onnx-optimum-fp16-exported\transformer\model.onnx `
# --vae-decoder   C:\flux-onnx-optimum-fp16-exported\vae_decoder\model.onnx `
# --encoder-type clip `
# --prompt "GIMME BEER!" `
# --height 1024 `
# --width 1024

python flux.py --verbose `
--text-encoder  C:\flux-onnx-optimum-fp16-exported\text_encoder\model.onnx `
--transformer   transformer_ctx.onnx `
--vae-decoder   vae_ctx.onnx `
--encoder-type clip `
--prompt "GIMME BEER!" `
--height 1024 `
--width 1024
