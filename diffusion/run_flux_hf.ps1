$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"

$env:PATH += ";C:\TensorRT-RTX-1.0.0.0\lib"

$text = "C:/FLUX.1-dev-onnx/clip.opt/model.onnx"
$transformer = "C:/FLUX.1-dev-onnx/transformer.opt/bf16/model.onnx"
$vae  = "C:/FLUX.1-dev-onnx/vae.opt/model.std.onnx"

& "C:\TensorRT-RTX-1.0.0.0\bin\trtexec" --onnx=$vae --saveEngine=vae.engine --stronglyTyped `
    --optShapes=latent:1x16x128x128
    # --maxShapes=latent:1x16x512x512 `
    # --minShapes=latent:1x16x8x8
if ($LASTEXITCODE -ne 0) { throw "trtexec failed" }
python ../onnxruntime/python/tools/tensorrt/gen_trt_engine_wrapper_onnx_model.py -p vae.engine -m vae_ctx.onnx

$vae  = "vae_ctx.onnx"

& "C:\TensorRT-RTX-1.0.0.0\bin\trtexec" --onnx=$transformer --saveEngine=transformer.engine --stronglyTyped `
    --optShapes="hidden_states:1x4096x64,encoder_hidden_states:1x512x4096,pooled_projections:1x768,timestep:1,img_ids:4096x3,txt_ids:512x3,guidance:1"
    # --optShapes="hidden_states:1x4080x64,encoder_hidden_states:1x512x4096,pooled_projections:1x768,timestep:1,img_ids:4080x3,txt_ids:512x3,guidance:1" `
    # --minShapes="hidden_states:1x3968x64,encoder_hidden_states:1x512x4096,pooled_projections:1x768,timestep:1,img_ids:3968x3,txt_ids:512x3,guidance:1" `
if ($LASTEXITCODE -ne 0) { throw "trtexec failed" }
python ../onnxruntime/python/tools/tensorrt/gen_trt_engine_wrapper_onnx_model.py -p transformer.engine -m transformer_ctx.onnx

$transformer = "transformer_ctx.onnx"

python main.py
