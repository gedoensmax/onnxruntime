set -e
export LD_LIBRARY_PATH=/mnt/hdd/TRT/trtjit/TensorRT-10.12.0.6/lib

# polygraphy inspect --ignore-external-data model.onnx

#/mnt/hdd/TRT/trtexec.sh /mnt/hdd/TRT/trtjit/TensorRT-10.12.0.6/ --onnx=/mnt/hdd/FLUX.1-schnell-onnx/vae.opt/model.onnx \
#  --saveEngine=vae.engine \
#  --optShapes=latent:1x16x64x64 \
#  --maxShapes=latent:1x16x512x512 \
#  --minShapes=latent:1x16x8x8
#
#python ../onnxruntime/python/tools/tensorrt/gen_trt_engine_wrapper_onnx_model.py -p vae.engine -m vae_ctx.onnx

TYPE=fp8
#    {hidden_states [dtype=bfloat16, shape=('batch_size', 'latent_dim', 64)],
#     encoder_hidden_states [dtype=bfloat16, shape=('batch_size', 256, 4096)],
#     pooled_projections [dtype=bfloat16, shape=('batch_size', 768)],
#     timestep [dtype=bfloat16, shape=('batch_size',)],
#     img_ids [dtype=float32, shape=('latent_dim', 3)],
#     txt_ids [dtype=float32, shape=(256, 3)]}
/mnt/hdd/TRT/trtexec.sh /mnt/hdd/TRT/trtjit/TensorRT-10.12.0.6/ --onnx=/mnt/hdd/FLUX.1-schnell-onnx/transformer.opt/${TYPE}/model.onnx \
  --saveEngine=transformer_schnell_${TYPE}.engine \
  --optShapes="hidden_states:1x4096x64,encoder_hidden_states:1x256x4096,pooled_projections:1x768,timestep:1,img_ids:4096x3,txt_ids:256x3" \

python ../onnxruntime/python/tools/tensorrt/gen_trt_engine_wrapper_onnx_model.py \
  -p transformer_schnell_${TYPE}.engine -m transformer_schnell_${TYPE}_ctx.onnx

TYPE=fp4
/mnt/hdd/TRT/trtexec.sh /mnt/hdd/TRT/trtjit/TensorRT-10.12.0.6/ --onnx=/mnt/hdd/FLUX.1-dev-onnx/transformer.opt/${TYPE}/model.onnx \
  --saveEngine=transformer_dev_${TYPE}.engine \
  --optShapes="hidden_states:1x4096x64,encoder_hidden_states:1x512x4096,pooled_projections:1x768,timestep:1,img_ids:4096x3,txt_ids:512x3,guidance:1"

python onnxruntime/python/tools/tensorrt/gen_trt_engine_wrapper_onnx_model.py -p transformer_dev_${TYPE}.engine -m transformer_dev_${TYPE}_ctx.onnx