#!/bin/bash
set -e

isv_root="../out"
share_root="/mnt/share"

#python -m run_ort -i \
#  "$share_root/onnx/yolo/yolov4_fp16.onnx" \
#  "$share_root/onnx/convnets/resnet101-v2-7_bs8.onnx" \
#  --exe "./cmake/cmake-build-relwithdebinfo-remote-host-vm/onnxruntime_perf_test" \
#  -t 5 --cuda --cuda_nhwc -o results.json # --dml # --trt
python -m run_ort -i \
  "$isv_root/isv_models/adobe_cmgan_tiny_gpu.onnx" \
  "$isv_root/isv_models/adobe_wire_seg_global.onnx" \
  "$isv_root/isv_models/Topaz_VAI_Apollo_576x384_1x_v8_fp16.onnx" \
  "$isv_root/isv_models/Topaz_VAI_Proteus_480x384_2x_v4_fp16.onnx" \
  "$isv_root/isv_models/adobe_lama_new_fp32_512x512.onnx" \
  "$isv_root/isv_models/dxo_DeepRaw2RGB_Bayer_Main_v4_fix_16f.onnx" \
  "$isv_root/isv_models/Topaz_VAI_GaiaHighQuality_576x384_2x_v5_fp16.onnx" \
  "$share_root/onnx/yolo/yolov4_fp16.onnx" \
  "$share_root/onnx/convnets/resnet101-v2-7_bs8.onnx" \
  "$share_root/onnx/sd/sd-1.5/trt_demo_export/unet.opt/sd_unet_fixed_simp.onnx" \
  "$share_root/onnx/sd/sdxl-turbo/trt_ep/unet.ort_trt/sd_xl_unet_fixed_simp.onnx" \
  --exe "./cmake/cmake-build-relwithdebinfo-remote-host-vm/onnxruntime_perf_test" \
  -t 5 --cuda --cuda_nhwc -o results.json # --dml # --trt
