#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Linux Multi GPU TensorRT CI Pipeline,Linux OpenVINO CI Pipeline,orttraining-linux-gpu-ci-pipeline
#This file is only for Linux pipelines that build on ubuntu. All the docker images here are based on ubuntu.
#Please don't put CentOS or manylinux2014 related stuffs here.
set -e -o -x
id
SOURCE_ROOT=$BUILD_SOURCESDIRECTORY
SCRIPT_DIR=$BUILD_SOURCESDIRECTORY/tools/ci_build/github/linux
BUILD_DIR=$BUILD_BINARIESDIRECTORY


YOCTO_VERSION="4.19"
#Training only
INSTALL_DEPS_DISTRIBUTED_SETUP=false
ALLOW_RELEASED_ONNX_OPSET_ONLY_ENV="ALLOW_RELEASED_ONNX_OPSET_ONLY="$ALLOW_RELEASED_ONNX_OPSET_ONLY
echo "ALLOW_RELEASED_ONNX_OPSET_ONLY environment variable is set as $ALLOW_RELEASED_ONNX_OPSET_ONLY_ENV"

while getopts o:d:p:x:v:y:t:i:mue parameter_Option
do case "${parameter_Option}"
in
#yocto, ubuntu22.04
o) BUILD_OS=${OPTARG};;
#gpu, tensorrt or openvino. It is ignored when BUILD_OS is yocto.
d) BUILD_DEVICE=${OPTARG};;
#python version: 3.8 3.9 3.10 3.11 3.12 (absence means default 3.10)
p) PYTHON_VER=${OPTARG};;
# "--build_wheel --use_openblas"
x) BUILD_EXTR_PAR=${OPTARG};;
# openvino version tag: 2020.3 (OpenVINO EP 2.0 supports version starting 2020.3)
v) OPENVINO_VERSION=${OPTARG};;
# YOCTO 4.19 + ACL 19.05, YOCTO 4.14 + ACL 19.02
y) YOCTO_VERSION=${OPTARG};;
# an additional name for the resulting docker image (created with "docker tag")
# this is useful for referencing the image outside of this script
t) EXTRA_IMAGE_TAG=${OPTARG};;
# the docker image cache container registry
i) IMAGE_CACHE_CONTAINER_REGISTRY_NAME=${OPTARG};;
# install distributed setup dependencies
m) INSTALL_DEPS_DISTRIBUTED_SETUP=true;;
*) echo "Invalid option";;
esac
done

# shellcheck disable=SC2034
EXIT_CODE=1
DEFAULT_PYTHON_VER="3.10"

PYTHON_VER=${PYTHON_VER:=$DEFAULT_PYTHON_VER}
echo "bo=$BUILD_OS bd=$BUILD_DEVICE bdir=$BUILD_DIR pv=$PYTHON_VER bex=$BUILD_EXTR_PAR"

GET_DOCKER_IMAGE_CMD="${SOURCE_ROOT}/tools/ci_build/get_docker_image.py"
if [[ -n "${IMAGE_CACHE_CONTAINER_REGISTRY_NAME}" ]]; then
    GET_DOCKER_IMAGE_CMD="${GET_DOCKER_IMAGE_CMD} --container-registry ${IMAGE_CACHE_CONTAINER_REGISTRY_NAME}"
fi
DOCKER_CMD="docker"
# If BUILD_OS is ubuntu, then UBUNTU_VERSION is set to the version string after ubuntu
if [[ $BUILD_OS == ubuntu* ]]; then
    UBUNTU_VERSION=${BUILD_OS#ubuntu}
fi

NEED_BUILD_SHARED_LIB=true
cd $SCRIPT_DIR/docker
if [ $BUILD_OS = "yocto" ]; then
    IMAGE="arm-yocto-$YOCTO_VERSION"
    DOCKER_FILE=Dockerfile.ubuntu_for_arm
    # ACL 19.05 need yocto 4.19
    TOOL_CHAIN_SCRIPT=fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4.19-warrior.sh
    if [ $YOCTO_VERSION = "4.14" ]; then
        TOOL_CHAIN_SCRIPT=fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4.14-sumo.sh
    fi
    $GET_DOCKER_IMAGE_CMD --repository "onnxruntime-$IMAGE" \
        --docker-build-args="--build-arg TOOL_CHAIN=$TOOL_CHAIN_SCRIPT --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER}" \
        --dockerfile $DOCKER_FILE --context .
elif [[ $BUILD_DEVICE = "openvino"* ]]; then
        BUILD_ARGS="--build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} --build-arg OPENVINO_VERSION=${OPENVINO_VERSION} --build-arg UBUNTU_VERSION=${UBUNTU_VERSION}"
        IMAGE="$BUILD_OS-openvino"
        DOCKER_FILE=Dockerfile.ubuntu_openvino
        $GET_DOCKER_IMAGE_CMD --repository "onnxruntime-$IMAGE" \
                --docker-build-args="${BUILD_ARGS}" \
                --dockerfile $DOCKER_FILE --context .
else
  exit 1
fi

if [[ $NEED_BUILD_SHARED_LIB = true ]]; then
    BUILD_EXTR_PAR=" --build_shared_lib ${BUILD_EXTR_PAR}"
fi

if [ -v EXTRA_IMAGE_TAG ]; then
    ${DOCKER_CMD} tag "onnxruntime-$IMAGE" "${EXTRA_IMAGE_TAG}"
fi

set +e
mkdir -p ~/.onnx

if [ -z "$NIGHTLY_BUILD" ]; then
    set NIGHTLY_BUILD=0
fi

if [ $BUILD_DEVICE = "cpu" ] || [ $BUILD_DEVICE = "openvino" ] || [ $BUILD_DEVICE = "arm" ]; then
    RUNTIME=
else
    RUNTIME="--gpus all"
fi

DOCKER_RUN_PARAMETER="--volume $SOURCE_ROOT:/onnxruntime_src \
                      --volume $BUILD_DIR:/build \
                      --volume /data/models:/build/models:ro \
                      --volume /data/onnx:/data/onnx:ro \
                      --volume $HOME/.onnx:/home/onnxruntimedev/.onnx"
if [ $BUILD_DEVICE = "openvino" ] && [[ $BUILD_EXTR_PAR == *"--use_openvino GPU_FP"* ]]; then
    DOCKER_RUN_PARAMETER="$DOCKER_RUN_PARAMETER --device /dev/dri:/dev/dri"
fi
# Though this command has a yocto version argument, none of our ci build pipelines use yocto.
$DOCKER_CMD run $RUNTIME --rm $DOCKER_RUN_PARAMETER \
    -e NIGHTLY_BUILD -e SYSTEM_COLLECTIONURI \
    -e $ALLOW_RELEASED_ONNX_OPSET_ONLY_ENV \
    "onnxruntime-$IMAGE" \
    /bin/bash /onnxruntime_src/tools/ci_build/github/linux/run_build.sh \
    -d $BUILD_DEVICE -x "$BUILD_EXTR_PAR" -o $BUILD_OS -y $YOCTO_VERSION
