// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "shared_inc/cuda_call.h"
#include <core/platform/env.h>

#ifdef _WIN32
#else  // POSIX
#include <unistd.h>
#include <string.h>
#endif

namespace onnxruntime {

using namespace common;

template <typename ERRTYPE>
const char* CudaErrString(ERRTYPE) {
  ORT_NOT_IMPLEMENTED();
}

template <typename ERRTYPE>
int CudaErrCode(ERRTYPE err_code) {
    return (int)err_code;
}

#define CASE_ENUM_TO_STR(x) \
  case x:                   \
    return #x

template <>
const char* CudaErrString<cudaError_t>(cudaError_t x) {
  cudaDeviceSynchronize();
  return cudaGetErrorString(x);
}

template <>
const char* CudaErrString<cudnn_frontend::error_t>(cudnn_frontend::error_t x) {
    auto msg = x.get_message();
    char* ret_msg = new char[msg.size() + 1];
    std::strcpy(ret_msg, msg.c_str());
    return ret_msg; // TODO this is terrible and leaks !
}

template <>
int CudaErrCode(cudnn_frontend::error_t err_code) {
    return (int)err_code.get_code();
}

template <>
const char* CudaErrString<cublasStatus_t>(cublasStatus_t e) {
  cudaDeviceSynchronize();

  switch (e) {
    CASE_ENUM_TO_STR(CUBLAS_STATUS_SUCCESS);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_NOT_INITIALIZED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_ALLOC_FAILED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_INVALID_VALUE);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_ARCH_MISMATCH);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_MAPPING_ERROR);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_EXECUTION_FAILED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_INTERNAL_ERROR);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_NOT_SUPPORTED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_LICENSE_ERROR);
    default:
      return "(look for CUBLAS_STATUS_xxx in cublas_api.h)";
  }
}

template <>
const char* CudaErrString<curandStatus>(curandStatus) {
  cudaDeviceSynchronize();
  return "(see curand.h & look for curandStatus or CURAND_STATUS_xxx)";
}

template <>
const char* CudaErrString<cudnnStatus_t>(cudnnStatus_t e) {
  cudaDeviceSynchronize();
  return cudnnGetErrorString(e);
}

template <>
const char* CudaErrString<cufftResult>(cufftResult e) {
  cudaDeviceSynchronize();
  switch (e) {
    CASE_ENUM_TO_STR(CUFFT_SUCCESS);
    CASE_ENUM_TO_STR(CUFFT_ALLOC_FAILED);
    CASE_ENUM_TO_STR(CUFFT_INVALID_VALUE);
    CASE_ENUM_TO_STR(CUFFT_INTERNAL_ERROR);
    CASE_ENUM_TO_STR(CUFFT_SETUP_FAILED);
    CASE_ENUM_TO_STR(CUFFT_INVALID_SIZE);
    default:
      return "Unknown cufft error status";
  }
}

#ifdef ORT_USE_NCCL
template <>
const char* CudaErrString<ncclResult_t>(ncclResult_t e) {
  cudaDeviceSynchronize();
  return ncclGetErrorString(e);
}
#endif

template <typename ERRTYPE, bool THRW, typename SUCCTYPE = ERRTYPE>
std::conditional_t<THRW, void, Status> CudaCall(
    ERRTYPE retCode, const char* exprString, const char* libName, SUCCTYPE successCode, const char* msg, const char* file, const int line) {
  if (retCode != successCode) {
    try {
#ifdef _WIN32
      std::string hostname_str = GetEnvironmentVar("COMPUTERNAME");
      if (hostname_str.empty()) {
        hostname_str = "?";
      }
      const char* hostname = hostname_str.c_str();
#else
      char hostname[HOST_NAME_MAX];
      if (gethostname(hostname, HOST_NAME_MAX) != 0)
        strcpy(hostname, "?");
#endif
      int currentCudaDevice;
      cudaGetDevice(&currentCudaDevice);
      cudaGetLastError();  // clear last CUDA error
      static char str[1024];
      snprintf(str, 1024, "%s failure %d: %s ; GPU=%d ; hostname=%s ; file=%s ; line=%d ; expr=%s; %s",
               libName, CudaErrCode(retCode), CudaErrString(retCode), currentCudaDevice,
               hostname,
               file, line, exprString, msg);
      if constexpr (THRW) {
        // throw an exception with the error info
        ORT_THROW(str);
      } else {
        LOGS_DEFAULT(ERROR) << str;
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, str);
      }
    } catch (const std::exception& e) {  // catch, log, and rethrow since CUDA code sometimes hangs in destruction, so we'd never get to see the error
      if constexpr (THRW) {
        ORT_THROW(e.what());
      } else {
        LOGS_DEFAULT(ERROR) << e.what();
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
      }
    }
  }
  if constexpr (!THRW) {
    return Status::OK();
  }
}

template Status CudaCall<cudaError, false>(cudaError retCode, const char* exprString, const char* libName, cudaError successCode, const char* msg, const char* file, const int line);
template void CudaCall<cudaError, true>(cudaError retCode, const char* exprString, const char* libName, cudaError successCode, const char* msg, const char* file, const int line);
template Status CudaCall<cublasStatus_t, false>(cublasStatus_t retCode, const char* exprString, const char* libName, cublasStatus_t successCode, const char* msg, const char* file, const int line);
template void CudaCall<cublasStatus_t, true>(cublasStatus_t retCode, const char* exprString, const char* libName, cublasStatus_t successCode, const char* msg, const char* file, const int line);
template Status CudaCall<cudnnStatus_t, false>(cudnnStatus_t retCode, const char* exprString, const char* libName, cudnnStatus_t successCode, const char* msg, const char* file, const int line);
template void CudaCall<cudnnStatus_t, true>(cudnnStatus_t retCode, const char* exprString, const char* libName, cudnnStatus_t successCode, const char* msg, const char* file, const int line);
template Status CudaCall<curandStatus_t, false>(curandStatus_t retCode, const char* exprString, const char* libName, curandStatus_t successCode, const char* msg, const char* file, const int line);
template void CudaCall<curandStatus_t, true>(curandStatus_t retCode, const char* exprString, const char* libName, curandStatus_t successCode, const char* msg, const char* file, const int line);
template Status CudaCall<cufftResult, false>(cufftResult retCode, const char* exprString, const char* libName, cufftResult successCode, const char* msg, const char* file, const int line);
template void CudaCall<cufftResult, true>(cufftResult retCode, const char* exprString, const char* libName, cufftResult successCode, const char* msg, const char* file, const int line);
template Status CudaCall<cudnn_frontend::error_t , false, cudnn_frontend::error_code_t>(cudnn_frontend::error_t  retCode, const char* exprString, const char* libName, cudnn_frontend::error_code_t  successCode, const char* msg, const char* file, const int line);
template void CudaCall<cudnn_frontend::error_t , true, cudnn_frontend::error_code_t>(cudnn_frontend::error_t  retCode, const char* exprString, const char* libName, cudnn_frontend::error_code_t  successCode, const char* msg, const char* file, const int line);

#ifdef ORT_USE_NCCL
template Status CudaCall<ncclResult_t, false>(ncclResult_t retCode, const char* exprString, const char* libName, ncclResult_t successCode, const char* msg, const char* file, const int line);
template void CudaCall<ncclResult_t, true>(ncclResult_t retCode, const char* exprString, const char* libName, ncclResult_t successCode, const char* msg, const char* file, const int line);
#endif
}  // namespace onnxruntime
