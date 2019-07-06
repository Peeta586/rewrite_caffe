#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

#ifdef CPU_ONLY // CPU-only Caffe

#include <vector>

// Stub out(捻灭) GPU calls as unavailable 

#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

#define STUB_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*> & top) { NO_GPU;} \
template <typename Dtype> \
void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob><Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NP_GPU; } \

#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& top) { NO_GPU;} \

#else // Normal GPU+CPU Caffe
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h> // cuda driver types

#ifdef USE_CUDNN
#include <caffe/util/cudnn.hpp>
#endif

// CUDA macros

// cuda: various checks for different functions calls

#define CUDA_CHECK(condition) \
/*code block avoids redefinition of cudaError_t error */ \
do{ \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " "<<cudaGetErrorString(error); \
} while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)


#endif // CPU_ONLY


#endif //CAFFE_UTIL_DEVICE_ALTERNATE_H_