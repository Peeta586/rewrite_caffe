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

/*code block avoids redefinition of cudaError_t error */ 
#define CUDA_CHECK(condition) \
  do{ \
      cudaError_t error = condition; \
      CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

//CHECK_EQ 如果等于就不打印，如果不等于就打印后面的string，这个跟python 的assert使用差不多
#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)

//CUDA: grid stride looping; 以一个grid为单位的跨度计算，也就是从i开始，下一个grid相同位置的线程是对应的实际ID
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

/**
 * 在CUDA编程中，也定义了错误类型 cudaError_t ，只不过这个错误值一般是runtime API的返回值。当且仅当这个API返回值为cudaSuccess时，才说明这个API调用正确。
typedef enum cudaError cudaError_t； 
关于cudaError_t类型变量的值很多，有70+种，具体的大家可以去查看nvidia的CUDA Runtime API手册。 
那么问题来了，当发生了API调用错误，我们如何知道错误信息呢？
主要通过如下两个函数：
__host__ ​ __device__ ​cudaError_t cudaGetLastError ( void )
__host__ ​ __device__ ​cudaError_t cudaPeekAtLastError ( void )

这两个函数获取最后的错误信息，返回值是一个cudaError_t类型。不同的是，cudaGetLastError()函数将重新将系统的全局错误信息变量重置为cudaSuccess，而cudaPeekAtLastError() 函数不会有这样的操作。
 * 
 * 也就是说用于返回程序最后的错误信息码(全局错误信息变量)， 以判断是否有程序运行出错， 而且这种方式不会使这个全局错误码重置为cudaSuccess.
 */
// CUDA: check for error after kernel execution and exit loudly if there is one
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace caffe {

  // cuda library error reporting
  const char* cublasGetErrorString(cublasStatus_t error);
  const char* curandGetErrorString(curandStatus_t error);

  // cuda: use 512 threads per block
  const int CAFFE_CUDA_NUM_THREADS = 512;

  // CUDA: number of blocks for threads
  inline int CAFFE_GET_BLOCKS(const int N){
    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
  }

} // namespace caffe


#endif // CPU_ONLY


#endif //CAFFE_UTIL_DEVICE_ALTERNATE_H_