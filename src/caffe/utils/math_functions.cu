/**
warning: #warning "math_functions.h is an internal header file and must not be used directly.  This file will be 
removed in a future CUDA release.  Please use cuda_runtime_api.h or cuda_runtime.h instead." [-Wcpp]
*/
#include <cuda_runtime.h>
// #include <math_functions.h>  // CUDA's not caffe's , for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h> // thrust::plus
#include <thrust/reduce.h>

#include <cmath>
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    /**
        定义kernel    
    */
    template <typename Dtype>
    __global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype * y){
        // 每一个kernel, 处理blockDim.x * gridDim.x跨度内的数据序列
        CUDA_KERNEL_LOOP(index, n){
            y[index] += alpha;
        }
    }

    template<>
    void caffe_gpu_add_scalar(const int n, const float alpha, float *y){
        // NOLINT_NEXT_LINE(whitespace/operators)
        add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, alpha, y);
    }

    template<>
    void caffe_gpu_add_scalar(const int n, const double alpha, double *y){
        // NOLINT_NEXT_LINE(whitespace/operators)
        add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, alpha, y);
    }

    void caffe_gpu_memcpy(const size_t n, const void* x, void*y){
        if (x != y){
            // x ---> y
            CUDA_CHECK(cudaMemcpy(y, x, n,cudaMemcpyDefault)); // NOLINT(caffe/alt_fn)
        }
    }

    template <>
    void caffe_gpu_scal<float>(const int n, const float alpha, float *x){
        CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, x, 1));
    }
    
    template <>
    void caffe_gpu_scal<double>(const int n, const double alpha, double *x){
        CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, x, 1));
    }

    template <>
    void caffe_gpu_scal<float>(const int n, const float alpha, float* x, cudaStream_t str){
        cudaStream_t initial_stream;
        CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
        CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
        CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, x, 1));
        CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
    }

    template <>
    void caffe_gpu_scal<double>(const int n, const double alpha, double* x, cudaStream_t str){
        cudaStream_t initial_stream;
        CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
        CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
        CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, x, 1));
        CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
    }

    void caffe_gpu_rng_uniform(const int n, unsigned int*r){
        CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
    }
    template <>
    void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b, float* r){
        CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
        const float range = b -a;
        // 将[0， MAX_FLAOT]范围内的均值采样，映射到(0, range)内
        if (range != static_cast<float>(1)){
            caffe_gpu_scal(n, range, r);
        }
        // 如果a不是0， 则所有随机数都+a, 这样就映射到（a,b）范围内的均值分布采样
        if (a != static_cast<float>(0)){
            caffe_gpu_add_scalar(n, a, r);
        }
    }

    template <>
    void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b, double* r){
        CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
        const double range = b -a;
        // 将[0， MAX_FLAOT]范围内的均值采样，映射到(0, range)内
        if (range != static_cast<double>(1)){
            caffe_gpu_scal(n, range, r);
        }
        // 如果a不是0， 则所有随机数都+a, 这样就映射到（a,b）范围内的均值分布采样
        if (a != static_cast<double>(0)){
            caffe_gpu_add_scalar(n, a, r);
        }
    }

    template <>
    void caffe_gpu_rng_gaussion(const int n, const float mu, float sigma, float* r){
        CURAND_CHECK(curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
    }

    template <>
    void caffe_gpu_rng_gaussion(const int n, const double mu, double sigma, double* r){
        CURAND_CHECK(curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
    }

} // namespace caffe

