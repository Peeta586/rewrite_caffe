#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath> // for std::fabs, std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
// #include "caffe/util/mkl_alternate.hpp"

namespace caffe {

    // ------------------------------------------------- CPU
    inline void caffe_memset(const size_t n, const int alpha, void* x){
        memset(x, alpha, n);  // NOLINT(caffe/alt_fn)
    }
    //----------------------------------- Random
    unsigned int caffe_rng_rand();
    template <typename Dtype>
    Dtype caffe_nextafter(const Dtype b);
    // n 是获取n个随机数， 每个随机数采用平均采样的方式，r是n个数要存储的空间指针，a和b是均值采样的最小和最大区间值
    // 一下类似函数的参数理解同理
    template <typename Dtype>
    void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

    template <typename Dtype>
    void caffe_rng_gaussion(const int n, const Dtype mu, const Dtype sigma, Dtype* r);

    template <typename Dtype>
    void caffe_rng_bernoulli(const int n, const Dtype p, int *r);

    template <typename Dtype>
    void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r);

    // ------------------------------------------------------ GPU
#ifndef CPU_ONLY // GPU
    template <typename Dtype>
    void caffe_gpu_scal(const int n, const Dtype alpha, Dtype *x);
    template <typename Dtype>
    void caffe_gpu_add_scalar(const int n, const Dtype alpha, Dtype* x);

    inline void caffe_gpu_memset(const size_t n, const int alpha, void* x){
    #ifndef CPU_ONLY
        CUDA_CHECK(cudaMemset(x, alpha, n)); // NOLINT(caffe/alt_fn)
    #else
        NO_GPU;
    #endif
    }

    // x--->y
    void caffe_gpu_memcpy(const size_t n, const void* x, const void* y);

#ifndef CPU_ONLY
    template <typename Dtype>
    void caffe_gpu_scal(const int n, const Dtype alpha, Dtype *x, cudaStream_t str);
#endif
    // caffe_gpu_rng_uniform with two arguments generates integers in the range
    // [0, UINT_MAX].
    template <typename Dtype>
    void caffe_gpu_rng_uniform(const int n, unsigned int* r);

    // caffe_gpu_rng_uniform with four arguments generates floats in the range
    // (a, b] (strictly greater than a, less than or equal to b) due to the
    // specification of curandGenerateUniform.  With a = 0, b = 1, just calls
    // curandGenerateUniform; with other limits will shift and scale the outputs
    // appropriately after calling curandGenerateUniform.
    template <typename Dtype>
    void caffe_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);
    template <typename Dtype>
    void caffe_gpu_rng_gaussion(const int n, const Dtype mu, const Dtype sigma, Dtype* r);
    template <typename Dtype>
    void caffe_gpu_rng_bernoulli(const int n, const Dtype p, int* r);


#endif // !CPU_ONLY


} // namespace caffe


#endif // CAFFE_UTIL_MATH_FUNCTIONS_H_