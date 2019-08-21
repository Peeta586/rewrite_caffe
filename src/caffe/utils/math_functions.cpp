#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

    unsigned int caffe_rng_rand(){
        // boost::mt19937()
        return (*caffe_rng())();
    }
    /*
    返回第一个参数和第二个参数之间与第一个参数相邻的浮点数。如果两个参数比较起来相等，则返回第二个参数
    这么做的作用是，因为uniform的去区间是左开右闭的(a,,b], 因此，在定义分布时，
    boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
    因为函数数双开的，所以需要取第一个大于b的浮点数作为边界，这样会取到b    
     */
    template <typename Dtype>
    Dtype caffe_nextafter(const Dtype b){
        return boost::math::nextafter<Dtype>(b, std::numeric_limits<Dtype>::max());
    }
    template
    float caffe_nextafter(const float b);

    template 
    double caffe_nextafter(const double b);

    template <typename Dtype>
    void caffe_rng_uniform(const int n, const Dtype a, Dtype b, Dtype* r){
        CHECK_GE(n, 0);
        /**
         * // CHECK dies with a fatal error if condition is not true.  It is *not*
        // controlled by NDEBUG, so the check will be executed regardless of
        // compilation mode.  Therefore, it is safe to do things like:
        //    CHECK(fp->Write(x) == 4)
        判断指针是否有效
         */
        CHECK(r);
        CHECK_LE(a, b);
        // 定义a,b区间平均分布配置
        boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
        // 随机种子设置的随机产生器，以及根据参数设置的平均分布配置， 从而生成在该分布下的随机产生器。
        boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> > 
            variate_generator(caffe_rng(), random_distribution);
        for (int i = 0; i < n; ++i){
            r[i] = variate_generator();
        }
    }

    template
    void caffe_rng_uniform<float>(const int n, const float a, float b, float* r);
    template
    void caffe_rng_uniform<double>(const int n, const double a, double b, double* r);

    template <typename Dtype>
    void caffe_rng_gaussion(const int n, const Dtype mu, const Dtype sigma, Dtype* r){
        CHECK_GE(n, 0);
        CHECK(r);
        CHECK_GT(sigma, 0);
        boost::normal_distribution<Dtype> random_distribution(mu, sigma);
        boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
            variate_generator(caffe_rng(), random_distribution);
        
        for (int i = 0; i < n; ++i){
            r[i] = variate_generator();
        }
    }

    template 
    void caffe_rng_gaussion<float>(const int n, const float mu, 
                                    const float sigma, float* r);

    template 
    void caffe_rng_gaussion<double>(const int n, const double mu, 
                                    const double sigma, double* r);

    template <typename Dtype>
    void caffe_rng_bernoulli(const int n, const Dtype p, int* r){
        CHECK_GE(n, 0);
        CHECK(r);
        CHECK_GE(p, 0);
        CHECK_LE(p, 1);
        boost::bernoulli_distribution<Dtype> random_distribution(p);
        boost::variate_generator<caffe::rng_t*,boost::bernoulli_distribution<Dtype> >
            variate_generator(caffe_rng(), random_distribution);
        for (int i =0; i < n; ++i){
            r[i] = variate_generator();
        }
    }

    template
    void caffe_rng_bernoulli<float>(const int n, const float p, int*r);
    template
    void caffe_rng_bernoulli<double>(const int n, const double p, int*r);

    template <typename Dtype>
    void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r){
        CHECK_GE(n, 0);
        CHECK(r);
        CHECK_GE(p, 0);
        CHECK_LE(p, 1);
        boost::bernoulli_distribution<Dtype> random_distribution(p);
        boost::variate_generator<caffe::rng_t*,boost::bernoulli_distribution<Dtype> >
            variate_generator(caffe_rng(), random_distribution);
        for (int i =0; i < n; ++i){
            r[i] = static_cast<unsigned int>(variate_generator());
        }
    }

    template
    void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int*r);
    template
    void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int*r);


    // z = a*x+y
    template <>
    void caffe_axpy<float>(const int n, const float alpha, const float*x, float* y){
        cblas_saxpy(n, alpha, x, 1, y, 1);
    }
    template <>
    void caffe_axpy<double>(const int n, const double alpha, const double*x, double* y){
        cblas_daxpy(n, alpha, x, 1, y, 1);
    }
    // y = abs(x)
    template <>
    float caffe_cpu_asum<float>(const int n, const float* x){
        return cblas_sasum(n, x, 1);
    }

    template <>
    double caffe_cpu_asum<double>(const int n, const double* x){
        return cblas_dasum(n, x, 1);
    }

    // y = x * x
    template <>
    float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx, 
        const float* y, const int incy){
        return cblas_sdot(n, x, incx, y, incy);
    }

    template <>
    double caffe_cpu_strided_dot<double>(const int n, const double* x, const int incx, 
        const double* y, const int incy){
        // x * y  点乘
        return cblas_ddot(n, x, incx, y, incy);
    }

    template <typename Dtype>
    Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y){
        return caffe_cpu_strided_dot(n,x,1,y,1);
    }
    // 注意下面是实例化，所以不能加<> 否则会报出
    /**
     * undefined reference to `double caffe::caffe_cpu_dot<double>(int, double const*, double const*)'
       undefined reference to `float caffe::caffe_cpu_dot<float>(int, float const*, float const*)'
    错误
     */
    template
    float caffe_cpu_dot<float>(const int n, const float* x, const float* y);
    template
    double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

    // y = alpha * x
    template <>
    void caffe_scal<float>(const int n, const float alpha, float* x){
        // incX 表示stride， 间隔，1 表示x的所有元素（每个元素间隔一个）
        // cblas_sscal(const int N, const float alpha, float *X, const int incX);
        cblas_sscal(n, alpha, x, 1);
    }
    template <>
    void caffe_scal<double>(const int n, const double alpha, double* x){
        cblas_dscal(n, alpha, x, 1);
    }

    template <>
    void caffe_cpu_scale<float>(const int n, const float alpha, const float* x, float* y){
        cblas_scopy(n, x, 1, y, 1);
        cblas_sscal(n, alpha, y, 1);
    }
    template <>
    void caffe_cpu_scale<double>(const int n, const double alpha, const double* x, double* y){
        cblas_dcopy(n, x, 1, y, 1);
        cblas_dscal(n, alpha, y, 1);
    }

    template <typename Dtype> 
    void caffe_copy(const int n, const Dtype* x, Dtype* y){
        if(x != y){
            if(Caffe::mode() == Caffe::GPU){
            #ifndef CPU_ONLY
                // NOLINT_NEXT_LINE(caffe/alt_fn)
                CUDA_CHECK(cudaMemcpy(y,x, sizeof(Dtype)*n, cudaMemcpyDefault));
            #else  
                NO_GPU;
            #endif
            } else {
                memcpy(y,x, sizeof(Dtype)* n);
            }
        }
    }

    // 这是实例化，不是特例化， 与template<> 区别是实例化只能使用这四个类别
    template void caffe_copy<int>(const int n, const int*x, int*y);
    template void caffe_copy<unsigned int>(const int n, const unsigned int* x, unsigned int* y);
    template void caffe_copy<float>(const int N, const float* X, float* Y);
    template void caffe_copy<double>(const int N, const double* X, double* Y);






} // namespace caffe