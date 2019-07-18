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



} // namespace caffe