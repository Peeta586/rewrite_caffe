#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits> //定义最大最小值，无穷什么的
#include <cmath>

//That's a comment. In this case, it's a comment designed to be read by a static analysis tool to tell it to shut up about this line.
#include <fstream> // NOLINT(readability/streams)
#include <iostream> // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream> //ostringstream:用于输出操作,istringstream:用于输入操作,stringstream:用于输入输出操作
#include <string>
#include <utility>
#include <vector>

#include "caffe/util/device_alternate.hpp"

// convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFLAGS_H_ to detect if it is version
// 2.1. If yes, we will add a temporary solution to redirect the namespace.
// TODO(LSHM): Once gflags solves the problem in a more elegant way, let's
// remove the following hack.

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google
#endif // GFLAGS_GFLAGS_H_

// disable the copy and assigment operator for a class
#define DISABLE_COPY_AND_ASSIGN(classname) \
private: \
    classname(const classname&);\
    classname & operator=(const classname&)

// instantiate a class with float and double specifications
// 限定了该类只能用float,double类型
#define INSTANTIATE_CLASS(classname) \
    char gInstantiationGuard##classname;\
    template class classname<float>; \
    template class classname<double>

// 实力化layer 类的前传和后传类别
#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
    template void classname<float>::Forward_gpu( \
        const std::vector<Blob<float>*>& bottom, \
        const std::vector<Blob<float>*>& top); \
    template void classname<double>::Forward_gpu( \
        const std::vector<Blob<double>*>& bottom,\
        const std::vector<Blob<double>*>& top) 

#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
    template void classname<float>::Backward_gpu(\
        const std::vector<Blob<float>*>& top, \
        const std::vector<bool>& propagate_down,\
        const std::vector<Blob<float>*>& bottom);\
    template void classname<double>::Backward_gpu(\
        const std::vector<Blob<double>*>& top, \
        const std::vector<bool>& propagate_down,\
        const std::vector<Blob<double>*>& bottom)

#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
    INSTANTIATE_LAYER_GPU_FORWARD(classname); \
    INSTANTIATE_LAYER_GPU_BACKWARD(classname)

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not implemented Yet"

// see PR #1236
namespace cv{ class Mat;}

namespace caffe{
    // we will use the boost shared_prt instead of the new C++11 one mainly
    // because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;

// common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
//vc++的编译器不支持这两个函数
using std::isnan;
using std::isinf;

using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;

// a global initialization function that you should call in your main function.
//currently it initializes google flags and google logging.
void GlobalInit(int* pargc, char*** pargv);

// a singleton class to hold common caffe stuff, such as the handler that caffe is going to use for cublas curand etc
/**
 * 设计模式中，单例模式相对来说算是比较简单的一种构建模式。适用的场景在于：对于定义的一个类，在整个应用程序执行期间只有唯一的一个实例对象。
 * 如Android中常见的Application对象。
通过单例模式，自行实例化并向这个系统提供这个单一实例的访问方法。
根据此单一实例产生的时机不同（当然，都是指第一次，也是唯一一次产生此单一实例时），可以将其分为懒汉式、饿汉式和登记式
//私有构造函数只能在函数内部调用，外部不能实例化，所以私有构造函数可以防止该类在外部被实例化
// 登记式单例模式，一般是通过一个专门的类对各单例模式的此单一实例进行管理和维护。
// 该类利用利用内部类RNG对随机产生器进行管理，然后该类本身有一些成员函数进行随机生成器的使用。
---------------------------------------------------------------------------------------------------------
类中定义一个类，虽然可以，但是建议尽量不要用，可读性不好。类都应当对是可以独立存在的抽象
这种方法主要是用于封装，要访问 RNG类，可以通过使用Caffe::RNG来用
这种方法可以 在类中封装结构体。但是在c++中结构体和类其实是一个东西，唯一区别是类的成员默认是private，而结构体是public
但是由于一直以来的习惯，结构体一般只是作为存储数据用的数据结构，没有具体行为，这点也可以看做和类的区别， 因为类是有行为的（成员函数）
结构体定义在类的内部和外部都是可以的，但是为了程序的可读性， 一般定义在类的外部。

这里有点绕，特别是Caffe类里面有个RNG，RNG这个类里面还有个Generator类
在RNG里面会用到Caffe里面的Get()函数来获取一个新的Caffe类的实例（如果不存在的话）。
然后RNG里面用到了Generator。Generator是实际产生随机数的
 */

class Caffe {
    public:
        ~Caffe();

        // 获取随机产生器的，常驻内存，
        // 这是singleston的主要函数，这个函数产生该类的实例，且封闭了Caffe的构造，拷贝，赋值函数，因此只有一个实例
        static Caffe& Get();

        enum Brew {CPU, GPU};

    // 这个类隐藏了 boost和CUDA随机数的生成，
    // 为什么里面包了一个Generator，我觉得应该是为了可读性，这样RNG表示随机数生成，而Generator表示里面的生成器，具有很好可读性
    // 要不直接用caffe:rng_t会很没有可读性
    // this random number generator facade hides boost and cuda rng
    // implementation from one another (for cross-platform compatibility)
    //explicit关键字的作用就是防止类构造函数的隐式自动转换. （也就是如果构造函数参数个数类型不同个数相同，那么可能会隐式自动转换） explicit防止这种转换
    // 这个内部类像用于存储随机产生器的一个盒子，其功能是制造一个产生器
    class RNG {
        public:
            // 构造
            RNG();
            explicit RNG(unsigned int seed);
            // 拷贝函数
            explicit RNG(const RNG&);
            // 赋值函数
            RNG& operator=(const RNG&);
            void* generator();
        private:
            class Generator;
            shared_ptr<Generator> generator_;
    };

    // Getters for boost rng curand and cublas handles
    inline static RNG& rng_stream() {
        if (!Get().random_generator_) {
            Get().random_generator_.reset(new RNG());
        }
        return *(Get().random_generator_);
    }

// 类设置静态函数，就是让其与了独立出来，不依附于对象，而是只与类有关，所有对象都可以使用该类，这与外部普通函数不一样，因为这个跟类绑定着呢
// 但效果跟外部函数产不多，而且对于类内，也只能访问静态的变量
/*
非静态成员引用必须与特定对象相对
因为静态成员函数属于整个类，在类实例化对象之前就已经分配空间了，
而类的非静态成员必须在类实例化对象后才有内存空间，所以这个调用就会出错，就好比没有声明一个变量却提前使用它一样。

××××××× 下面静态成员函数使用非静态成员变量时， 因为用Get()已经指定了该类的实例，所以是不会编译出错的
 */
#ifndef CPU_ONLY
// 当CPU_ONLY没有设置时，需要获取curand_generator_
    inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_;}
    inline static curandGenerator_t curand_generator(){
        return Get().curand_generator_;
    }
#endif
    // returns the mode: running on CPU or GPU
    inline static Brew mode() { return Get().mode_; }
    // The setters for the variables
    // Sets the mode. It is recommended that you don't change the mode halfway
    // into the program since that may cause allocation of pinned memory being
    // freed in a non-pinned way, which may cause problems - I haven't verified
    // it personally but better to note it here in the header file.
    inline static void set_mode(Brew mode) { Get().mode_ = mode; }
    // sets the random seed of both boost and curand
    static void set_random_seed(const unsigned int seed);
    // sets the device. since we have cublas and curand stuff, set device also 
    // requires us to reset those values
    static void SetDevice(const int device_id);
    // print the current GPU status
    static void DeviceQuery();
    // check if specified device is available
    static bool CheckDevice(const int device_id);
    // search from start_id to the highest possible device ordinal
    // return the ordinal（序数的） of the first available device
    static int FindDevice(const int start_id = 0);
    // parallel training
    // 所有参数设置都使用Get(), 这样保证了只对这一个实例进行操作设置
    inline static int solver_count() { return Get().solver_count_; }
    inline static void set_solver_count(int val) { Get().solver_count_ = val; }
    inline static int solver_rank() { return Get().solver_rank_; }
    inline static void set_solver_rank(int val) { Get().solver_rank_ = val; }
    inline static bool multiprocess() { return Get().multiprocess_; }
    inline static void set_multiprocess(bool val) { Get().multiprocess_ = val; }
    inline static bool root_solver() { return Get().solver_rank_ == 0; }

    protected:
#ifndef CPU_ONLY
        // 这个handle被很多地方使用，只要用到cublas加速的地方都用到这个唯一定义的cublas_handle_
        cublasHandle_t cublas_handle_;
        curandGenerator_t curand_generator_;
#endif
        shared_ptr<RNG> random_generator_;

        Brew mode_;

        // parallel training
        int solver_count_;
        int solver_rank_;
        bool multiprocess_;

    private:
        Caffe();
        DISABLE_COPY_AND_ASSIGN(Caffe);

}; //  类和结构体都要加;, 而函数不加;

} // namespace caffe

#endif // CAFFE_COMMON_HPP_