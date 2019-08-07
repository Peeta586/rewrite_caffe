// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.

#ifndef CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "caffe/common.hpp"

using std::cout;
using std::endl;

#ifdef CAMKE_BUILD
    #include "caffe_config.h"
#else
    #define CUDA_TEST_DEVICE -1
    #define EXAMPLES_SOURCE_DIR "examples/"
    #define ABS_TEST_DATA_DIR "src/caffe/test/test_data"
#endif

int main(int argc, char** argv);

// 所有测试都继承于gtest::testing::Test
// 设备测试， CPU, GPU
namespace caffe {

    /***
     * gtest还提供了应付各种不同类型的数据时的方案，
     * 以及参数化类型的方案。我个人感觉这个方案有些复杂。
     * 首先要了解一下类型化测试，就用gtest里的例子了。
     *  这是gtest用于验证类型参数的方式， 这是类型参数化， 这样就可以用传入参数的方式
     * 验证不同类型的参数是否正确
     */
    template <typename TypeParam>
    class MultiDeviceTest : public ::testing::Test {
        public:
            typedef typename TypeParam::Dtype Dtype;
            protected:
                MultiDeviceTest() {
                    Caffe::set_mode(TypeParam::device);
                }
                virtual ~MultiDeviceTest() {}
    };

    // 这参数list用于测试类型, 测试这两个类型
    typedef ::testing::Types<float, double> TestDtypes;

    template <typename TypeParam>
    struct CPUDevice {
        typedef TypeParam Dtype;
        static const Caffe::Brew device = Caffe::CPU;
    };
    template <typename Dtype>
    class CPUDeviceTest:public MultiDeviceTest<CPUDevice<Dtype> > {
    };

    #ifdef CPU_ONLY
        typedef ::testing::Types<CPUDevice<float>,
                                CPUDevice<double> > TestDtypesAndDevices;
    #else
        template <typename TypeParam>
        struct GPUDevice {
            typedef TypeParam Dtype;
            static const Caffe::Brew device = Caffe::GPU;
        };
        // 设置一个参数类型测试的类， 然后
        /***
         * 1）定义case
         * TYPED_TEST_CASE_P(FooTest);
            接着又是一个新的宏TYPED_TEST_P类完成我们的测试案例：
            2） 生成多个用例
            TYPED_TEST_P(FooTest, DoesBlah) {
            // Inside a test, refer to TypeParam to get the type parameter.
            TypeParam n = 0;
            
            }
            TYPED_TEST_P(FooTest, HasPropertyA) {  }

            3） 注册case 和用例
             REGISTER_TYPED_TEST_CASE_P(FooTest, DoesBlah, HasPropertyA);
             用前面生命的类型测试TestDtypes
             4) 实例化case和用例
             INSTANTIATE_TYPED_TEST_CASE_P(My, FooTest, TestDtypes);
         * 
         */
        template <typename Dtype>
        class GPUDeviceTest : public MultiDeviceTest<GPUDevice<Dtype> > {
        };

        typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>,
                                GPUDevice<float>, GPUDevice<double> >
                                TestDtypesAndDevices;

    #endif // CPU_ONLY
}  // namespace caffe

#endif // CAFFE_TEST_TEST_CAFFE_MAIN_HPP_