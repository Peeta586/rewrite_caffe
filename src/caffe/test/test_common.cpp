#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
    class CommonTest: public ::testing::Test{
        public:
            // Sets up the stuff shared by all tests in this test case.
            //
            // Google Test will call Foo::SetUpTestCase() before running the first
            // test in test case Foo.  Hence a sub-class can define its own
            // SetUpTestCase() method to shadow the one defined in the super
            // class.
            static void SetUpTestCase() {
                std::cout<<"We begin to test Common file--------------------------------"<<std::endl;
            }

            // Tears down the stuff shared by all tests in this test case.
            //
            // Google Test will call Foo::TearDownTestCase() after running the last
            // test in test case Foo.  Hence a sub-class can define its own
            // TearDownTestCase() method to shadow the one defined in the super
            // class.
            static void TearDownTestCase() {
                std::cout<<"Finish to test Common file--------------------------------"<<std::endl;
            }

    };

    #ifndef CPU_ONLY
        /**
         * 注意， Caffe 在实现函数定义的过程中，调用了Get， 那么，它已经实现了实例化，
         * 因此直接用Caffe::cublas_handle等成员变量即是实例化的
         */
        TEST_F(CommonTest, TestCublasHandlerGPU) {
            int cuda_device_id;
            CUDA_CHECK(cudaGetDevice(&cuda_device_id));
            EXPECT_TRUE(Caffe::cublas_handle());
        }
    #endif
    
    // Defines a test that uses a test fixture.
    //
    // The first parameter is the name of the test fixture class, which
    // also doubles as the test case name.  The second parameter is the
    // name of the test within the test case.
    //
    // A test fixture class must be declared earlier.  The user should put
    // his test code between braces after using this macro.  Example:
    //
    //   class FooTest : public testing::Test {
    //    protected:
    //     virtual void SetUp() { b_.AddElement(3); }
    //
    //     Foo a_;
    //     Foo b_;
    //   };
    //
    //   TEST_F(FooTest, InitializesCorrectly) {
    //     EXPECT_TRUE(a_.StatusIsOK());
    //   }
    //
    //   TEST_F(FooTest, ReturnsElementCountCorrectly) {
    //     EXPECT_EQ(0, a_.size());
    //     EXPECT_EQ(1, b_.size());
    //   }
    TEST_F(CommonTest, TestBrewMode) {
        Caffe::set_mode(Caffe::CPU);
        EXPECT_EQ(Caffe::mode(), Caffe::CPU);
        Caffe::set_mode(Caffe::GPU);
        EXPECT_EQ(Caffe::mode(), Caffe::GPU);
    }

    TEST_F(CommonTest, TestRandSeedCPU){
        SyncedMemory data_a(10 * sizeof(int));
        SyncedMemory data_b(10 * sizeof(int));

        Caffe::set_random_seed(1701);
        // 初始化
        caffe_rng_bernoulli(10, 0.5, static_cast<int*>(data_a.mutable_cpu_data()));

        Caffe::set_random_seed(1701);
        caffe_rng_bernoulli(10, 0.5, static_cast<int*>(data_b.mutable_cpu_data()));
        
        for(int i =0; i< 10; ++i){
            EXPECT_EQ(static_cast<const int*>(data_a.cpu_data())[i],
                static_cast<const int*>(data_b.cpu_data())[i]);
        }

    }

#ifndef CPU_ONLY
    TEST_F(CommonTest, TestRandSeedGPU){
        SyncedMemory data_a(10 * sizeof(unsigned int));
        SyncedMemory data_b(10 * sizeof(unsigned int));

        Caffe::set_random_seed(1701);
        // 产生10个随机数， 由curand_generator产生
        CURAND_CHECK(curandGenerate(Caffe::curand_generator(), 
            static_cast<unsigned int*>(data_a.mutable_gpu_data()), 10));
        
        Caffe::set_random_seed(1701);
        CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
            static_cast<unsigned int*>(data_b.mutable_gpu_data()), 10));
        
        for(int i = 0; i < 10; i++){
            EXPECT_EQ(((const unsigned int*)(data_a.cpu_data()))[i], 
            ((const unsigned int*)(data_b.cpu_data()))[i]);
        }
    }
#endif



    
} //namespace caffe

