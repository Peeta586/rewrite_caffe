#include "gtest/gtest.h"

#include "caffe/common.hpp"
// #include "caffe/syncedmen.hpp"
// #include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
    class CommonTest: public ::testing::Test{};

    #ifndef CPU_ONLY
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
        EXPECT_EQ(Caffe::mode(), CAFFE::CPU);
        Caffe::set_mode(Caffe::GPU);
        EXPECT_EQ(Caffe::mode(), Caffe::GPU);
    }
} //namespace caffe

