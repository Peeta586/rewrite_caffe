#include <vector>

#include "gtest/gtest.h"


#include "caffe/blob.hpp"  // this conflicts with filler.hpp, 可以隐去，不过由于blob添加了ifndef所以可以不隐去
#include "caffe/common.hpp"
#include "caffe/filler.hpp"  // this contain blob.hpp

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {


    template <typename Dtype>
    class BlobSimpleTest : public ::testing::Test {
    protected:
    BlobSimpleTest()
        : blob_(new Blob<Dtype>()),
            blob_preshaped_(new Blob<Dtype>(2, 3, 4, 5)) {}
    virtual ~BlobSimpleTest() { delete blob_; delete blob_preshaped_; }
    Blob<Dtype>* const blob_;
    Blob<Dtype>* const blob_preshaped_;
    };

} // namespace caffe