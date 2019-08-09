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

    // 对测试类进一步的设置， 设置他要测试的类型
    TYPED_TEST_CASE(BlobSimpleTest, TestDtypes); // BlobSimpleTest

    // 继承Testing, 后面的是测试用例的名字
    TYPED_TEST(BlobSimpleTest, TestInitialization){
        EXPECT_TRUE(this->blob_);  // 用这个用例类的内部成员变量
        EXPECT_TRUE(this->blob_preshaped_);
        EXPECT_EQ(this->blob_preshaped_->num(), 2);
        EXPECT_EQ(this->blob_preshaped_->channels(), 3);
        EXPECT_EQ(this->blob_preshaped_->height(), 4);
        EXPECT_EQ(this->blob_preshaped_->width(), 5);
        EXPECT_EQ(this->blob_preshaped_->count(), 120);
        EXPECT_EQ(this->blob_->num_axes(), 0);
        EXPECT_EQ(this->blob_->count(), 0);
    }

    TYPED_TEST(BlobSimpleTest, TestPointersCPUGPU){
        EXPECT_TRUE(this->blob_preshaped_->gpu_data());
        EXPECT_TRUE(this->blob_preshaped_->cpu_data());
        EXPECT_TRUE(this->blob_preshaped_->mutable_gpu_data());
        EXPECT_TRUE(this->blob_preshaped_->mutable_cpu_data());
    }

    TYPED_TEST(BlobSimpleTest, TestReshape){
        this->blob_->Reshape(2, 3, 4, 5);
        EXPECT_EQ(this->blob_->num(), 2);
        EXPECT_EQ(this->blob_->channels(), 3);
        EXPECT_EQ(this->blob_->height(), 4);
        EXPECT_EQ(this->blob_->width(), 5);
        EXPECT_EQ(this->blob_->count(), 120);
    }

    TYPED_TEST(BlobSimpleTest, TestReshapeZero){
        vector<int> shape(2);
        shape[0] = 0;  // Reshape 中 count_ *= shape[i];， count为0
        shape[1] = 5;
        this->blob_->Reshape(shape);
        EXPECT_EQ(this->blob_->count(), 0);
    }

    TYPED_TEST(BlobSimpleTest, TestLegacyBlobProtoShapeEquals){
        BlobProto blob_proto;

        // Reshape to (3 x 2)
        vector<int> shape(2);
        shape[0] = 3;
        shape[1] = 2;
        this->blob_->Reshape(shape);

        // (3x2) blob == (1x1x3x2) legacy blob
        /**
         * ReshapeEquals :
         * if(other.has_num() || other.has_channels() ||
            other.has_height() || other.has_width()){
            因为shape.size=2, 所以进入这个if， 执行
            shape_.size() <= 4 && LegacyShape(-4) == other.num() &&
                    LegacyShape(-3) == other.channels() &&
                    LegacyShape(-2) == other.height() &&
                    LegacyShape(-1) == other.width();

            而LegacyShape 如果index超界，则为1;
         * if (index >= num_axes() || index < -num_axes()) {
                // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
                // indexing) -- this special case simulates the one-padding used to fill
                // extraneous axes of legacy blobs.
                return 1;
            }
         */
        blob_proto.set_num(1);
        blob_proto.set_channels(1);
        blob_proto.set_height(3);
        blob_proto.set_width(2);
        EXPECT_TRUE(this->blob_->ShapeEquals(blob_proto));

        // （3x2) blob != (0,1,3,2) legacy blob
        blob_proto.set_num(0);
        blob_proto.set_channels(1);
        blob_proto.set_height(3);
        blob_proto.set_width(2);
        EXPECT_FALSE(this->blob_->ShapeEquals(blob_proto));

        // Reshape to (1 x 3 x 2).
        shape.insert(shape.begin(), 1);
        this->blob_->Reshape(shape);

        // (1 x 3 x 2) blob == (1 x 1 x 3 x 2) legacy blob
        blob_proto.set_num(1);
        blob_proto.set_channels(1);
        blob_proto.set_height(3);
        blob_proto.set_width(2);
        EXPECT_TRUE(this->blob_->ShapeEquals(blob_proto));

        // Reshape to (2 x 3 x 2).
        shape[0] = 2;
        this->blob_->Reshape(shape);

        // (2 x 3 x 2) blob != (1 x 1 x 3 x 2) legacy blob
        blob_proto.set_num(1);
        blob_proto.set_channels(1);
        blob_proto.set_height(3);
        blob_proto.set_width(2);
        EXPECT_FALSE(this->blob_->ShapeEquals(blob_proto));
    }

    template <typename TypeParam>
    class BlobMathTest: public MultiDeviceTest<TypeParam> {
        typedef typename TypeParam::Dtype Dtype; // typename 将模板表示为类型
        protected:
            BlobMathTest():
                blob_(new Blob<Dtype>(2,3,4,5)),
                epsilon_(1e-6){}
            virtual ~BlobMathTest() {
                delete blob_;
            }
            Blob<Dtype>* const blob_;  // 指针内容不可变
            Dtype epsilon_;
    };

    /**
     * typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>,
                                GPUDevice<float>, GPUDevice<double> >
                                TestDtypesAndDevices;
     */
    TYPED_TEST_CASE(BlobMathTest, TestDtypesAndDevices);

    TYPED_TEST(BlobMathTest, TestsumOfSquares) {
        // TypeParam 是MultiDeviceTest的模板变量， 被实例化成GPUDevice和CPUDevice 结构体
        // 也就是TypeParam实际上表示为结构体变量
        /**
        template <typename TypeParam> // 此处的TypeParam表示float或double
        struct GPUDevice {
            typedef TypeParam Dtype;
            static const Caffe::Brew device = Caffe::GPU;
        };或

        template <typename TypeParam>
        struct CPUDevice {
            typedef TypeParam Dtype;
            static const Caffe::Brew device = Caffe::CPU;
        };
         */
        typedef typename TypeParam::Dtype Dtype;

        // unintialized blob should have sum of squares == 0
        EXPECT_EQ(0, this->blob_->sumsq_data());
        EXPECT_EQ(0, this->blob_->sumsq_diff());

        FillerParameter filler_param;
        filler_param.set_min(-3);
        filler_param.set_max(3);
        UniformFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_);

        Dtype expected_sumsq = 0;
        const Dtype * data = this->blob_->cpu_data();
        for (int i = 0; i < this->blob_->count(); ++i){
            expected_sumsq += data[i] * data[i];
        }

        // Do a mutable access on the current device,
        // so that the sumsq computation is done on that device.
        // (Otherwise, this would only check the CPU sumsq implementation.)
        switch (TypeParam::device)
        {
        case Caffe::CPU:
            // mutable的函数调用包含数据在不同设备上的传输过程，这样能测试更全面
            this->blob_->mutable_cpu_data();
            break;
        case Caffe::GPU: 
            this->blob_->mutable_gpu_data();
            break;
        default:
            LOG(FATAL)<< "Unknown device: " << TypeParam::device;
        }
        EXPECT_NEAR(expected_sumsq, this->blob_->sumsq_data(),
                this->epsilon_ * expected_sumsq);
        EXPECT_EQ(0, this->blob_->sumsq_diff());

        // check sumsq_diff too

    }


} // namespace caffe
