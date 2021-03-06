#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

/***
 * blob的书写方式参考了，"caffe/proto/caffe.pb.h"中类的写法， 获取数据函数时，有两种形式，一种是const，一种是mutable
 * 
 */

namespace caffe {

    /** =------------------------------------------------------
     * 注意 Blob的Reshape不是跟numpy的reshape函数一样，而是用于申请内存空间，创建实例。
     * 
     * 
     */
    template <typename Dtype>
    void Blob<Dtype>::Reshape(const int num, const int channels, const int height, const int width){
        vector<int> shape(4);
        shape[0] = num;
        shape[1] = channels;
        shape[2] = height;
        shape[3] = width;
        Reshape(shape);
    }

    template <typename Dtype>
    void Blob<Dtype>::Reshape(const vector<int>& shape){
        /**
         *  1) 更新count_, shape_, shape_data_的值
         *  2) 如果reshape更大空间，需要重新reset data_, diff_, 否则用已有的空间
         *  3) 注意reshape不是numpy那样，保证原始数据不变，而是重新申请内存空间
         */
        CHECK_LE(shape.size(), kMaxBlobAxes);
        count_ = 1;
        shape_.resize(shape.size());
        if(!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)){
            shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
        }
        int * shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
        for(int i =0; i< shape.size(); ++i){
            CHECK_GE(shape[i], 0);
            if(count_ != 0){
                CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
            }
            count_ *= shape[i];
            shape_[i] = shape[i];
            shape_data[i] = shape[i];
        }
        // 如果shape的大小大于原始容量，则重新定义data_, diff_
        // capacity_ = 0, 该函数被Blob构造函数调用，用于创建对象申请空间，因此，capacity_必须被初始化为0，
        // 这样才能比较
        if(count_ > capacity_){
            capacity_ = count_;
            data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
            diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
        }
    }

    template <typename Dtype>
    void Blob<Dtype>::Reshape(const BlobShape& shape){
        CHECK_LE(shape.dim_size(), kMaxBlobAxes);
        vector<int> shape_vec(shape.dim_size());
        for(int i=0; i< shape.dim_size(); ++i){
            shape_vec[i] = shape.dim(i);
        }
        Reshape(shape_vec);
    }
    template <typename Dtype> 
    void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other){
        Reshape(other.shape());
    }

    template <typename Dtype>
    Blob<Dtype>::Blob(const int n, const int channels, const int height, const int width)
        :capacity_(0){
        // capacisty_ must be initialized before calling Reshape
        Reshape(n, channels, height, width);
    }

    template <typename Dtype>
    Blob<Dtype>::Blob(const vector<int>& shape):capacity_(0){
        // capacity_ must be initialized before calling Reshape
        Reshape(shape);
    }

    template <typename Dtype>
    const int* Blob<Dtype>::gpu_shape() const {
        CHECK(shape_data_);
        return (const int* ) shape_data_->gpu_data();
    }

    template <typename Dtype>
    const Dtype* Blob<Dtype>::cpu_data() const {
        CHECK(data_);
        return (const Dtype*)data_->cpu_data();
    }

    template <typename Dtype>
    void Blob<Dtype>::set_cpu_data(Dtype* data){
        CHECK(data);
        // make sure cpu and gpu sizes remain equal
        size_t size = count_ * sizeof(Dtype);
        if (data_->size() != size){
            data_.reset(new SyncedMemory(size));
            diff_.reset(new SyncedMemory(size));
        }
        // call SyncedMemory class function
        data_->set_cpu_data(data);
    }

    template <typename Dtype>
    const Dtype* Blob<Dtype>::gpu_data() const {
        CHECK(data_);
        return (const Dtype*) data_->gpu_data();
    }

    template <typename Dtype>
    void Blob<Dtype>::set_gpu_data(Dtype* data){
        CHECK(data);
        // make sure cpu and gpu sizes remain equal
        size_t size = count_ * sizeof(Dtype);
        if (data_->size() != size){
            data_.reset(new SyncedMemory(size));
            diff_.reset(new SyncedMemory(size));
        }
        // call SyncedMemory class function
        data_->set_gpu_data(data);
    }

    template <typename Dtype>
    const Dtype* Blob<Dtype>::cpu_diff() const {
        CHECK(diff_);
        return (const Dtype*)diff_->cpu_data();
    }

    template <typename Dtype>
    const Dtype* Blob<Dtype>::gpu_diff() const {
        CHECK(diff_);
        return (const Dtype*)diff_->gpu_data();
    }

    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_cpu_data(){
        CHECK(data_);
        return static_cast<Dtype*>(data_->mutable_cpu_data());
    }

    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_gpu_data() {
        CHECK(data_);
        return static_cast<Dtype*>(data_->mutable_gpu_data());
    }

    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_cpu_diff() {
        CHECK(diff_);
        return static_cast<Dtype*>(diff_->mutable_cpu_data());
    }

    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_gpu_diff() {
        CHECK(diff_);
        return static_cast<Dtype*>(diff_->mutable_gpu_data());
    }

    template <typename Dtype>
    void Blob<Dtype>::ShareData(const Blob& other){
        CHECK_EQ(count_, other.count());
        data_ = other.data();
    }

    template <typename Dtype>
        void Blob<Dtype>::ShareDiff(const Blob& other) {
        CHECK_EQ(count_, other.count());
        diff_ = other.diff();
    }

    // The "update" method is used for parameter blobs in a Net, which are stored
    // as Blob<float> or Blob<double> -- hence we do not define it for
    // Blob<int> or Blob<unsigned int>.
    // 这是特例化， 对于这种类型的模板不执行， 与实例化不同，实例化没有<>,
    template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
    template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }

    template <typename Dtype>
    void Blob<Dtype>::Update(){
        // we will perform update based on where the data is located
        // 实现权重更新，将diff*(-1)+data 
        switch (data_->head())
        {
        case SyncedMemory::HEAD_AT_CPU:
            // perform computation on CPU
            caffe_axpy<Dtype>(count_, Dtype(-1), static_cast<const Dtype*>(diff_->cpu_data()),
                static_cast<Dtype*>(data_->mutable_cpu_data()));
            break;
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED: 
        #ifndef CPU_ONLY
            // perform computation on GPU
            caffe_gpu_axpy<Dtype>(count_, Dtype(-1), static_cast<const Dtype*>(diff_->gpu_data()),
                static_cast<Dtype*>(data_->mutable_gpu_data()));
        #else
            NO_GPU;
        #endif
            break;
        default:
            LOG(FATAL) << "Syncedmem not initialized [in blob.cpp]";
        }
    }

    template <> unsigned int Blob<unsigned int>::asum_data() const {
        NOT_IMPLEMENTED;
        return 0;
    }
    template <> int Blob<int>::asum_data() const {
        NOT_IMPLEMENTED;
        return 0;
    }

    template <typename Dtype>
    Dtype Blob<Dtype>::asum_data() const {
        if(!data_) { return 0; }
        switch (data_->head())
        {
        case SyncedMemory::HEAD_AT_CPU:
            return caffe_cpu_asum(count_, cpu_data());
        case SyncedMemory::HEAD_AT_GPU: 
        case SyncedMemory::SYNCED: 
        #ifndef CPU_ONLY
            {
                Dtype asum;
                caffe_gpu_asum(count_, gpu_data(), &asum);
                return asum;
            }
        #else
            NO_GPU;
        #endif
        case SyncedMemory::UNINITIALIZED:
            return 0;
        default:
            LOG(FATAL) << "Unknown SyncedMemory head state" << data_->head();
        }
        return 0;
    }

    template <> unsigned int Blob<unsigned int>::asum_diff() const {
        NOT_IMPLEMENTED;
        return 0;
    }

    template <> int Blob<int>::asum_diff() const {
        NOT_IMPLEMENTED;
        return 0;
    }

    template <typename Dtype>
    Dtype Blob<Dtype>::asum_diff() const {
        if (!diff_) { return 0; }
        switch (diff_->head()) {
            case SyncedMemory::HEAD_AT_CPU:
                return caffe_cpu_asum(count_, cpu_diff());
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
            #ifndef CPU_ONLY
                {
                    Dtype asum;
                    caffe_gpu_asum(count_, gpu_diff(), &asum);
                    return asum;
                }
            #else
                NO_GPU;
            #endif
                break;
            case SyncedMemory::UNINITIALIZED:
                return 0;
            default:
                LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
        }
        return 0;
    }

    template <> unsigned int Blob<unsigned int>::sumsq_data() const {
        NOT_IMPLEMENTED;
        return 0;
    }

    template <> int Blob<int>::sumsq_data() const {
        NOT_IMPLEMENTED;
        return 0;
    }

    template <typename Dtype>
    Dtype Blob<Dtype>::sumsq_data() const {
        Dtype sumsq;
        const Dtype* data;
        if (!data_) { return 0;}
        switch (data_->head())
        {
        case SyncedMemory::HEAD_AT_CPU:
            data = cpu_data();
            sumsq = caffe_cpu_dot(count_, data, data);
            break;
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED: 
        #ifndef CPU_ONLY
            data = gpu_data();
            caffe_gpu_dot(count_, data,data,&sumsq);
        #else
            NO_GPU;
        #endif 
            break;
        case SyncedMemory::UNINITIALIZED:
            return 0;
        default:
            LOG(FATAL)<< "Unknown SyncedMemory head state" << data_->head();
        }
        return sumsq;
    }

    template <> unsigned int Blob<unsigned int>::sumsq_diff() const {
        NOT_IMPLEMENTED;
        return 0;
    }

    template <> int Blob<int>::sumsq_diff() const {
        NOT_IMPLEMENTED;
        return 0;
    }

    template <typename Dtype>
    Dtype Blob<Dtype>::sumsq_diff() const {
        Dtype sumsq;
        const Dtype* diff;
        if (!diff_) { return 0; }
        switch (diff_->head()) {
            case SyncedMemory::HEAD_AT_CPU:
                diff = cpu_diff();
                sumsq = caffe_cpu_dot(count_, diff, diff);
                break;
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
            #ifndef CPU_ONLY
                diff = gpu_diff();
                caffe_gpu_dot(count_, diff, diff, &sumsq);
                break;
            #else
                NO_GPU;
            #endif
            case SyncedMemory::UNINITIALIZED:
                return 0;
            default:
                LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
        }
        return sumsq;
    }

    template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
        NOT_IMPLEMENTED;
    }

    template <> void Blob<int>::scale_data(int scale_factor) {
        NOT_IMPLEMENTED;
    }

    template <typename Dtype>
    void Blob<Dtype>::scale_data(Dtype scale_factor){
        Dtype* data;
        if(!data_) { return;}
        switch (data_->head())
        {
        case SyncedMemory::HEAD_AT_CPU:
            data = mutable_cpu_data();
            caffe_scal(count_, scale_factor, data);
            return;
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED: 
        #ifndef CPU_ONLY
            data = mutable_gpu_data();
            caffe_gpu_scal(count_, scale_factor, data);
            return;
        #else
            NO_GPU;
        #endif
        default:
            break;
        }
    }

    template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
        NOT_IMPLEMENTED;
    }

    template <> void Blob<int>::scale_diff(int scale_factor) {
        NOT_IMPLEMENTED;
    }
    
    template <typename Dtype>
    void Blob<Dtype>::scale_diff(Dtype scale_factor) {
        Dtype* diff;
        if (!diff_) { return; }
        switch (diff_->head()) {
            case SyncedMemory::HEAD_AT_CPU:
                diff = mutable_cpu_diff();
                caffe_scal(count_, scale_factor, diff);
                return;
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
            #ifndef CPU_ONLY
                diff = mutable_gpu_diff();
                caffe_gpu_scal(count_, scale_factor, diff);
                return;
            #else
                NO_GPU;
            #endif
            case SyncedMemory::UNINITIALIZED:
                return;
            default:
                LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
        }
    }

    template <typename Dtype> 
    bool Blob<Dtype>::ShapeEquals(const BlobProto& other){
        if(other.has_num() || other.has_channels() || 
            other.has_height() || other.has_width()){
            // 也就是如果blobproto使用的是旧的shape表示，则使用LegacyShape去获取对应的值，与other比较
            // Using deprecated 4D Blob dimensions --
            // shape is (num, channels, height, width).
            // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
            // methods as these index from the beginning of the blob shape, where legacy
            // parameter blobs were indexed from the end of the blob shape (e.g., bias
            // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
            return shape_.size() <= 4 && LegacyShape(-4) == other.num() &&
                    LegacyShape(-3) == other.channels() &&
                    LegacyShape(-2) == other.height() &&
                    LegacyShape(-1) == other.width();
        }
        // 将BlobProto 形式的shape---> vector<int>形式
        vector<int> other_shape(other.shape().dim_size());
        for(int i = 0; i < other.shape().dim_size(); ++i){
            other_shape[i] = other.shape().dim(i);
        }
        return shape_ == other_shape;
    }

    template <typename Dtype> 
    void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape){
        if(source.count() != count_ || source.shape() != shape_) {
            if(reshape) {
                ReshapeLike(source);
            } else {
                LOG(FATAL) << "Trying to copy blobs of different sizes.";
            }
        }
        switch (Caffe::mode())
        {
        case Caffe::GPU:
            if(copy_diff){
                caffe_copy(count_, source.gpu_diff(), static_cast<Dtype*>(diff_->mutable_gpu_data()));
            } else {
                caffe_copy(count_, source.gpu_data(), static_cast<Dtype*>(data_->mutable_gpu_data()));
            }
            break;
        case Caffe::CPU: 
            if(copy_diff){
                caffe_copy(count_, source.cpu_diff(), static_cast<Dtype*>(diff_->mutable_cpu_data()));
            } else {
                caffe_copy(count_, source.cpu_data(), static_cast<Dtype*>(data_->mutable_cpu_data()));
            }
            break;
        default:
            LOG(FATAL)<< "Unknow caffe mode.";
        }
    }

    template <typename Dtype> 
    void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape){
        if(reshape){
            vector<int> shape;
            // 将proto的shape转化为vector<int>形式表示
            if(proto.has_num() || proto.has_channels() || 
            proto.has_height() || proto.has_width()){
                // Using deprecated 4D Blob dimensions --
                // shape is (num, channels, height, width).
                shape.resize(4);
                shape[0] = proto.num();
                shape[1] = proto.channels();
                shape[2] = proto.height();
                shape[3] = proto.width();
            } else{
                shape.resize(proto.shape().dim_size());
                for (int i = 0; i< proto.shape().dim_size(); ++i){
                    shape[i] = proto.shape().dim(i);
                }
            }
            Reshape(shape);
        } else {
            CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
        }
        // copy data
        Dtype* data_vec = mutable_cpu_data();
        // 当数据类型为double时的处理， 因为proto里面设置了double_data
        // 内存空间数据与protobuf类型数据的交互
        if(proto.double_data_size() > 0){ // double type
            CHECK_EQ(count_, proto.double_data_size());
            for(int i =0; i< count_; ++i){
                data_vec[i] = proto.double_data(i);
            }
        } else {  // float type
            CHECK_EQ(count_, proto.data_size());
            for(int i =0; i< count_; ++i){
                data_vec[i] = proto.data(i);
            }
        }
        // 因为有的层可能没有diff，所以不在外部定义diff_vec变量
        if(proto.double_diff_size() > 0){
            CHECK_EQ(count_, proto.double_diff_size());
            Dtype* diff_vec = mutable_cpu_diff();
            for(int i = 0; i < count_; ++i){
                diff_vec[i] = proto.double_diff(i);
            }
        } else if (proto.diff_size() > 0) {  // float type
            CHECK_EQ(count_, proto.diff_size());
            Dtype* diff_vec = mutable_cpu_diff();
            for(int i =0; i< count_; ++i){
                diff_vec[i] = proto.diff(i);
            }
        }
    }

    template <> 
    void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
        proto->clear_shape();
        for(int i =0; i< shape_.size(); ++i){
            proto->mutable_shape()->add_dim(shape_[i]);
        }
        proto->clear_double_diff();
        proto->clear_double_data();
        const double* data_vec = cpu_data();
        for(int i=0; i< count_; ++i){
            proto->add_double_data(data_vec[i]);
        }
        if(write_diff){
            const double* diff_vec = cpu_diff();
            for(int i=0; i< count_; ++i){
                proto->add_double_diff(diff_vec[i]);
            }
        }
    }

    template <> 
    void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
        proto->clear_shape();
        for(int i =0; i< shape_.size(); ++i){
            proto->mutable_shape()->add_dim(shape_[i]);
        }
        proto->clear_diff();
        proto->clear_data();
        const float* data_vec = cpu_data();
        for(int i=0; i< count_; ++i){
            proto->add_data(data_vec[i]);
        }
        if(write_diff){
            const float* diff_vec = cpu_diff();
            for(int i=0; i< count_; ++i){
                proto->add_diff(diff_vec[i]);
            }
        }
    }

    // 最后注意下Blob的显示模板实例化：int, unsigned int, float, double:
    INSTANTIATE_CLASS(Blob);
    template class Blob<int>;
    template class Blob<unsigned int>;

} // namespace caffe