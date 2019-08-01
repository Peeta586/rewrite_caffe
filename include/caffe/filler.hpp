// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP_
#define CAFFE_FILLER_HPP_
#include <string>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe{

    /**
     * 这是个基类，用虚函数定义一些成员函数，或者析构函数， 虚函数用于实现多态的，也就是基类指针指向不同的子类会调用不同子类的虚函数实现
     * （from 百度百科）简单的总结，指向基类的指针在操作它的多态类对象时，会根据不同的类对象，调用其相应的函数，这个函数就是虚函数。
     * 析构函数定义为虚是为了在delete 基类指针时，基类指针指向的子类对象也会被是否； 否则只释放父类的。
     */
    /// @brief Fills a Blob with constant or randomly-generated data.
    template <typename Dtype> 
    class Filler {
        public: 
            explicit Filler(const FillerParameter& param): filler_param_(param){}
            virtual ~Filler() {}
            virtual void Fill(Blob<Dtype>* blob) = 0;
        protected: 
            FillerParameter filler_param_;
    }; // class Filler

    /// @brief Fills a Blob with constant values @f$ x = 0 @f$.
    template <typename Dtype>
    class ConstantFiller : public Filler<Dtype> {
        public: 
            explicit ConstantFiller(const FillerParameter& param): Filler<Dtype>(param){}

            virtual void Fill(Blob<Dtype>* blob){
                Dtype* data = blob->mutable_cpu_data();
                const int count = blob->count();
                // 这个可能通过读取prototxt获取
                const Dtype value = this->filler_param_.value();
                CHECK(count);
                for(int i =0; i< count; ++i){
                    data[i] = value;
                }
                // The expected number of non-zero output weights for a given input in
                // Gaussian filler -- the default -1 means don't perform sparsification.
                CHECK_EQ(this->filler_param_.sparse(), -1)
                    << "Sparsity not supported by this filler";
            }
    };

    /// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$
    template <typename Dtype> 
    class UniformFiller : public Filler<Dtype> {
        public:
            explicit UniformFiller(const FillerParameter& param) : Filler<Dtype>(param){}

            virtual void Fill(Blob<Dtype>* blob){
                CHECK(blob->count());
                caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()), 
                    Dtype(this->filler_param_.max()), blob->mutable_cpu_data());
                CHECK_EQ(this->filler_param_.sparse(), -1)
                    << "sparsity not supported by this filler";
            }
    };

    /// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
    template <typename Dtype> 
    class GaussianFiller : public Filler<Dtype> {
        public: 
            explicit GaussianFiller(const FillerParameter& param) : Filler<Dtype>(param){}
            virtual void Fill(Blob<Dtype>* blob){
                Dtype* data =blob->mutable_cpu_data();
                CHECK(blob->count());
                caffe_rng_gaussion<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
                    Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
                int sparse = this->filler_param_.sparse();
                CHECK_GE(sparse, -1);
                if (sparse >= 0){
                    // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
                    // These have num == channels == 1; width is number of inputs; height is
                    // number of outputs.  The 'sparse' variable specifies the mean number(平均个数)
                    // of non-zero input weights for a given output.
                    CHECK_GE(blob->num_axes(), 1);
                    const int num_outputs = blob->shape(0);
                    // probability= 1 / outputs
                    Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
                    rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
                    int * mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
                    // 以伯努利的分布实现稀疏化mask的生成， 按照概率是non_zero_probability= sparse/ outputs
                    // mask 非零即一;
                    caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
                    for(int i =0; i < blob->count(); ++i){
                        data[i] *= mask[i];
                    }
                }
            }
        protected: 
            shared_ptr<SyncedMemory> rand_vec_;
    };

    /** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
     *         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
     * 将总和为1的值填充到blob中
     */
    template <typename Dtype> 
    class PositiveUnitballFiller : public Filler<Dtype> {
        public: 
            explicit PositiveUnitballFiller(const FillerParameter& param) : Filler<Dtype>(param){}
            virtual void Fill(Blob<Dtype>* blob){
                Dtype* data = blob->mutable_cpu_data();
                // debug-only checking.  not executed in NDEBUG mode.
                DCHECK(blob->count());
                // 0,1 之间均匀分布， 然后再将每个数除以该样本的总和，从而使得所有初始化分布的总和为1，且呈现均匀分布
                caffe_rng_uniform<Dtype>(blob->count(), 0, 1, blob->mutable_cpu_data());
                // We expect the filler to not be called very frequently, so we will
                // just use a simple implementation
                int dim = blob->count() / blob->shape(0);
                CHECK(dim);
                for(int i =0; i <blob->shape(0); ++i){
                    Dtype sum = 0;
                    for(int j = 0; j < dim; ++j){
                        sum += data[i * dim + j];
                    }
                    for(int j = 0; j < dim; ++j){
                        data[i * dim + j] /= sum;
                    }
                }
                CHECK_EQ(this->filler_param_->sparse(), -1) 
                    << "sparsity not supported by this filler";
            }
    };

    /**
     * @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$ is
     *        set inversely proportional to number of incoming nodes, outgoing
     *        nodes, or their average.
     *
     * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
     * the difficulty of training deep feedforward neuralnetworks.
     *
     * It fills the incoming matrix by randomly sampling uniform data from [-scale,
     * scale] where scale = sqrt(3 / n) where n is the fan_in, fan_out, or their
     * average, depending on the variance_norm option. You should make sure the
     * input blob has shape (num, a, b, c)[注意这是卷积核filter的shape不是featuremap的shape
     * 所以num是输出通道，a是输入通道数， b,c是卷积核的大小] where a * b * c = fan_in and num * b * c
     * = fan_out. Note that this is currently not the case for inner product layers.
     *
     * TODO(dox): make notation in above comment consistent with rest & use LaTeX.
     */
    template <typename Dtype>
    class XavierFiller : public Filler<Dtype> {
        public: 
            explicit XavierFiller(const FillerParameter& param):Filler<Dtype>(param){}

            virtual void Fill(Blob<Dtype>* blob){
                CHECK(blob->count());
                int fan_in = blob->count() / blob->shape(0);
                // compatibility with nd blobs
                // 全连接层num_axes=2, 这个blob 是权重blob; 全连接层的权重num_axes=2
                int fan_out = blob->num_axes() > 1 ? blob->count() / blob->shape(1) : blob->count();
                Dtype n = fan_in; // default to fan in
                if(this->filler_param_.variance_norm() == FillerParameter_VarianceNorm_AVERAGE){
                    n = (fan_in + fan_out) / Dtype(2);
                }else if (this->filler_param_.variance_norm() == FillerParameter_VarianceNorm_FAN_OUT){
                    n = fan_out;
                }
                Dtype scale = sqrt(Dtype(3) / n);
                caffe_rng_uniform<Dtype>(blob->count(), -scale, scale, blob->mutable_cpu_data());
                CHECK_EQ(this->filler_param_.sparse(), -1)
                    << "sparsity not supported by this filler";
            }
    };

    /**
     * @brief Fills a Blob with values @f$ x \sim N(0, \sigma^2) @f$ where
     *        @f$ \sigma^2 @f$ is set inversely proportional to number of incoming
     *        nodes, outgoing nodes, or their average.
     *
     * A Filler based on the paper [He, Zhang, Ren and Sun 2015]: Specifically
     * accounts for ReLU nonlinearities.
     *
     * Aside: for another perspective on the scaling factor, see the derivation of
     * [Saxe, McClelland, and Ganguli 2013 (v3)].
     *
     * It fills the incoming matrix by randomly sampling Gaussian data with std =
     * sqrt(2 / n) where n is the fan_in, fan_out, or their average, depending on
     * the variance_norm option. You should make sure the input blob has shape (num,
     * a, b, c) where a * b * c = fan_in and num * b * c = fan_out. Note that this
     * is currently not the case for inner product layers.
     */
    template <typename Dtype> 
    class MSRAFiller : public Filler<Dtype> {
        public: 
            explicit MSRAFiller(const FillerParameter& param): Filler<Dtype>(param) { }

            virtual void Fill(Blob<Dtype>* blob) {
                CHECK(blob->count());
                int fan_in = blob->count() / blob->shape(0);

                // compatibility with ND blobs
                int fan_out = blob->num_axes > 1 ? blob->count() / blob->shape(1) : blob->count();

                Dtype n = fan_in; // default to fan_in
                if(this->filler_param_.variance_norm() == FillerParameter_VarianceNorm_AVERAGE){
                    n = (fan_in + fan_out) / Dtype(2);
                } else if (this->filler_param_.variance_norm() == FillerParameter_VarianceNorm_FAN_OUT){
                    n = fan_out;
                }
                Dtype std = sqrt(Dtype(2) / n);
                caffe_rng_gaussion<Dtype>(blob->count(), Dtype(0), std, blob->mutable_cpu_data());
                CHECK_EQ(this->filler_param_.sparse(), -1) << "Sparsity not supported by this filler.";
            }
    };

    /*!
    @brief Fills a Blob with coefficients for bilinear interpolation.

    A common use case is with the DeconvolutionLayer acting as upsampling.
    You can upsample a feature map with shape of (B, C, H, W) by any integer factor
    using the following proto.
    \code
    layer {
    name: "upsample", type: "Deconvolution"
    bottom: "{{bottom_name}}" top: "{{top_name}}"
    convolution_param {
        kernel_size: {{2 * factor - factor % 2}} stride: {{factor}}
        num_output: {{C}} group: {{C}}
        pad: {{ceil((factor - 1) / 2.)}}
        weight_filler: { type: "bilinear" } bias_term: false
    }
    param { lr_mult: 0 decay_mult: 0 }
    }
    \endcode
    Please use this by replacing `{{}}` with your values. By specifying
    `num_output: {{C}} group: {{C}}`, it behaves as
    channel-wise convolution. The filter shape of this deconvolution layer will be
    (C, 1, K, K) where K is `kernel_size`, and this filler will set a (K, K)
    interpolation kernel for every channel of the filter identically. The resulting
    shape of the top feature map will be (B, C, factor * H, factor * W).
    Note that the learning rate and the
    weight decay are set to 0 in order to keep coefficient values of bilinear
    interpolation unchanged during training. If you apply this to an image, this
    operation is equivalent to the following call in Python with Scikit.Image.
    \code{.py}
    out = skimage.transform.rescale(img, factor, mode='constant', cval=0)
    \endcode
    这个主要是将上采样操作用反向卷积操作的形式代替， 上采样其实相当于2x2的反卷积操作，
    而权重是按照二插值上采样的操作的映射关系进行权重操作。
    */

    
    template <typename Dtype>
    class BilinearFiller : public Filler<Dtype> {
        public:
            explicit BilinearFiller(const FillerParameter& param):Filler<Dtype>(param){}

            virtual void Fill(Blob<Dtype>* blob){
                CHECK_EQ(blob->num_axes(), 4) << "blob must be 4 dim";
                CHECK_EQ(blob->width(), blob->height()) << "Filler must be square";
                Dtype* data = blob->mutable_cpu_data();
                int f = ceil(blob->width() / 2.);
                Dtype c = (blob->width() - 1) / (2. * f);
                for( int i = 0; i < blob->count(); ++i){
                    Dtype x = i % blob->width();
                    Dtype y = (i / blob->width()) % blob->height();
                    data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
                }
                CHECK_EQ(this->filler_param_.sparse(), -1)
                    << "Sparsity not supported by this filler";
            }
    };

    /**
     * @brief Get a specific filler from the specification given in FillerParameter.
     *
     * Ideally this would be replaced by a factory pattern, but we will leave it
     * this way for now.
     */
    template <typename Dtype> 
    Filler<Dtype> * GetFiller(const FillerParameter& param){
        const std::string& type = param.type();
        if(type == "constant"){
            return new ConstantFiller<Dtype>(param);
        } else if (type == "gaussian") {
            return new GaussianFiller<Dtype>(param);
        } else if (type == "positive_unitball") {
            return new PositiveUnitballFiller<Dtype>(param);
        } else if (type == "uniform") {
            return new UniformFiller<Dtype>(param);
        } else if (type == "xavier") {
            return new XavierFiller<Dtype>(param);
        } else if (type == "msra") {
            return new MSRAFiller<Dtype>(param);
        } else if (type == "bilinear") {
            return new BilinearFiller<Dtype>(param);
        } else{
            CHECK(false) << "Unknow filler name: " << param.type();
        }
        return (Filler<Dtype>*) (NULL);
    }

} // namespace caffe


#endif // CAFFE_FILLER_HPP_