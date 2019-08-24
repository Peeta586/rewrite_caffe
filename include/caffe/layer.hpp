#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>


#include "blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost {class mutex; }

namespace caffe {
    /**
     * @brief An interface for the units of computation which can be composed into a
     *        Net.
     *
     * Layer%s must implement a Forward function, in which they take their input
     * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
     * They may also implement a Backward function, in which they compute the error
     * gradients with respect to their input Blob%s, given the error gradients with
     * their output Blob%s.
     * 一个统一接口， 也就是一个基类，用于封装各种不同的层进入到caffe中
     */

    template <typename Dtype> 
    class Layer {
        public: 
        /**
         * You should not implement your own constructor. Any set up code should go
         * to SetUp(), where the dimensions of the bottom blobs are provided to the
         * layer.
         * 再setup函数中获取输入的维度
         */
        explicit Layer(const LayerParameter& param): layer_param_(param){
            // set phase and copy blobs (if there are any)
            phase_ = param.phase();
            if(layer_param_.blobs_size() > 0){
                blobs_.resize(layer_param_.blobs_size());
                for(int i = 0; i < layer_param_.blobs_size(), ++i){
                    blobs_[i].reset(new Blob<Dtype>());
                    blobs_[i]->FromProto(layer_param_.blobs(i));
                }
            }
        }
        virtual ~Layer() {}

        /**
         * @brief Implements common layer setup functionality.
         *
         * @param bottom the preshaped input blobs
         * @param top
         *     the allocated but unshaped output blobs, to be shaped by Reshape
         *
         * Checks that the number of bottom and top blobs is correct.
         * Calls LayerSetUp to do special layer setup for individual layer types,
         * followed by Reshape to set up sizes of top blobs and internal buffers.
         * Sets up the loss weight multiplier blobs for any non-zero loss weights.
         * This method may not be overridden.
         */



        /**
         * @brief Sets the loss associated with a top blob at a given index.
         */
        inline void set_loss(const int top_index, const Dtype value){
            if(loss_.size() <= top_index){
                loss_.resize(top_index + 1, Dtype(0));
            }
            loss_[top_index] = value;
        }

        // ----------------------------- 保护成员
        protected: 
        /** The protobuf that stores the layer parameters */
        LayerParameter layer_param_;
        /** The phase: TRAIN or TEST */
        Phase phase_;
        /** The vector that stores the learnable parameters as a set of blobs. */
        vector<shared_ptr<Blob<Dtype> > > blobs_;
        /** Vector indicating whether to compute the diff of each param blob. */
        vector<bool> param_propagate_down_;

        /** The vector that indicates whether each top blob has a non-zero weight in
        *  the objective function. */
        vector<Dtype> loss_;

        //---------------------------- 前传函数
        /** @brief Using the CPU device, compute the layer output. */
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) = 0;
        /**
         * @brief Using the GPU device, compute the layer output.
         *        Fall back to Forward_cpu() if unavailable.[这个设计比较好，值得学习]
         */
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
            // LOG(WARNING) << "Using CPU code as backup";
            // 可能不需要return
            return Forward_cpu(bottom, top);
        }

        /**
         * @brief Using the CPU device, compute the gradients for any parameters and
         *        for the bottom blobs if propagate_down is true.
         */
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& top) = 0;
        /**
         * @brief Using the GPU device, compute the gradients for any parameters and
         *        for the bottom blobs if propagate_down is true.
         *        Fall back to Backward_cpu() if unavailable.
         */
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& top){
            // LOG(WARNING) << "Using CPU code as backup.";
            Backward_cpu(top, propagate_down, bottom);
        }

        /**
         * Called by the parent Layer's SetUp to check that the number of bottom
         * and top Blobs provided as input match the expected numbers specified by
         * the {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
         */
        virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top){
            if(ExactNumBottomBlobs() >=0){
                CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
                    << type() << "Layer takes " << ExactNumBottomBlobs()
                    <<"bottom blob(s) as input";
            }
            if(MinBottomNBlobs() >= 0) {
                CHECK_LE(MinBottomBlobs(), bottom.size())
                    << type() << "Layer takes at least " << MinBottomBlobs()
                    << "bottom blob(s) as input.";
            }
            if(MaxBottomNBlobs() >= 0) {
                CHECK_GE(MaxBottomBlobs(), bottom.size())
                    << type() << "Layer takes at least " << MaxBottomBlobs()
                    << "bottom blob(s) as input.";
            }
            if (ExactNumTopBlobs() >= 0) {
                CHECK_EQ(ExactNumTopBlobs(), top.size())
                    << type() << " Layer produces " << ExactNumTopBlobs()
                    << " top blob(s) as output.";
            }
            if (MinTopBlobs() >= 0) {
                CHECK_LE(MinTopBlobs(), top.size())
                    << type() << " Layer produces at least " << MinTopBlobs()
                    << " top blob(s) as output.";
            }
            if (MaxTopBlobs() >= 0) {
                CHECK_GE(MaxTopBlobs(), top.size())
                    << type() << " Layer produces at most " << MaxTopBlobs()
                    << " top blob(s) as output.";
            }
            if (EqualNumBottomTopBlobs()) {
                CHECK_EQ(bottom.size(), top.size())
                    << type() << " Layer produces one top blob as output for each "
                    << "bottom blob input.";    
            }
        }
        /**
         * Called by SetUp to initialize the weights associated with any top blobs in
         * the loss function. Store non-zero loss weights in the diff blob.
         * 对输出损失计算的时候，我们赋予每个输出top不同的权重，来控制输出或者哪个输出需要计算loss
         */
        inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {
            const int num_loss_weights = layer_param_.loss_weight_size();
            if(num_loss_weights) {
                CHECK_EQ(top.size(), num_loss_weights)<< "loss_weight must be "
                    <<"unspectified or specified once per top blob.";
                for(int top_id =0; top_id < top.size(); ++top_id){
                    const Dtype loss_weight = layer_param_.loss_weight(top_id);
                    if(loss_weight == Dtype(0)) { continue; }
                    // 不仅仅对该层loss赋值loss_weight， 而且对权重的diff设置loss_weight
                    this->set_loss(top_id, loss_weight); // loss_[top_id] = loss_weight;
                    const int count = top[top_id]->count();
                    // 注意 loss_weight 计算在cpu上
                    Dtype* loss_multiplier = top[top_id]->multable_cpu_diff();
                    caffe_set(count, loss_weight, loss_multiplier);
                }
            }
        }
        
        // ------------------------------ 私有成员
        private: 
        DISABLE_COPY_AND_ASSIGN(Layer);
    };

} // namespace caffe



#endif