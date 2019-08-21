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

        
        
        // ------------------------------ 私有成员
        private: 
        DISABLE_COPY_AND_ASSIGN(Layer);
    };

} // namespace caffe



#endif