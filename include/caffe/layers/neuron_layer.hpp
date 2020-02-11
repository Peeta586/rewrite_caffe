#ifndef CAFFE_NEURON_LAYER_HPP_
#define CAFFE_NEURON_LAYER_HPP_

#include <vector>

#include  "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
/**
 *  个人理解:  neuronLayer主要是用于一些对神经元进行原地操作的一些操作类的抽象
 *  也就是不进行太大的数值计算,而是对神经元进行一种过滤操作, 如clip,等操作; 可继承
 *  neuronLayer; 正如类的如下说明, 输入和输出元素个数是一样的,只是进行了一些简单操作.
 * 
 * @brief An interface for layers that take one blob as input (@f$ x @f$)
 *        and produce one equally-sized blob as output (@f$ y @f$), where
 *        each element of the output depends only on the corresponding input
 *        element.
 */

template <typename Dtype> 
class NeuronLayer : public Layer<Dtype> {
    public: 
    explicit NeuronLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top);
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
};

} // namespace caffe


#endif // CAFFE_NEURON_LAYER_HPP_