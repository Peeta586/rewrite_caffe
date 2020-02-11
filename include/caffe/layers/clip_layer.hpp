#ifndef CAFFE_CLIP_LAYER_HPP_
#define CAFFE_CLIP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"


namespace caffe{
/**
 * @brief Clip: @f$ y = \max(min, \min(max, x)) @f$.
 */
template <typename Dtype> 
class ClipLayer : public NeuronLayer<Dtype> {
    public: 
    /**
     * @ param provides ClipParameter clip_param,
     *    with ClipLayer options:
     *  - min
     *  - max
     */
    explicit ClipLayer(const LayerParameter& param)
        : NeuronLayer<Dtype>(param){}
    virtual inline const char* type() const { return "clip"; }

};
 
} // namespace caffe



#endif // !CAFFE_CLIP_LAYER_HPP_

