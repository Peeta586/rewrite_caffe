// Make sure we include Python.h before any system header
// to avoid _POSIX_C_SOURCE redefinition

#ifdef WITH_PYTHON_LAYER
#include <boost/python.hpp> 
#endif
#include <string>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"

#ifdef USE_CUDNN
// add some layer by cudnn
#endif


#ifdef WITH_PYTHON_LAYER
// include python_layer.hpp

#endif

namespace caffe {

// Get convolution layer according to engine
// template <typename Dtype>
// shared_ptr<Layer<Dtype> > GetCovolutionLayer(
//     const LayerParameter& param) {
//     CovolutionParameter conv_param = param.convolution_param();
//     ConvolutionParamter_Engine engin = conv_param.engine()

// }

} // namespace caffe