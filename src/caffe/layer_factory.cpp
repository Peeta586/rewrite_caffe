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

#endif


#ifdef WITH_PYTHON_LAYER

#endif