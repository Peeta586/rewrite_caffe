#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

// 使用gflags

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs seperated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch_size is multiplied by the number of devices. ");
DEFINE_string(solver, "", 
    "the solver definition protocol buffer text file");
DEFINE_string(model, "",
    "the model definition protocol buffer text file");
DEFINE_string(phase, "",
    "Optional; network phase (TRAIN or TEST). Only used for 'time'." );
DEFINE_int32(level, 0,
    "Optional; network level.");
DEFINE_string(stage, "",
    "OPtional; network stages (not to be confused with phase), "
    "separated by ','.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "seperated by ','. Cannot be set simultaneously with snapshot");
DEFINE_int32(iterations, 50,
    "the number of iterations to run");
// snapshot的存储是采用信号控制的方式， signal_handle.h也就是处理信号的
DEFINE_string(sigint_effect, "stop",
    "Optional; action to take when a sigint signal is received:"
    "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
    "optional; action to take when a sighup signal is received: ",
    "snapshot, stop or none");

// a simple registry for caffe commands; 简单的caffe命令注册器
// Brew酿制，混合
typedef int (*BrewFunction)(); // using BrewFunction = int (*)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

/*
这个注释： /* NOLINT */表示不让审核
*/
#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
    public: /* NOLINT */ \
        __Registerer_##func() { \
            // #func是将func直接转化为字符串，也就是两边加上“”
            g_brew_map[#func] = &func; \
        } \ 
}; \
// func函数都添加到该注册类中的g_brew_map字典中，用函数的名字符串作为键，函数地址作为value，这样生成一个map
// 注意g_brew_map是std::map<caffe::string, BrewFunction> 结构
__Registerer_##func g_registerer_##func; \
}

