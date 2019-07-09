#ifndef CAFFE_UTIL_CUDNN_H_
#define CAFFE_UTIL_CUDNN_H_
#ifdef USE_CUDNN

#include <cudnn.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#define CUDNN_VERSION_MIN(major, minor, patch) \
    (CUDA_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDNN_CHECK(condtion) \ 
    do{ \
        cudnnStatus_t status = condition; \
        CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " " \
            << cudnnGetErrorString(status); \
    } while (0)


inline const char* cudnnGetErrorString(cudnnStatus_t status) {
    switch (status)
    {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
#if CUDNN_VERSION_MIN(6,0,0)
    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
      return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
#endif
#if CUDNN_VERSION_MIN(7,0,0)
    case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
        return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
    case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
        return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
#endif
    }
    return "Unknow cudnn status";
}

namespace caffe {

namespace cudnn {

// 声明一个模板类，名称是dataDtype，模板类的数据类型DataType
template <typename Dtype> class dataType;

// 用float专门化该模板类
/*
template<>前缀表示这是一个专门化,描述时不用模板参数;
//compare.h
template <class T>
 class compare
 {
  public:
  bool equal(T t1, T t2)
  {
       return t1 == t2;
  }
};
//compare.h
#include <string.h>
template <class T>
 class compare
 {
  public:
  bool equal(T t1, T t2)
  {
       return t1 == t2;
  }
};
   
// 专门化就是为了某个特殊类型，进行特别的设计，如字符串比较需要用strcmp，代替等于号
template<>class compare<char *>  
{
public:
    bool equal(char* t1, char* t2)
    {
        return strcmp(t1, t2) == 0;
    }
};
 */
template<> class dataType<float> {
    public:
     static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
     static float oneval, zeroval;
     static const void *one, *zero;
};
typename<> class dataType<double> {
    public:
    static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
    static double oneval, zeroval;
    static const void *one, *zero;
};

/**
 *  对于cudnn的使用，第一步当然是在代码中包含cudnn的头文件。
 * 在使用cudnn的函数之前，需要创建cudnn的句柄，定义为：cudnnHandle_t 类型。
 *  对于卷积计算来说，主要有三个参数，输入，输出和权重。对于这个参数来说，在cudnn中称为tensor
 * 现在主要针对tensor格式来简要说明。对于一幅图像的存储，我们知道一般的存储方式为RGBA，RGBA，RGBA……的方式存储，、
 * 也就是4个通道的值交织一起。我们的例子考虑RGB三通道的图像，输入为一幅图，输出为卷积后的图像,采用OpenCV来操作图像。
 * 说完图像的一半存储方式。我们再来看cudnn中的格式，定义的类型为：cudnnTensorFormat_t。
 * 对应的值分别为：CUDNN_TENSOR_NCHW,CUDNN_TENSOR_NHWC,CUDNN_TENSOR_NCHW_VECT_C。
 * 从描述来说，我们可以知道对于输入的图像tensor，格式为CUDNN_TENSOR_NHWC,对应的输出图像格式也是CUDNN_TENSOR_NHWC。
 * 那我们现在来创建输入和输出tensor的描述； （也就是tensor的使用要提前添加描述，是针对怎样的数据做的输入）
 *  1) 先创建描述
 *  2）再设置描述
 * cudnnSetTensor4dDescriptor（output_descriptor
 *  /*format=*CUDNN_TENSOR_NHWC,
    /*dataType=*CUDNN_DATA_FLOAT,
    /*batch_size=*1,
    /*channels=*3,
    /*image_height=*image.rows,
    /*image_width=*image.cols）
 */

template <typename Dtype>
inline void createTensor4dDesc(cudnnTensorDescriptor_t* desc){
    CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}
// 
// C++ 不允许变量重名，但是允许多个函数取相同的名字，只要参数表不同即可，这叫作函数的重载
template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int n, int c, int h, int w,
    int stride_n, int stride_c, int stride_h, int stride_w) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
            n,c,h,w,stride_n,stride_c,stride_h,stride_w));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int n, int c, int h, int w) {
        const int stride_w = 1;
        const int stride_h = w * stride_w; // w
        const int stride_c = h * stride_h; // h*w
        const int stride_n = c * stride_c; //c *h * w
        setTensor4dDesc<Dtype>(desc, n, c, h, w, stride_n, stride_c, stride_h, stride_w);
}

//--------------------------- filter descriptor tensor的设置
// 不同版本的cudnn 调用的函数名有变化，因此需要宏判断
// 这个设置为了cudnn的使用提前创建描述符
/***
 * 对于权重数组的大小为input_channel*output_channel*kernel_size*kernel_size。
 * 在我们的例子中kernel_size为3,权重tensor描述如下
 *   cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*CUDNN_DATA_FLOAT,
                                      /*format=*CUDNN_TENSOR_NCHW,
                                      /*out_channels=*3,
                                      /*in_channels=*3,
                                      /*kernel_height=*3,
                                      /*kernel_width=*3);
 */
template <typename Dtype>
inline void createFilterDesc(cudnnFilterDecriptor_t* desc, 
    int n, int c, int h, int w) {
        CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
#if CUDNN_VERSION_MIN(5,0,0)
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type, 
        CUDNN_TENSOR_NCHW, n,c,h,w));
#else
    CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(*desc, dataType<Dtype>::type,
        CUDNN_TENSOR_NCHW,n,c,h,w));
#endif
}

// ---------------------set convolution函数参数的设置
/**
 * 对于权重tensor的格式，取决于我们保存的方式。对于tensorflow默认的保存方式为：kernel_h*kernel_w*input_channel*output_channel。
 * 对不符合上述几种格式描述，需要我们认为调整下。在我们的项目中，与IOS MPS保持一致，
 * 统一保存为：output_channel*kernel_h*kernel_w*input_channel，对于这种格式为CUDNN_TENSOR_NHWC。
 * 而这个例子中，采用output_channel*input_channel*kernel_h*kernel_w来说明，故格式为CUDNN_TENSOR_NCHW
 * 
   cudnnConvolutionDescriptor_t convolution_descriptor;
   cudnnCreateConvolutionDescriptor(&convolution_descriptor);
   cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=* 1,
                                           /*pad_width=*1,
                                           /*vertical_stride=*1,
                                           /*horizontal_stride=*1,
                                           /*dilation_height=*1,
                                           /*dilation_width=*1,
                                           /*mode=*CUDNN_CROSS_CORRELATION,
                                           /*computeType=*CUDNN_DATA_FLOAT));
 */
template <typename Dtype>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv){
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
}

template <typename Dtype>
inline void setConvolutionDesc(cudnnConvolutionDescriptor_t* conv,
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h,, int pad_w, int stride_h, int stride_w) {
#if CUDNN_VERSION_MIN(6,0,0)
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
        pad_h, pad_w, stride_h, stride_w, 1, 1 CUDNN_CROSS_CORRELATION,
        dataType<Dtype>::type));
#else
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
        pad_h, pad_w, stride_h, stride_w, 1,1,CUDNN_CROSS_CORRELATION));
#endif
}

template <typename Dtype>
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
        switch (poolmethod)
        {
        case PoolingParameter_PoolMethod_MAX:
            *mode = CUDNN_POOLING_MAX;
            break;
        case PoolingParameter_PoolMethod_AVE:
            *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
            break;
        
        default:
            LOG(FATAL) << "Unknown pooling method.";
        }
        CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
#if CUDNN_VERSION_MIN(5,0,0)
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode,
        CUDNN_PROPAGATE_NAN, h, w, pad_h, pad_w, stride_h, stride_w));
#else
    CUDNN_CHECK(cudnnSetPooling2dDescriptor_v4(*pool_desc, *mode,
        CUDNN_PROPAGATE_NAN, h, w, pad_h, pad_w, stride_h, stride_w));

#endif
}

template <typename Dtype>
inline void createActivationDescriptor(cudnnActivationDescriptor_t* activ_desc,
    cudnnActivationMode_t mode){
        CUDNN_CHECK(cudnnCreateActivationDescriptor(activ_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(*actv_desc, mode,
                                                CUDNN_PROPAGATE_NAN, Dtype(0)));
}

} // namespace cudnn

} // namespace caffe

#endif // USE_CUDNN
#endif // CAFFE_UTIL_CUDNN_H_