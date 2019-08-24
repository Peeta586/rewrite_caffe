/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 * 
 * ------------ 学习注册设计模式
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

    template <typename Dtype> 
    class Layer;

    template <typename Dtype> 
    class LayerRegistry {
        public: 
        // 函数指针Creator，返回的是Layer<Dtype>类型的指针
        typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
        // CreatorRegistry是字符串与对应的Creator的映射
        typedef std::map<string, Creator> CreatorRegistry;
        /**
         * 
        // 产生一个CreatorRegistry映射的的实例赋值给g_registry_
        // 表示内部的注册表
        // 静态函数，第一次的时候会new然后return，其余时间都是return
         */
        static CreatorRegistry& Registry() {
            static CreatorRegistry* g_registry_ = new CreatorRegistry();  // 创建一个map实例， CreatorRegistry是一个map实例
            return *g_registry_;
        }

        // add a creator
        // Creator 是一个函数指针类型, 制定不同层类的创造函数
        static void AddCreator(const string& type, Creator creator){
            CreatorRegistry& registry = Registry();
            CHECK_EQ(registry.count(type), 0)
                << "Layer type " << type << "already registered.";
            registry[type] = creator;
        }

        // Get a layer using a LayerParameter
        // Creator这个函数指针变量， 而CreateLayer是实际的函数， 执行指定类型的创造函数，并将param传入
        static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param){
            if(Caffe::root_solver()){
                LOG(INFO) << "Creating layer" << param.name();
            }
            const string& type = param.type();
            CreatorRegistry& registry = Registry(); // 静态函数， 直接调用会跟踪以前的状态，所以会统一维护唯一一个static CreatorRegistry* g_registry_这样一个map变量
            CHECK_EQ(registry.count(type), 1) << "Unknow layer type: " << type
                << "(known types: " << LayerTypeListString() << ")";
            
            return registry[type](param);
        }

        static vector<string> LayerTypeList() {
            CreatorRegistry& registry = Registry();
            vector<string> layer_types;
            // 这是在告诉编译器CreatorRegistry::iterator是个类型，而不是变量，因为CreatorRegistry是一个模板
            for(typename CreatorRegistry::iterator iter = registry.begin();
                iter != registry.end(); ++iter){
                layer_types.push_back(iter->first);  // only store name string
            }
        }

        private: 
        // Layer registry should never be instantiated - everything is done with its
        // static variables.
        // 构造函数私有化，该类不能被实例化，因此里面的函数都是静态类型，而且对象也是静态类型，因此常驻内存，但是是属于同一功能类
        LayerRegistry() {}  

        static string LayerTypeListString(){
            vector<string> layer_types = LayerTypeList();
            string layer_types_str;
            fro(vector<string>::iterator iter = layer_types.begin();
                iter != layer_types.end(); ++iter){
                    if (iter != layer_types.begin()){
                        layer_types_str += ",";
                    }
                    layer_types_str += *iter;
            }
            return layer_types_str;
        }
    };

    template <typename Dtype>
    class LayerRegisterer{
        public: 
        LayerRegisterer(const string& type,
                        shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter& )){
            // LOG(INFO) <<"Registering layer type: " << type;
            LayerRegistry<Dtype>::AddCreator(type, creator);
        } 
    };
    // #type 它代表把宏的参数变成字符串, 如果type是 conv1, 则#type就是“conv1”
    #define REGISTER_LAYER_CREATOR(type, creator) \
        static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>); \
        static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>); 
    
    #define REGISTER_LAYER_CLASS(type)  \
    template <typename Dtype> \
    shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
    { \
        return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param)) \
    } \
    REGISTER_LAYER_CREATOR(type, Creator_##type##Layer) 
    
} // namespace caffe


#endif



