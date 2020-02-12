#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>  // pair
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/** 构建网络图, 通过NetParameter进行控制
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */

template <typename Dtype> 
class Net {
    public: 
     explicit Net(const NetParameter& param);
     /**
      * Phase, level, stages都是控制网络流的, 不同phase,level,stages的配置, 
      * 会根据prototxt中的配置自行组装不同层和不同结构的网络.
      * */
     explicit Net(const string& param_file, Phase phase,
        const int level = 0, const vector<string>* stages = NULL);

    virtual ~Net() {}

    // brief initialize a network with a NetParameter
    void Init(const NetParameter& param);

    // brief run forward and return the result
    const vector<Blob<Dtype>*>& Forward(Dtype* loss=NULL);
    // @brief DEPRECATED; use Forward() instead.
    const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss=NULL){
        LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: ForwardPredfilled() "
            << "will be removed in a future version. Use Forward().";
        return Forward(loss);
    }

    /** 指定网络从那一层开始,那一层结束; 用于一层到另一层的额外计算
     * $$$ 主要用于额外分支的计算过程 $$
     * The From and To variants of Forward and Backward operate on the
     * (topological) ordering by which the net is specified. For general DAG
     * networks, note that (1) computing from one layer to another might entail
     * extra computation on unrelated branches, and (2) computation starting in
     * the middle may be incorrect if all of the layers of a fan-in are not
     * included.
     */
    Dtype ForwardFromTo(int start, int end);
    Dtype ForwardFrom(int start);
    Dtype ForwardTo(int end);

    // brief DEPRECATED; set input blobs then use Forward() instead
    const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>*>& bottom,
        Dtype* loss=NULL);
    
    /** 在反传播之前, 清空diff, 类似pytorch的zero_grad
     * @brief Zeroes out the diffs of all net parameters.
     *        Should be run before Backward.
     */
    void ClearParamDiffs();

    /**
     * The network backward should take no input and output, since it solely
     * computes the gradient w.r.t the parameters, and the data has already been
     * provided during the forward pass.
     */
    void Backward();
    void BackwardFromTo(int start, int end);
    void BackwardFrom(int start);
    void BackwardTo(int end);

    /** 用于数据流动时, 保持数据大小一致性;
     * @brief Reshape all layers from bottom to top.
     *
     * This is useful to propagate changes to layer sizes without running
     * a forward pass, e.g. to compute output feature size.
     */
    void Reshape();

    Dtype ForwardBackward() {
        Dtype loss;
        Forward(&loss);
        Backward();
        return loss;
    }

    /// @brief Updates the network weights based on the diff values computed.
    void Update();

    /**  $$ 存在疑问, 这个共享权重啥意思? $$$
     * @brief Shares weight data of owner blobs with shared blobs.
     *
     * Note: this is called by Net::Init, and thus should normally not be
     * called manually.
     */
    void ShareWeights();

    /**  执行权重初始化的来源于其他网络的层
     * @brief For an already initialized net, implicitly copies (i.e., using no
     *        additional memory) the pre-trained layers from another Net.
     */
    void ShareTrainedLayersWith(const Net* other);

    // For an already initialized net, CopyTrainedLayersFrom() copies the already
    // trained layers from another net parameter instance.
    /**
     * @brief For an already initialized net, copies the pre-trained layers from
     *        another Net.
     */
    void CopyTrainedLayersFrom(const NetParameter& param);
    void CopyTrainedLayersFrom(const string& trained_filename);
    void CopyTrainedLayersFromBinaryProto(const string& trained_filename);
    void CopyTrainedLayersFromHDF5(const string& trained_filename);

    // brief Writes the net to a proto.
    void ToProto(NetParameter* param, bool write_diff = false) const;
    // brief Writes the net to an HDF5 file
    void ToHDF5(const string& filename, bool write_diff = false) const;

    // brief returns the network name.
    inline const string& name() const { return name_; }
    // brief return the layer names, 返回引用, 则返回的不应该是一个局部变量
    inline const vector<string>& layer_names() const { return layer_names_; }
    // brief returns the blob names
    inline const vector<string>& blob_names() const { return blob_names_; }
    // brief returns the blobs
    inline const vector<shared_ptr<Blob<Dtype> >& blobs() const {
        return blob_;
    }

    /// @brief returns the layers
    inline const vector<shared_ptr<Layer<Dtype> >& layers() const {
        return layers_;
    }

    


};


}  // namespace caffe


#endif