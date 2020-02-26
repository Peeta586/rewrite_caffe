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

    // brief returns the phase: TTRAIN or TEST
    // Phase: 在caffe proto中定义, caffe.pb.h中
    inline Phase phase() const { return phase_; }
    
    /** 返回网络中每一层的bottom向量, 也就是每一层的输入,每一层输入不一定一个,可以有很多,所以是向量的向量;
     * 一个网络的所有层输入组成数组返回
     * @brief returns the bottom vecs for each layer -- usually you won't
     *        need this unless you do per-layer checks such as gradients.
     */
    inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
        return bottom_vecs_;
    }
    /**
     * @brief returns the top vecs for each layer -- usually you won't
     *        need this unless you do per-layer checks such as gradients.
     */
    inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
        return top_vecs_;
    }

    // brief returns the ids of the top blobs of layer i
    inline const vector<int>& top_ids(int i) const {
        CHECK_GE(i, 0) <<"Invalid layer id";
        CHECK_LT(i, top_id_vec_.size()) << "Invalid layer id";
        return top_id_vecs_[i];
    }
    /// @brief returns the ids of the bottom blobs of layer i
    inline const vector<int> & bottom_ids(int i) const {
        CHECK_GE(i, 0) << "Invalid layer id";
        CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
        return bottom_id_vecs_[i];
    }

    inline const vector<vector<bool> >& bottom_need_backward() const {
        return bottom_need_backward_;
    }

    inline const vector<Dtype>& blob_loss_weights() const {
        return blob_loss_weights_;
    }
    inline const vector<bool>& layer_need_backward() const {
        return layer_need_backward_;
    }
    // brief returns the parameters
    inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
        return params_;
    }
    inline const vector<Blob<Dtype>*>& learnable_params() const {
        return learnable_params_;
    }
     
    // brief returns the learnable parameter learning rate multipiers
    inline const vector<float>& param_lr() const { return params_lr_; }
    inline const vector<bool>& has_param_lr() const { return has_param_lr_; }

    // brief: returns the learnable parameter decay multipliers
    inline const vector<float>& params_weight_decay() const {
        return params_weigh_decay_;
    }
    inline const vector<bool>& has_params_decay() const {
        return has_params_decay_;
    }

    const map<string, int>& param_names_index() const {
        return param_names_index_;
    }
    inline const vector<int>& param_owners() const { return param_owners_; }
    inline const vector<string>& param_display_names() const {
        return param_display_names_;
    }

    // brief: Input and output blob numbers.
    inline int num_inputs() const { return net_input_blobs_.size(); }
    inline int num_outputs() const { return net_output_blobs_.size(); }
    inline const vector<Blob<Dtype>*>& input_blobs() const {
        return net_input_blobs_;
    }
    inline const vector<Blob<Dtype>*>& output_blobs() const {
        return net_output_blobs_;
    }

    inline const vector<int>& input_blob_indices() const {
        return net_input_blob_indices_;
    }
    inline const vector<int>& output_blob_indices() const {
        return net_output_blob_indices_;
    }

    bool has_blob(const string& blob_name)const;
    const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
    bool has_layer(const string& layer_name) const; 
    const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;
    void set_debug_info(const bool value) { debug_info_ = value; }

    // Helpers for Init.
    /**
     * @brief Remove layers that the user specified should be excluded given the current
     *        phase, level, and stage.
     */
    static void FilterNet(const NetParameter& param,
        NetParameter* param_filtered);
    // brief: return whether NetState state meets NetStateRule rule
    static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
        const string& layer_name);

    // Invoked at specific points during an iteration
    // 会不会是和pytorch中的hook函数一样, 是一个钩子函数, 用于跟踪数据前传播或反传中某一层的具体细节
    class Callback {
        protected: 
        virtual void run(int layer) = 0;

        template <typename T> 
        friend class Net;
    }

    const vector<Callback*>& before_forward() const { return before_forward_; }
    void add_before_forward(Callback* value){
        before_forward_.push_back(value);
    }
    const vector<Callback*>& after_forward() const { return after_forward_; }
    void add_after_forward(Callback* value){
        after_forward_.push_back(value);
    }
    const vector<Callback*>& before_backward() const { return before_backward_; }
    void add_before_backward(Callback* value) {
        before_backward_.push_back(value);
    }
    const vector<Callback*>& after_backward() const { return after_backward_; }
    void add_after_backward(Callback* value) {
        after_backward_.push_back(value);
    }

    protected: 
    // Helpers for Init, 添加到网络指定层等
    /// brief Append a new top blob to the net
    void AppendTop(const NetParameter& param, const int layer_id,
                    const int top_id, set<string>* available_blobs,
                    map<string, int>* blob_name_to_idx);
    /// brief append a new bottom blob to the net.
    int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
    /// @brief Append a new parameter blob to the net.
    void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

    /// brief Helper for displaying debug info in forward
    void ForwardDebugInfo(const int layer_id);
    /// @brief Helper for displaying debug info in Backward.
    void BackwardDebugInfo(const int layer_id);
    /// @brief Helper for displaying debug info in Update.
    void UpdateDebugInfo(const int param_id);

    // 变量*************************
    // network name
    string name_;
    // brief the phase: TRIAN or TEST
    Phase phase_;

    // brief Individual layers in the net
    vector<shared_ptr<Layer<Dtype> > > layers_;
    vector<string> layer_names_;
    map<string,int> layer_names_index_;
    vector<bool> layer_need_backward_; //相当于pytorch的required_grad

    // brief the blobs storing intermediate results between the layer
    vector<shared_ptr<Blob<Dtype> > > blobs_;  // 模型训练过程中的Feature map
    vector<string> blob_names_;
    map<string, int> blob_names_index_;
    vector<blob> blob_need_backward_;

    /// bottom_vecs stores the vectors containing the input for each layer.
    /// They don't actually host the blobs (blobs_ does), so we simply store
    /// pointers.
    // 这是将blobs_中的实际内容抽象化, 用指针数组来表示层结构
    vector<vector<Blob<Dtype>*> > bottom_vecs_;
    vector<vector<int> > bottom_id_vecs_;
    vector<vector<bool> > bottom_need_backward_;

    /// top_vecs stores the vectors containing the output for each layer
    vector<vector<Blob<Dtype>*> > top_vecs_;
    vector<vector<int> > top_id_vecs_;

    /// Vector of weight in the loss (or objective) function of each net blob,
    /// indexed by blob_id.
    // 计算损失函数时, 每个层的权重的损失系数, 以层为单位;
    // 每个层都有一个param{ lr:, weight: }这是更新权重的时候的伸缩系数
    vector<Dtype> blob_loss_weights_;
    vector<vector<int> > param_id_vecs_;
    vector<int> param_owners_;
    vector<string> param_display_names_;
    vector<pair<int,int> > param_layer_indices_;
    map<string, int> param_names_index_;

    


};


}  // namespace caffe


#endif