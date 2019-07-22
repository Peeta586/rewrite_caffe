#ifndef CAFFE_BLOB_HPP_
#define CAFEE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

// blob的坐标轴最大32
const int kMaxBlobAxes = 32;

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */

template <typename Dtype>
class Blob {
    public:
        Blob():data_(), diff_(), count_(0), capacity_(0){}

        // @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
        explicit Blob(const int num, const int channels, const int height, const int width);
        explicit Blob(const vector<int>& shape);

        /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
        void Reshape(const int num, const int channels, const int height,
            const int width);


    protected:
        shared_ptr<SyncedMemory> data_;
        shared_ptr<SyncedMemory> diff_;
        shared_prt<SyncedMemory> shape_data_;
        vector<int> shape_;
        int count_;
        int capacity_;

        DISABLE_COPY_AND_ASSIGN(Blob);
}



#endif