/*
author LSHM

 */

// stdint.h 包含一些数据类型的宏定义，如int16_t, 16位整形数，uint32_t 32位无符号整形数 等。
// 也可以定义一些常亮或者最大值最小值等
#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>  # make_pair等操作　自动包含map

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe; // NOLINT(build/namespaces)



