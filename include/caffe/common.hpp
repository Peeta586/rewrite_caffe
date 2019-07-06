#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits> //定义最大最小值，无穷什么的
#include <cmath>

//That's a comment. In this case, it's a comment designed to be read by a static analysis tool to tell it to shut up about this line.
#include <fstream> // NOLINT(readability/streams)
#include <iostream> // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream> //ostringstream:用于输出操作,istringstream:用于输入操作,stringstream:用于输入输出操作
#include <string>
#include <utility>
#include <vector>

#include "caffe/util/device_alternate.hpp"




#endif // CAFFE_COMMON_HPP_