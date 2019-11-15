#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <boost/filesystem.hpp>
#include <iomanip>
#include <iostream> //NOLINT (readability/streams)
#include <string>

#include "google/protobuf/message.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

//  RETRIES: 重新审理 ? 临时路径审视
#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100 
#endif

namespace caffe{

using ::google::protobuf::Message;
using ::boost::filesystem::path;

/**
 * 这个函数就是创建一个日期（某天）为粒度的文件夹表示， 然后在这一天里查找不同时刻生成的文件夹
 * 这样能保证文件夹命名唯一。
 */
inline void MakeTempDir(string* temp_dirname){
    temp_dirname->clear();
    const path& model = boost::filesystem::temp_directory_path()/"caffe_test.%%%%-%%%%";
    for(int i =0; i < CAFFE_TMP_DIR_RETRIES; i++){
        /** 获取以时间为粒度的文件表示
         * The unique_path function generates a path name suitable for creating temporary files, 
         * including directories. The name is based on a model that uses the percent sign character 
         * to specify replacement by a random hexadecimal digit. [Note: The more bits of randomness 
         * in the generated path name, the less likelihood of prior existence or being guessed. 
         * Each replacement hexadecimal digit in the model adds four bits of randomness. 
         * The default model thus provides 64 bits of randomness. 
         * This is sufficient for most applications. —end note]
            Returns: A path identical to model, 
            except that each occurrence of a percent sign character is replaced 
            by a random hexadecimal digit character in the range 0-9, a-f.
        Throws: As specified in Error reporting.

        Remarks: Implementations are encouraged to obtain the required randomness 
        via a cryptographically secure pseudo-random number generator, 
        such as one provided by the operating system. [Note: 
        Such generators may block until sufficient entropy develops. —end note]

         * inline 得到随机的16进制数字序列，代替“caffe_test.%%%%-%%%%”中的百分号%，形成新的文件夹名，但是路径与model相同
            path unique_path(const path& p="%%%%-%%%%-%%%%-%%%%")
                                       { return detail::unique_path(p); }
        */
        const path& dir = boost::filesystem::unique_path(model).string();
        bool done = boost::filesystem::create_directory(dir);
        if( done ){
            *temp_dirname = dir.string();
            return;
        }
    }
    LOG(FATAL) << "Failed to create a temporary directory";
}

inline void MakeTempFilename(string* temp_filename){
    static path temp_files_subpath;
    static uint64_t next_temp_file = 0;
    temp_filename->clear();
    if(temp_files_subpath.empty()){
        string path_string = "";
        MakeTempDir(&path_string);
        temp_files_subpath = path_string;
    }
    // 文件名长度为9，
    *temp_filename = (temp_files_subpath/caffe::format_int(next_temp_file++, 9)).string();
}

bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const string& filename, Message* proto){
    return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto){
    CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto){
    CHECK(ReadProtoFromTextFile(filename.c_str(), proto));
}

// **************************
void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename){
    WriteProtoToTextFile(proto, filename.c_str());
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);
inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

// 文件写到数据库， lmdb/leveldb
bool ReadFileToDatum(const string& filename, const int label, Datum* datum);
inline bool ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}
inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

bool DecodeDatumNative(Datum* datum);
bool DecodeDatum(Datum* datum, bool is_color);

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename, 
    const int height, const int width, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
const int height, const int width);

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);

#endif // USE_OPENCV

} // namespace caffe

#endif // !CAFFE_UTIL_IO_H_