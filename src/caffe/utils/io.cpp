#include <fcntl.h>  //open(file) and change desc of file
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/highgui/highgui_c.h> 
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

#include <stdint.h> // int的一些类型

#include <algorithm>
#include <fstream> // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX; // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream; // 读取文件流
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream; // 类似输入流，但是尽量少的拷贝
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::Message;

// .proto中定义每个模块，都是用message
bool ReadProtoFromTextFile(const char* filename, Message* proto){
    /**
     * open函数是我们开发中经常会遇到的，这个函数是对文件设备的打开操作，
     * 这个函数会返回一个句柄fd，我们通过这个句柄fd对设备文件读写操作。
     * 
     *  调用成功时返回一个文件描述符fd
        调用失败时返回-1，并修改errno

        open函数返回的文件描述符fd一定是未使用的最小的文件描述符，那么如果0没有使用，
        那么我们open的时候，首先就会获取到fd=0的情况。
        默认情况下，0,1,2这三个句柄对应的是标准输入，标准输出，标准错误，
        系统进程默认会打开0，1，2这三个文件描述符，而且指向了键盘和显示器的设备文件。
        所以通常我们open的返回值是从3开始的。
        如果我们在open之前，close其中的任何一个，则open的时候，则会用到close的最小的值：
     */
    int fd = open(filename, O_RDONLY); // 仅读
    CHECK_NE(fd, -1) << "File not found: " << filename;
    FileInputStream* input = new FileInputStream(fd);
    // 将.proto文件读成Message
    bool success = google::protobuf::TextFormat::Parse(input, proto);
    delete input;
    close(fd);
    return success; // 打开是否成功
}

void WriteProtoToTextFile(const Message& proto, const char* filename){
    int fd = open(filename, O_RDONLY);
    // fd是系统默认分配的描述符，基本上就是一个int类型整形数， 比如，0,1,2分别表示标准输入，输出，错误； 
    // 3以后开始表示系统当前分配这个要操作的文件的描述符； 也就是这是一个文件句柄， 由系统管控
    FileOutputStream* output = new FileOutputStream(fd); // 从系统的这个描述符开始建立文件流
    CHECK(google::protobuf::TextFormat::Print(proto, output));
    delete output;
    close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto){
    int fd = open(filename, O_RDONLY);
    CHECK_NE(fd, -1) << "file not found: " << filename;
    ZeroCopyInputStream* raw_input = new FileInputStream(fd);
    /**CodedInputStream实现反序列化逻辑
     * protobuf使用CodedOutputStream实现序列化逻辑、CodedInputStream实现反序列化逻辑，
     * 他们都包含write/read基本类型和Message类型的方法，write方法中同时包含fieldNumber和value参数，
     * 在写入时先写入由fieldNumber和WireType组成的tag值
     * （添加这个WireType类型信息是为了在对无法识别的字段编码时可以通过这个类型信息判断使用那种方式解析这个未知字段，
     * 所以这几种类型值即可），这个tag值是一个可变长int类型，所谓的可变长类型就是一个字节的最高位（msb，most significant bit）
     * 用1表示后一个字节属于当前字段，而最高位0表示当前字段编码结束。在写入tag值后，
     * 再写入字段值value，对不同的字段类型采用不同的编码方式：
     */
    CodedInputStream* coded_input = new CodedInputStream(raw_input); // 解析二进制
    coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

    bool success = proto->ParseFromCodedStream(coded_input);
    delete raw_input;
    delete coded_input;
    close(fd);
    return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename){
    // ios::trunc ios::trunc 覆盖存在的文件 即如果原来文件中有数据原来的数据就被清空了，清空后再写入就可以了
    fstream output(filename, ios::out | ios::trunc | ios::binary); // 文件流
    // 不需要CodedOutputStream实现序列化逻辑 输出
    //读取时使用CodedInputStream
    CHECK(proto.SerializePartialToOstream(&output)); 
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename, 
    const int height, const int width, const bool is_color) {
    
    cv::Mat cv_img;
    int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
    if(!cv_img_origin.data){
        LOG(ERROR) << "Could not open or find file " << filename;
        return cv_img_origin;
    }
    // 根据参数进行resize image
    if(height > 0 && width > 0){
        cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
    }else{
        cv_img = cv_img_origin;
    }
    return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width){
    return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename, 
    const bool is_color){
    return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename){
    return ReadImageToCVMat(filename, 0, 0, true); // 默认彩色
}

// Do the file extension and encoding match?
static bool matchExt(const std::string& fn, std::string en){
    size_t p = fn.rfind('.');
    std::string ext = p != fn.npos ? fn.substr(p + 1) : fn;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower); // 后缀名转化成小写字母
    std::transform(en.begin(), en.end(), en.begin(), ::tolower); // 编码转小写
    if(ext == en){
        return true;
    }
    if( en == "jpg" && ext == "jpeg")
        return true;
    return false;
}

bool ReadImageToDatum(const string& filename, const int label, 
    const int height, const int width, const bool is_color,
    const std::string& encoding, Datum* datum){
    cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
    if(cv_img.data){
        if(encoding.size()){
            if((cv_img.channels() == 3) == is_color && !height && !width 
                && matchExt(filename, encoding))
                return ReadFileToDatum(filename, label, datum);

            std::vector<uchar> buf;
            cv::imencode("." + encoding, cv_img, buf); // 将图像按照一定编码模式转到vector buf 中
            // 将地址转化成数字， 然后在转化成string类型，形成键值对的key； 一般地址唯一
            datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]), buf.size()));
            datum->set_label(label);
            datum->set_encoded(true);
            return true;
        }
        CVMatToDatum(cv_img, datum);
        datum->set_label(label);
        return true;
    } else {
        return false;
    }
}

#endif // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label, Datum* datum){
    std::streampos size; // 数据流位置
    // Open and seek to end immediately after opening, 当已打开时寻找到EOF
    fstream file(filename.c_str(), ios::in | ios::binary|ios::ate);
    if(file.is_open()){
        size = file.tellg(); // 告诉数据流总的大小 gobal count
        /** std::string buffer(size, ' ')调用的是下面的函数， 声明指定大小的string
         * 
        *  @brief  Construct string as multiple characters.
        *  @param  __n  Number of characters.
        *  @param  __c  Character to use.
        *  @param  __a  Allocator to use (default is default allocator).
        *
        basic_string(size_type __n, _CharT __c, const _Alloc& __a = _Alloc());
         */
        // 这个内存管理比自己定义new char[]更加方便
        // 定义大小为size， 内容为space的字符串, 相当于char类型的数组,也可以声明vector<char>但是可能read不能匹配
        std::string buffer(size, ' '); 
        file.seekg(0, ios::beg); // Request a seek relative to the beginning of the stream
        file.read(&buffer[0], size);
        file.close();
        datum->set_data(buffer);
        datum->set_label(label);
        datum->set_encoded(true);
        return true;
    }else{
        return false;
    }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum){
    cv::Mat cv_img;
    CHECK(datum.encoded()) << "datum not encoded";
    const string& data = datum.data(); // string 或者char*; 写的时候是string进去的，则读取也是string接受
    std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
    cv_img = cv::imdecode(vec_data, -1);
    if(!cv_img.data){
        LOG(ERROR) << " Could not decode datum ";
    }
    return cv_img;
}

cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color){
    cv::Mat cv_img;
    CHECK(datum.encoded()) << " Datum not encoded ";
    const string& data = datum.data();
    std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
    int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
    cv_img = cv::imdecode(vec_data, cv_read_flag);
    if(!cv_img.data){
        LOG(ERROR) << " Could not decode datum ";
    }
    return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum){
    if(datum->encoded()){
        cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
        CVMatToDatum(cv_img, datum);
        return true;
    }else{
        return false;
    }
}

bool DecodeDatum(Datum* datum, bool is_color){
    if(datum->encoded()){
        cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
        CVMatToDatum(cv_img, datum);
        return true;
    }else{
        return false;
    }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum){
    CHECK(cv_img.depth() == CV_8U) << "Image data type must be usigned byte";
    datum->set_channels(cv_img.channels());
    datum->set_height(cv_img.rows);
    datum->set_width(cv_img.cols);
    datum->clear_data();
    datum->clear_float_data();
    datum->set_encoded(false);
    int datum_channels = datum->channels();
    int datum_height = datum->height();
    int datum_width = datum->width();
    int datum_size = datum_channels * datum_height * datum_width;
    std::string buffer(datum_size, ' ');
    for(int h = 0; h < datum_height; ++h){
        const uchar* ptr = cv_img.ptr<uchar>(h);
        int img_index = 0;
        for(int w = 0; w < datum_width; ++w){
            for(int c = 0; c < datum_channels; ++c){
                int datum_index = (c * datum_height + h) * datum_width + w;
                buffer[datum_index] = static_cast<char>(ptr[img_index++]);
            }
        }
    }
    datum->set_data(buffer);
}

#endif // USE_OPENCV

} // namespace caffe

