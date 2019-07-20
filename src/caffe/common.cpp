#include <boost/thread.hpp>
#include <glog/logging.h>
#include <cmath>
#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
    // make sure each thread can have different values
    // 因为Caffe是个singleton类，且有静态变量， 因为该类是不可重入的，所以多线程情况下如果不使用下面的thread_specific_ptr， 则该类不安全
    // 该类采用get， reset方式进行获取和创建的操作
    static boost::thread_specific_ptr<Caffe> thread_instance_;

    Caffe& Caffe::Get() {
        if (!thread_instance_.get()){
            thread_instance_.reset(new Caffe());
        }
        return *(thread_instance_.get());
    }

    // random seeding cluster_seedgen()的功能就是获取一个int64_t类型的随机种子
    int64_t cluster_seedgen(void){
        int64_t s, seed, pid;
        // 用系统的熵池或者时间来初始化随机数, 打开系统熵池获取随机数
        FILE* f = fopen("/dev/urandom", "rb");
        //fopen fread from stdio.h
        /*
        fopen函数是打开一个文件，其调用的一般形式为：
        文件指针名=fopen（文件名,使用文件方式）;
        “文件指针名”必须是被声明为FILE 类型的指针变 [1]  量
            
            fread是一个函数，它从文件流中读数据，函数原型为：
            size_t fread ( void *buffer, size_t size, size_t count, FILE *stream) ;
            从给定流 stream 读取数据，最多读取count个项，每个项size个字节，
            如果调用成功返回实际读取到的项个数（小于或等于count），如果不成功或读到文件末尾返回 0,
            所以：
            fread(&seed, 1, sizeof(seed), f) == sizeof(seed)表示调用成功
         */
        if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)){
            fclose(f);
            return seed;
        }
        // 如果上面使用系统熵池的方法出错，则会继续执行，则需要输出通知， 下面使用时间来初始化随机数
        LOG(INFO) << "System entropy source not available, "
                    "using fallback algorithm to generate seed instead";
        
        if (f) // 如果已经打开，而熵池不可用则关闭文件
            fclose(f);
        
        pid = getpid();
        s = time(NULL);
        seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
        return seed;
    }

    // a global initialization function that you should call in your main function.
    //currently it initializes google flags and google logging.
    // 这个全局初始化函数需要再main中调用，为了初始化整个工程的一些全局配置
    void GlobalInit(int* pargc, char*** pargv){
        // Google flags
        ::gflags::ParseCommandLineFlags(pargc, pargv, true);
        // google logging
        ::google::InitGoogleLogging(*(pargv)[0]);
        // provide a backtrace on segfault
        // The function should be called before threads are created, if you want
        // to use the failure signal handler for all threads.  The stack trace
        // will be shown only for the thread that receives the signal.  In other
        // words, stack traces of other threads won't be shown.
        ::google::InstallFailureSignalHandler();
    }

#ifdef CPU_ONLY // CPU-only Caffe
    Caffe::Caffe():random_generator_(), mode_(Caffe::CPU),solver_count_(1),solver_rank_(0),multiprocess_(false) { };
    Caffe::~Caffe() { };

    void Caffe::set_random_seed(const unsigned int seed){
        // RNG seed
        Get().random_generator_.reset(new RNG(seed));
    }
    void Caffe::SetDevice(const int device_id) {
        NO_GPU;
    }
    void Caffe::DeviceQuery() {
        NO_GPU;
    }
    bool Caffe::CheckDevice(const int device_id){
        NO_GPU;
        return false;
    }
    int Caffe::FindDevice(const int start_id){
        NO_GPU;
        return -1;
    }
    
    //用boost::mt19937 产生一个随机数
    // 真正产生随机数的是Generator 调用boost::mt19937产生的，而外面有包装了一个RNG 感觉冗余啊
    // 
    class Caffe::RNG::Generator {
        public:
        // Caffe::rng_t is typedef boost::mt19937 
            Generator():rng_(new caffe::rng_t(cluster_seedgen())) {}
            explicit Generator(unsigned int seed):rng_(new caffe::rng_t(seed)){}
            caffe::rng_t* rng() { return rng_.get();}
        private:
            shared_ptr<caffe::rng_t> rng_;
    };
    // 实现RNG 类的一些函数
    Caffe::RNG::RNG():generator_(new Generator()){ }
    Caffe::RNG::RNG(unsigned int seed):generator_(new Generator(seed)) { }
    Caffe::RNG& Caffe::RNG::operator=(const RNG& other){
        generator_ = other.generator_;
        return *this;
    }
    void* Caffe::RNG::generator(){
        return static_cast<void*>(generator_->rng());
    } 

#else // Normal GPU + CPU caffe

    Caffe::Caffe():cublas_handle_(NULL), curand_generator_(NULL), random_generator_(),mode_(Caffe::GPU),
                solver_count_(1), solver_rank_(0), multiprocess_(false) {
        // Try to create a cublas handler, and report an error if failed (but we will
        // keep the program running as one might just want to run CPU code).
        if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS){
            LOG(ERROR) << "can not create Cublas handle. cublas won't be available.";
        }
        // try to create a curand handle
        if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT) !=
            CURAND_STATUS_SUCCESS || curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen()) 
            != CURAND_STATUS_SUCCESS){
            LOG(ERROR) << "can not create curand genertor. curand won't be available.";
        }
    }
    Caffe::~Caffe(){
        if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
        if (curand_generator_){
            CURAND_CHECK(curandDestroyGenerator(curand_generator_));
        }
    }

    // random_generator_还是boost产生的，
    // 并且也设置了成员变量curand_generator_的随机种子， 供后面GPU上用
    void Caffe::set_random_seed(const unsigned int seed){
        //curand seed
        static bool g_curand_availability_logged = false;
        if(Get().curand_generator_){
            // curand_generator() 与Get().curand_generator_ 一样
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(), seed));
            CURAND_CHECK(curandSetGeneratorOffset(curand_generator(), 0));
        } else {
            // 如果第一次Get().curand_generator_ 出错，则后面跳过这个设置，不会再有提示，因为g_curand_availability_logged被设为true了
            if(!g_curand_availability_logged){
                LOG(ERROR) << "Curand not available. skipping setting the curand seed.";
                g_curand_availability_logged = true;
            }
        }
        //RNG seed
        Get().random_generator_.reset(new RNG(seed));
    }

    void Caffe::SetDevice(const int device_id){
        int current_device;
        CUDA_CHECK(cudaGetDevice(&current_device));
        if (current_device == device_id){
            return;
        }
        // 调用Get前要设置SetDevice
        // the call to cudaSetDevice must come before any calls to Get, which
        // may perform initialization using the GPU
        CUDA_CHECK(cudaSetDevice(device_id));
        // 每次设置设备的时候。需要重新设置handle和curand_generator_， 因为不同的设备不同的地址空间，需要释放旧的，然后再重新创建
        if (Get().cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
        if (Get().curand_generator_) {
            CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
        }
        CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
        CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_, cluster_seedgen()));
    }

    void Caffe::DeviceQuery(){
        cudaDeviceProp prop;
        int device;
        if (cudaSuccess != cudaGetDevice(&device)){
            printf("no cuda device present.\n");
            return;
        }
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        LOG(INFO) << "Device id:                      " << device;
        LOG(INFO) << "Major revision number:          " << prop.major;
        LOG(INFO) << "Minor revision number:          " << prop.minor;
        LOG(INFO) << "Name:                           " << prop.name;
        LOG(INFO) << "Total global memory:            " << prop.totalGlobalMem;
        LOG(INFO) << "Total shared memory per block:  " << prop.sharedMemPerBlock;
        LOG(INFO) << "Total registers per block:      " << prop.regsPerBlock;
        LOG(INFO) << "warp size:                      " << prop.warpSize;
        LOG(INFO) << "Maximum memory pitch:           " << prop.memPitch;
        LOG(INFO) << "Maximum thread per block:       " << prop.maxThreadsPerBlock;
        LOG(INFO) << "Maximum dimension of block:     " << prop.maxThreadsDim[0]
                  << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2];
        LOG(INFO) << "Maximum dimension of grid:      " << prop.maxGridSize[0] 
                  << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2];
        LOG(INFO) << "Clock rate:                     " << prop.clockRate;
        LOG(INFO) << "Total constant memory:          " << prop.totalConstMem;
        LOG(INFO) << "Texture alignment:              " << prop.textureAlignment;
        LOG(INFO) << "Concurrent copy and execution:  "
            << (prop.deviceOverlap ? "Yes" : "No"); 
        LOG(INFO) << "Number of multiprocessors:      " << prop.multiProcessorCount;
        LOG(INFO) << "Kernel execution timeout:       "
            << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
        return;
    }

    bool Caffe::CheckDevice(const int device_id){
        // This function checks the availability of GPU #device_id.
        // It attempts to create a context on the device by calling cudaFree(0).
        // cudaSetDevice() alone is not sufficient to check the availability.
        // It lazily records device_id, however, does not initialize a
        // context. So it does not know if the host thread has the permission to use
        // the device or not.
        //
        // In a shared environment where the devices are set to EXCLUSIVE_PROCESS
        // or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
        // even if the device is exclusively occupied by another process or thread.
        // Cuda operations that initialize the context are needed to check
        // the permission. cudaFree(0) is one of those with no side effect,
        // 也就是说，cudasetDevice不能检查设备的是否可用，所以用一个cudaFree(0)这个函数代替进行检测
        bool r = ((cudaSuccess == cudaSetDevice(device_id)) && 
                (cudaSuccess == cudaFree(0)));
        
        // reset any error that may have occurred
        // Returns the last error that has been produced by any of the runtime calls
        // in the same host thread and resets it to ::cudaSuccess.
        cudaGetLastError();
        return r;
    }

    int Caffe::FindDevice(const int start_id){
        // This function finds the first available device by checking devices with
        // ordinal from start_id to the highest available value. In the
        // EXCLUSIVE_PROCESS or EXCLUSIVE_THREAD mode, if it succeeds, it also
        // claims the device due to the initialization of the context.
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        for (int i= start_id; i< count; i++){
            if (CheckDevice(i)) return i;
        }
        return -1;
    }

    class Caffe::RNG::Generator {
        public:
            Generator(): rng_(new caffe::rng_t(cluster_seedgen())) {}
            explicit Generator(unsigned int seed):rng_(new caffe::rng_t(seed)){ }
            caffe::rng_t* rng() { return rng_.get();}
        private:
            shared_ptr<caffe::rng_t> rng_;
    };

    Caffe::RNG::RNG():generator_(new Generator()) { }
    Caffe::RNG::RNG(unsigned int seed):generator_(new Generator(seed)) { }
    Caffe::RNG& Caffe::RNG::operator=(const RNG& other){
        generator_.reset(other.generator_.get());
        return *this;
    }

    void * Caffe::RNG::generator() {
        return static_cast<void*>(generator_->rng());
    }

    const char* cublasGetErrorString(cublasStatus_t error) {
        switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        #if CUDA_VERSION >= 6000
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        #endif
        #if CUDA_VERSION >= 6050
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        #endif
        }
        return "Unknown cublas status";
    }

    const char* curandGetErrorString(curandStatus_t error) {
        switch (error) {
        case CURAND_STATUS_SUCCESS:
            return "CURAND_STATUS_SUCCESS";
        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";
        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";
        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
        }
        return "Unknown curand status";
    }



#endif // CPU_ONLY

} // namespace caffe

