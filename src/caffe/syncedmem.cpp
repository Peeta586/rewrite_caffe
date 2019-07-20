#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    SyncedMemory::SyncedMemory(): cpu_ptr_(NULL),
        gpu_ptr_(NULL),size_(0),head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false){
        
    #ifndef CPU_ONLY
    #ifdef DEBUG
        CUDA_CHECK(cudaGetDevice(&device_));
    #endif
    #endif
    }

    SyncedMemory::SyncedMemory(size_t size):
        cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false),own_gpu_data_(false) {
    #ifndef CPU_ONLY
    #ifdef DEBUG
        CUDA_CHECK(cudaGetDevice(&device_));
    #endif
    #endif
    }

    SyncedMemory::~SyncedMemory() {
        check_device();
        if (cpu_ptr_ && own_cpu_data_){
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }

    #ifndef CPU_ONLY
        if (gpu_ptr_ && own_gpu_data_){
            CUDA_CHECK(cudaFree(gpu_ptr_));
        }
    #endif // CPU_ONLY
    }

    // 类似域pytorch的cpu()
    inline void SyncedMemory::to_cpu(){
        check_device();
        switch (head_)
        {
        case UNINITIALIZED:
        // 如果数据刚创建状态，则CPU上开启空间
        // 注意，内存的分配是按照字节来计数的，size_是字节的大小
            CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
            caffe_memset(size_, 0, cpu_ptr_);
            head_ = HEAD_AT_CPU;
            own_cpu_data_ = true;
            break;
        case HEAD_AT_GPU:
        // 如果数据在GPU上， 则设置head为SYNCED， 将GPU的数据拷到CPU指针上
        #ifndef CPU_ONLY
            if (cpu_ptr_ == NULL){
                CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
                own_cpu_data_ = true;
            }
            caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
            head_ = SYNCED;
        #else
            NO_GPU;
        #endif
            break;
        case HEAD_AT_CPU:
        case SYNCED:
            break;
        }
    }
    // 类似pytorch 的.cuda()/to(device)
    inline void SyncedMemory::to_gpu() {
        check_device();

    #ifndef CPU_ONLY
        switch (head_)
        {
        case UNINITIALIZED: // 如果还没空间
            CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
            caffe_gpu_memset(size_, 0, gpu_ptr_);
            head_ = HEAD_AT_GPU;
            own_gpu_data_ = true;
            break;
        case HEAD_AT_CPU: // 如果数据在host
            if (gpu_ptr_ == NULL){
                CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
                own_gpu_data_ = true;
            }
            caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
            head_ = SYNCED;
            break;
        case HEAD_AT_GPU:
        case SYNCED: // 如果已经将数据导到gpu了，则不操作，直接使用gpu_ptr
            break;
        }
    #else
        NO_GPU;
    #endif

    }

    const void* SyncedMemory::cpu_data(){
        check_device();
        to_cpu();
        return (const void*)cpu_ptr_;
    }

    void SyncedMemory::set_cpu_data(void* data){
        check_device();
        CHECK(data); // 是否不为空
        if (own_cpu_data_){
            //CaffeFreeHost 如果这个指针数据被移到了GPU, 则申请的时候用的是cudaMallohost（pinned memory），则对应的需要cudaFreeHost释放
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }
        cpu_ptr_ = data;
        head_ = HEAD_AT_CPU;
        own_cpu_data_ = false; // 外面的cpu数据
    }

    const void* SyncedMemory::gpu_data() {
        check_device();
    #ifndef CPU_ONLY
        to_gpu();
        return (const void*)gpu_ptr_;
    #else
        NO_GPU;
        return NULL;
    #endif
    }

    void SyncedMemory::set_gpu_data(void* data){
        check_device();
    #ifndef CPU_ONLY
        CHECK(data);
        if(own_gpu_data_){
            CUDA_CHECK(cudaFree(gpu_ptr_));
        }
        gpu_ptr_ = data;
        head_ = HEAD_AT_GPU;
        own_gpu_data_ = false; //  说明不是自己持有
    #else  
        NO_GPU;
    #endif
    }

    void* SyncedMemory::mutable_cpu_data(){
        check_device();
        to_cpu();
        // 因为数据是可变的，所以head_不设置为SYNCED， 而是设置为仅指向CPU,这样就算是gpu_ptr指向同步数据，也作为失效
        head_ = HEAD_AT_CPU; 
        return cpu_ptr_;
    }

    void* SyncedMemory::mutable_gpu_data(){
        check_device();
    #ifndef CPU_ONLY
        to_gpu();
        // 因为数据是可变的，所以head_不设置为SYNCED， 而是设置为仅指向GPU，这样就算是cpu_ptr指向同步数据，也作为失效
        head_ = HEAD_AT_GPU; 
        return gpu_ptr_;
    #else
        NO_GPU;
        return NULL;
    #endif

    }


    #ifndef CPU_ONLY
    // 异步gpu数据传输
    void SyncedMemory::async_gpu_push(const cudaStream_t& stream){
        check_device();
        CHECK(head_ == HEAD_AT_CPU);
        if(gpu_ptr_ == NULL){
            //  first param is void **, 二级指针
            CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
            own_gpu_data_ = true;
        }
        const cudaMemcpyKind put = cudaMemcpyHostToDevice;
        CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
        // assume caller will synchronize on the stream before use
        head_ = SYNCED;
    }
    #endif

    void SyncedMemory::check_device(){
    #ifndef CPU_ONLY
    #ifdef DEBUG
        int device;
        // 当前设备
        cudaGetDevice(&device);
        CHECK(device_ == device);
        // 当前指针是否在该设备上
        if (gpu_ptr_ && own_gpu_data_){
            cudaPointerAttributes attributes;
            CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
            CHECK(attributes.device == device_);
        }
    #endif
    #endif
    }


} // namespace caffe
