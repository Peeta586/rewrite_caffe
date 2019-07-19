#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    SyncedMemory::SyncedMemory(): cpu_ptr_(NULL),
        gpu_ptr(NULL),size_(0),head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false){
        
    #ifndef CPU_ONLY
    #ifdef DEBUG
        CUDA_CHECK(cudaGetDevice(&device_));
    #endif
    #endif
    }

    SyncedMemory::SyncedMemory(size_t size):
        :cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false), cpu_malloc_use_cuda_(false) {
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

    inline void SyncedMemory::to_cpu(){
        check_device();
        switch (head_)
        {
        case UNINITIALIZED:
        // 如果数据刚创建状态，则CPU上开启空间
            CaffeMallocHost(&cpu_ptr, size_, &cpu_malloc_use_cuda_);
            caffe_memset(size_, 0, cpu_ptr_);
            head_ = HEAD_AT_CPU;
            own_cpu_data_ = true;
            break;
        case HEAD_AT_GPU:
        // 如果数据在GPU上， 则设置head为SYNCED， 将GPU的数据拷到CPU指针上
        #ifndef CPU_ONLY
            if (cpu_ptr_ == NULL){
                CaffeMallocHost(&cpu_str_, size_, &cpu_malloc_use_cuda_);
                own_cpu_data_ = true;
            }
            caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
            head_ = SYNCED;
        #else
            NO_GPU;
        #endif
            break;
        default:
            break;
        }
    }


} // namespace caffe
