#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>
#ifdef USE_MKL
    #include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {
    // If CUDA is available and in GPU mode, host memory will be allocated pinned,
    // using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
    // The improvement in performance seems negligible in the single GPU case,
    // but might be more significant for parallel training. Most importantly,
    // it improved stability for large models on many GPUs.

    inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda){
    #ifndef CPU_ONLY
    /**
     * 使用分页锁定内存，分页锁定内存和显存之间的拷贝速度大约是6GB/s，普通的分页内存和GPU间的速度大约是3GB/s，
     * （另外：GPU内存间速度是30G,CPU间内存速度是10GB/s），但是这种方法会带来额外的cpu内存间的拷贝时间
     * 调用cudaMallocHost得到的pinned的内存, 不是显存
     * cudaMalloc是直接分配显存空间，
     * malloc 则是直接分配内存，但不是pinned
     * 建议针对cudaMemcpy()调用中的源内存或者目标内存,才使用页锁定内存,
     * 并且在不在使用他们的时候立即释放,而不是在应用程序关闭的时候才释放.我们使用下面的测试实例，
     * 因为如果ptr是GPU的，那么我们要分配内存的空间，那么一定是要传输到这个内存空间的，所以要分配pinned memory
     * 注意，所谓的Pinned Memory都是在Host端的，而不是Device端。
     * 用use_cuda 指示当前cpu数据是用于GPU的pinned memory（cudaMallocHost申请的）还是普通的memory(malloc申请的)；
     * 
     * 所有的其他类都使用Caffe类，因此，再开始编译的时候会设置Caffe::mode等一些参数的
     */
        if (Caffe::mode() == Caffe::GPU){
            CUDA_CHECK(cudaMallocHost(ptr, size));
            *use_cuda = true;
            return;
        }
    #endif

    #ifdef USE_MKL
        *ptr = mkl_malloc(size ? size:1, 64);
    #else
        *ptr = malloc(size);
    #endif
        *use_cuda = false;
        CHECK(*ptr) << "host allocation of size" << size << "failed";
    }

    // 如果这个指针数据被移到了GPU,则申请的时候是cudaMallocHost，则对应的需要用cudaFreeHost， 否则free
    inline void CaffeFreeHost(void* ptr, bool use_cuda){
    #ifndef CPU_ONLY
        if (use_cuda) {
            CUDA_CHECK(cudaFreeHost(ptr));
            return;
        }
    #endif

    #ifdef USE_MKL
        mkl_free(ptr);
    #else
        free(ptr);
    #endif
    }

    /**
     * @brief Manages memory allocation and synchronization between the host (CPU)
     *        and device (GPU).
     *
     * TODO(dox): more thorough description.
     */

    class SyncedMemory{
        public:
            SyncedMemory();
            explicit SyncedMemory(size_t size);
            ~SyncedMemory();

            const void* cpu_data();
            void set_cpu_data(void* data);
            const void* gpu_data();
            void set_gpu_data(void* data);

            // 可改动的
            void* mutable_cpu_data();
            void* mutable_gpu_data();

            // SYNCED 表示cpu_ptr和gpu_ptr都指向数据，一个指向host的拷贝数据，一个指向device的数据；
            // HEAD_AT_CPU, HEAD_AT_GPU 仅是一端的数据;
            // 这个枚举用来控制或者说明当前数据是同时有CPU和GPU拷贝，还是只有一个
            enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED};
            /** 为了防止类的数据成员被非法访问，将类的成员函数分成了两类，一类是常量成员函数（也被称为观察着）；
             * 另一类是非常量成员函数（也被成为变异者）.在一个函数的签名后面加上关键字const后该函数就成了常量函数。
             * 对于常量函数，最关键的不同是编译器不允许其修改类的数据成员
            */
            SyncedHead head() const { return head_;} // 不允许修改成员
            size_t size() const { return size_; }

        #ifndef CPU_ONLY
            void async_gpu_push(const cudaStream_t& stream);
        #endif

        private:
            void check_device();

            void to_cpu();
            void to_gpu();
            void * cpu_ptr_;
            void * gpu_ptr_;
            size_t size_;
            SyncedHead head_;

            /*
            表示是自己的数据，而不是持有数据的意思。 如果是自己分配的，那么是true，如果是外部传输的调用set_cpu_data设置的，
            则为false 说明数据指针只是一个指针拷贝而已；
            gpu同理
             */
            bool own_cpu_data_;
            /*
             cpu_malloc_use_cuda_ 被CaffeMallocHost设置， 如果申请cpu资源时，设定这个cpu资源是为了GPU使用的，
             那么cpu_malloc_use_cuda_=true， 只用于cpu；
             而释放资源时，如果这个为true， 那么这个指针有cpu和gpu两个持有资源需要释放
             */
            bool cpu_malloc_use_cuda_;
            bool own_gpu_data_;
            int device_;

            DISABLE_COPY_AND_ASSIGN(SyncedMemory);

    }; // class SyncedMemory

} //namespace caffe

#endif // CAFFE_SYNCEDMEM_HPP_

