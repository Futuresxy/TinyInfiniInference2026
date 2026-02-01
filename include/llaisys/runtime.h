#ifndef LLAISYS_RUNTIME_H
#define LLAISYS_RUNTIME_H

#include "../llaisys.h"
//Runtime 是程序在运行时（而非编译时）负责管理资源、调度任务的一套系统。
/*
不同的硬件（NVIDIA GPU、AMD GPU、Intel CPU、华为昇腾 NPU）有不同的底层指令.
Runtime 提供了一套统一的接口（如“申请内存”、“拷贝数据”）。当你调用 malloc_device 时，Runtime 会根据你当前使用的是哪种设备，自动去调用 CUDA 的 cudaMalloc 或者其他平台的分配函数。
*/
__C {
    // Runtime API Functions
    // Device
    typedef int (*get_device_count_api)();  //看看你的电脑里有几张显卡。
    typedef void (*set_device_api)(int);  //切换当前正在操作哪一张卡
    typedef void (*device_synchronize_api)();  //强制 CPU 等待 GPU 完成所有任务
    // Stream
    typedef llaisysStream_t (*create_stream_api)();  //创建一个“任务队列”。
    typedef void (*destroy_stream_api)(llaisysStream_t);  //销毁任务队列，释放资源
    typedef void (*stream_synchronize_api)(llaisysStream_t);//只等待某个特定的任务队列完成，而不影响其他队列。
    // Memory
    typedef void *(*malloc_device_api)(size_t); //函数指针类型:它指向一个函数，该函数接收一个 size_t（内存大小）作为参数.返回一个 void *（通用内存地址）
    typedef void (*free_device_api)(void *);// 释放之前分配的设备内存
    typedef void *(*malloc_host_api)(size_t);// 分配主机（CPU）内存
    typedef void (*free_host_api)(void *); //释放主机内存
    // Memory copy
    typedef void (*memcpy_sync_api)(void *, const void *, size_t, llaisysMemcpyKind_t);//同步内存拷贝函数
    typedef void (*memcpy_async_api)(void *, const void *, size_t, llaisysMemcpyKind_t, llaisysStream_t);//异步内存拷贝函数

    /*
    当框架启动时，它会检测你有哪种 GPU。如果是 NVIDIA，它就把这个结构体里的 malloc_device 赋值为 CUDA 的分配函数；如果是 CPU，就赋值为普通的 malloc
    */
    struct LlaisysRuntimeAPI {  //“模拟类”或“虚函数表”
        get_device_count_api get_device_count;
        set_device_api set_device;
        device_synchronize_api device_synchronize;
        create_stream_api create_stream;
        destroy_stream_api destroy_stream;
        stream_synchronize_api stream_synchronize;
        malloc_device_api malloc_device;
        free_device_api free_device;
        malloc_host_api malloc_host;
        free_host_api free_host;
        memcpy_sync_api memcpy_sync;
        memcpy_async_api memcpy_async;
    };

    // Llaisys API for getting the runtime APIs
    //Tensor 类的代码只需要调用 runtime->malloc_device()，而不需要关心底层到底是 CUDA 还是 OpenCL。这实现了解耦
    __export const LlaisysRuntimeAPI *llaisysGetRuntimeAPI(llaisysDeviceType_t);

    // Llaisys API for switching device context
    __export void llaisysSetContextRuntime(llaisysDeviceType_t, int);
}

#endif // LLAISYS_RUNTIME_H
