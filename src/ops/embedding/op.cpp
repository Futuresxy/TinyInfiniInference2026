#include "op.hpp"
#include "cpu/embedding_cpu.hpp"
namespace llaisys::ops {
/*
从weight（2-D）中复制index（1-D）中的行到output（2-D）。index必须是Int64类型
*/
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out , index, weight);
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index tensor must be of type Int64");
    ASSERT(weight->shape().size() == 2, "Embedding: weight tensor must be 2-D");

 // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), index->numel(), weight->strides().data());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), index->numel(), weight->strides().data());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }    
}
} // namespace llaisys::ops
