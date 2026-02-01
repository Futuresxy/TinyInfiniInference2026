#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp" // 确保包含对应的 CPU 头文件

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 1. 基础校验
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64.");
    ASSERT(in->isContiguous() && out->isContiguous() && pos_ids->isContiguous(), 
           "RoPE: all tensors must be contiguous.");

    // 2. 形状校验
    auto in_shape = in->shape();
    auto out_shape = out->shape();
    ASSERT(in_shape.size() == 3, "RoPE: input must be [seqlen, nhead, d].");
    ASSERT(in_shape == out_shape, "RoPE: input and output shape mismatch.");
    ASSERT(pos_ids->shape()[0] == in_shape[0], "RoPE: pos_ids length mismatch seqlen.");
    
    size_t seqlen = in_shape[0];
    size_t nhead  = in_shape[1];
    size_t d      = in_shape[2];
    ASSERT(d % 2 == 0, "RoPE: head dimension d must be even.");

    // 3. 分发到 CPU 实现
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), 
                         in->dtype(), seqlen, nhead, d, theta);
    }

    // 4. NVIDIA 或其他设备支持
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), 
                         in->dtype(), seqlen, nhead, d, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED(); // 暂未实现
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
