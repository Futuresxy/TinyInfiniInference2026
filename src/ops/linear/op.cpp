#include "op.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    
    ASSERT(weight->isContiguous(), "Linear: weight tensor must be contiguous");
    ASSERT(in->shape().size() == 2 , "Linear: input tensor must be 2-D ");
    ASSERT(out->shape()[0] == in->shape()[0] && out->shape()[1] == weight->shape()[0], "Linear: output tensor shape is incorrect");
    ASSERT( in->shape()[1] == weight->shape()[1], "Linear: weight and input tensor shape mismatch");
    CHECK_SAME_DEVICE(out , in, weight);
    const size_t M = out->shape()[0];
    const size_t N = out->shape()[1];
    const size_t K =  in->shape()[1];
    // std::cout<< "weight info"<< std::endl;
    // weight->debug();
    // std::cout<< "indata info"<< std::endl;
    // in->debug();
    // std::cout<< "bias info"<< std::endl;
    // bias->debug();

    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), in->dtype() ,M, N ,K ,in->strides().data(),weight->strides().data());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),bias->data(), in->dtype(), M , N ,K ,in->strides().data(),weight->strides().data());
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
