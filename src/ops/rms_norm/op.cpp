#include "op.hpp"
#include "cpu/rms_norm_cpu.hpp"
namespace llaisys::ops {
    //输入X 你暂时可以假设输入是一个2D连续张量 标准化沿输入张量的最后一个维度（即每一行，长度为d）执行。
    //权重W 。1D张量，与输入张量的一行长度相同 d 
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DTYPE(in->dtype(), weight->dtype());
    CHECK_SAME_DEVICE(in , out , weight);
    ASSERT(in->shape().size()== 2&&out->shape().size()== 2 ,"input and output tensor must be 2 dim");
    ASSERT(in->isContiguous()&&out->isContiguous() ,"input and output tensor must be Contiguous");
    ASSERT(weight->shape().size()== 1 ,"weight tensor must be 1 dim");
    ASSERT(in->shape()[1] == weight->shape()[0] ,"input tensor dim 1 must be same as the weight dim 0 " );
    ASSERT(out->shape()[0] == in->shape()[0] &&  out->shape()[1] == in->shape()[1]," input and output dim must be same");


    size_t dimk = in->shape()[1];
    size_t dimm = in->shape()[0];
    llaisysDataType_t type = in->dtype();

    if(out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(),dimm,dimk ,weight->strides().data(),eps,type);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(),dimm,dimk ,weight->strides().data(),eps,type);
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
