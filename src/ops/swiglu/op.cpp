#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"
namespace llaisys::ops {
    //out、up和gate是具有相同形状 [seqlen, intermediate_size] 的2D连续张量。
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out , gate , up);
    ASSERT(out->shape()[0] == up->shape()[0]&&out->shape()[0] == gate->shape()[0] ,"out in gate tensor must be the same shape dim0");
    ASSERT(out->shape()[1] == up->shape()[1]&&out->shape()[1] == gate->shape()[1] ,"out in gate tensor must be the same shape dim1");
    CHECK_SAME_DTYPE(out->dtype(),gate->dtype(),up->dtype());
    llaisysDataType_t type = out->dtype();

    if(out->deviceType() == LLAISYS_DEVICE_CPU){
        return cpu::swiglu(out->data(),gate->data(),up->data(),type,gate->shape()[0],gate->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(),out->deviceId());
    switch (out->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::swiglu(out->data(),gate->data(),up->data(),type,gate->shape()[0],gate->shape()[1]);
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
