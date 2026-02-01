#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void argmax_(const T *vals, T *max_val, int64_t *max_idx, size_t numel) {
    max_val[0] = vals[0];
    max_idx[0] = 0;

    for (size_t i = 1; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            if (llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(max_val[0])) {
                max_val[0] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(vals[i]));
                max_idx[0] = static_cast<int64_t>(i);
            }
        } else {
            if (vals[i] > max_val[0]) {
                max_val[0] = vals[i];
                max_idx[0] = static_cast<int64_t>(i);
            }
        }
    }
} 

namespace llaisys::ops::cpu {
    
void argmax(std::byte * max_idx, std::byte * max_val, const std::byte * vals, const llaisysDataType_t type, size_t numel) {

    switch (type) {
        case LLAISYS_DTYPE_F32:
            argmax_<float>(reinterpret_cast<const float *>(vals), reinterpret_cast<float *>(max_val), reinterpret_cast<int64_t *>(max_idx), numel);
        break;
        case LLAISYS_DTYPE_BF16:
            argmax_<llaisys::bf16_t>(reinterpret_cast<const llaisys::bf16_t *>(vals), reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<int64_t *>(max_idx), numel);
        break;
        case LLAISYS_DTYPE_F16:
            argmax_<llaisys::fp16_t>(reinterpret_cast<const llaisys::fp16_t *>(vals), reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<int64_t *>(max_idx), numel);
        break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    
    
}
} // namespace llaisys::ops::cpu
