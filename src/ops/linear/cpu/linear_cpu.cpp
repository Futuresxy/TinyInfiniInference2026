#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, llaisysDataType_t type, 
             const size_t M, const size_t N, const size_t K, 
             const int64_t *in_stride, const int64_t *weight_stride) {

    for (size_t row = 0; row < M; row++) {
        // in_src 只与 row 有关
        const T *in_src = in + row * static_cast<size_t>(in_stride[0]); 
        for (size_t col = 0; col < N; col++) {
            // weight_src 指向第 col 行
            const T *weight_src = weight + col * static_cast<size_t>(weight_stride[0]);
            
            float Mulsum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    Mulsum += (llaisys::utils::cast<float>(in_src[k]) * llaisys::utils::cast<float>(weight_src[k]));
                } else {
                    Mulsum += static_cast<float>(in_src[k] * weight_src[k]);
                }
            }

            // 处理偏置和赋值
            size_t out_indx = row * N + col;
            float b_val = (bias == nullptr) ? 0.0f : llaisys::utils::cast<float>(bias[col]); // 注意这里是 bias[col]
            
            out[out_indx] = llaisys::utils::cast<T>(Mulsum + b_val);
        }
    }
}

namespace llaisys::ops::cpu {

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type,
            const size_t M, const size_t N, const size_t K, const int64_t *in_stride, const int64_t *weight_stride) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias), type, M, N, K, in_stride, weight_stride);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_<llaisys::bf16_t>(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(in), reinterpret_cast<const bf16_t *>(weight),
                                 reinterpret_cast<const bf16_t *>(bias), type, M, N, K, in_stride, weight_stride);
        break;
    case LLAISYS_DTYPE_F16:
        linear_<llaisys::fp16_t>(reinterpret_cast<fp16_t *>(out), reinterpret_cast<const fp16_t *>(in), reinterpret_cast<const fp16_t *>(weight),
                                 reinterpret_cast<const fp16_t *>(bias), type, M, N, K, in_stride, weight_stride);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
