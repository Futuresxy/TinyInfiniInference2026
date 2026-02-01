#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, const size_t M, const size_t K, const int64_t* stride_W, float eps) {
    
    for (size_t row = 0; row < M; row++) {
        const T *in_src = in + row * K; 
        T *out_src = out + row * K; 

        // 第一遍：计算平方和
        double sum_sq = 0.0;
        for (size_t k = 0; k < K; k++) {
            float val = llaisys::utils::cast<float>(in_src[k]);
            sum_sq += static_cast<double>(val * val);
        }

        // 计算 RMS 因子
        float inv_rms = static_cast<float>(1.0 / std::sqrt(sum_sq / K + static_cast<double>(eps)));

        // 第二遍：计算归一化并乘上 weight
        for (size_t k = 0; k < K; k++) {
            float val = llaisys::utils::cast<float>(in_src[k]);
            float w = llaisys::utils::cast<float>(weight[k * stride_W[0]]);
        
            out_src[k] = llaisys::utils::cast<T>(val * inv_rms * w);
        }
    }
}

namespace llaisys::ops::cpu {

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, const size_t dimM,const size_t dimk, const int64_t* stride_W, float eps,llaisysDataType_t type) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight),
                       dimM, dimk, stride_W, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_<llaisys::bf16_t>(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(in), reinterpret_cast<const bf16_t *>(weight),
                                 dimM, dimk, stride_W, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_<llaisys::fp16_t>(reinterpret_cast<fp16_t *>(out), reinterpret_cast<const fp16_t *>(in), reinterpret_cast<const fp16_t *>(weight),
                                 dimM, dimk, stride_W, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
