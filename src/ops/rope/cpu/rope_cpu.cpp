#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, 
           size_t seqlen, size_t nhead, size_t d, float theta) {
    
    const size_t half_d = d / 2;
    
    // 优化 1：将频率计算提前到 head 循环之外
    // 这样每一行（每个 token）只需要计算一次频率向量
    std::vector<double> inv_freqs(half_d);
    for (size_t j = 0; j < half_d; ++j) {
        inv_freqs[j] = 1.0 / std::pow((double)theta, (double)(2 * j) / (double)d);
    }

    for (size_t i = 0; i < seqlen; ++i) {
        double p_i = static_cast<double>(pos_ids[i]); 
        
        for (size_t j = 0; j < half_d; ++j) {
            // 优化 2：统一计算当前维度在当前位置的角度
            double phi = p_i * inv_freqs[j];
            float cos_phi = static_cast<float>(std::cos(phi));
            float sin_phi = static_cast<float>(std::sin(phi));

            for (size_t h = 0; h < nhead; ++h) {
                // 优化 3：调整循环顺序，提高内存访问的局部性
                // 将 nhead 放在最内层（如果数据是连续的，这样可以利用 L1 Cache）
                size_t base_offset = i * nhead * d + h * d;
                size_t idx_a = base_offset + j;
                size_t idx_b = base_offset + j + half_d;

                float a = llaisys::utils::cast<float>(in[idx_a]);
                float b = llaisys::utils::cast<float>(in[idx_b]);

                out[idx_a] = llaisys::utils::cast<T>(a * cos_phi - b * sin_phi);
                out[idx_b] = llaisys::utils::cast<T>(b * cos_phi + a * sin_phi);
            }
        }
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, 
          llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d, float theta) {
    
    const int64_t *pids = reinterpret_cast<const int64_t *>(pos_ids);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), 
                            pids, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), 
                                      reinterpret_cast<const llaisys::bf16_t *>(in), 
                                      pids, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), 
                                      reinterpret_cast<const llaisys::fp16_t *>(in), 
                                      pids, seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu