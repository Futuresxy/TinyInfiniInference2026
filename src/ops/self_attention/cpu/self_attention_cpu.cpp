#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace llaisys::ops::cpu {

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                     size_t d, size_t dv, float scale) {
    
    size_t group_size = nhead / nkvhead;
    size_t past_len = total_len - seqlen;

    for (size_t i = 0; i < seqlen; ++i) {
        for (size_t h = 0; h < nhead; ++h) {
            size_t h_kv = h / group_size;
            const T *q_ptr = q + (i * nhead + h) * d;
            
            // 使用 float 存储中间分数，保证 Softmax 精度
            std::vector<float> scores(total_len);
            float max_score = -std::numeric_limits<float>::infinity();

            // --- 环节 1: QK^T 计算 (高精度累加) ---
            for (size_t j = 0; j < total_len; ++j) {
                if (j > (past_len + i)) {
                    scores[j] = -1e10f; 
                    continue;
                }
                const T *k_ptr = k + (j * nkvhead + h_kv) * d;
                
                // 使用 double 累加点积，减少 FP16 精度丢失
                double dot = 0.0;
                for (size_t dim = 0; dim < d; ++dim) {
                    dot += (double)llaisys::utils::cast<float>(q_ptr[dim]) * (double)llaisys::utils::cast<float>(k_ptr[dim]);
                }
                scores[j] = (float)(dot * (double)scale);
                if (scores[j] > max_score) max_score = scores[j];
            }

            // --- 环节 2: Softmax 计算 (Double 累加分母) ---
            double sum_exp = 0.0;
            for (size_t j = 0; j < total_len; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += (double)scores[j];
            }
            float inv_sum = (float)(1.0 / (sum_exp + 1e-12));

            // --- 环节 3: Score * V 计算 (Float32 累加输出) ---
            T *out_ptr = attn_val + (i * nhead + h) * dv;
            
            // 使用临时 float 数组存储这一行的结果，避免频繁 cast
            std::vector<float> line_buffer(dv, 0.0f);

            for (size_t j = 0; j < total_len; ++j) {
                float s = scores[j] * inv_sum;
                if (s < 1e-8f) continue; 

                const T *v_ptr = v + (j * nkvhead + h_kv) * dv;
                for (size_t dim = 0; dim < dv; ++dim) {
                    // 在 float 空间累加
                    line_buffer[dim] += s * llaisys::utils::cast<float>(v_ptr[dim]);
                }
            }

            // 最后统一写回 T 类型
            for (size_t dim = 0; dim < dv; ++dim) {
                out_ptr[dim] = llaisys::utils::cast<T>(line_buffer[dim]);
            }
        }
    }
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t seqlen, size_t total_len, size_t nhead, 
                    size_t nkvhead, size_t d, size_t dv, float scale) {
    // 根据数据类型分发模板
    switch (type) {
        case LLAISYS_DTYPE_F32:
            self_attention_<float>((float*)attn_val, (const float*)q, (const float*)k, (const float*)v, 
                                   seqlen, total_len, nhead, nkvhead, d, dv, scale);
            break;
        case LLAISYS_DTYPE_BF16:
            self_attention_<llaisys::bf16_t>((llaisys::bf16_t*)attn_val, (const llaisys::bf16_t*)q, 
                                            (const llaisys::bf16_t*)k, (const llaisys::bf16_t*)v, 
                                            seqlen, total_len, nhead, nkvhead, d, dv, scale);
            break;
        case LLAISYS_DTYPE_F16:
            self_attention_<llaisys::fp16_t>((llaisys::fp16_t*)attn_val, (const llaisys::fp16_t*)q, 
                                            (const llaisys::fp16_t*)k, (const llaisys::fp16_t*)v, 
                                            seqlen, total_len, nhead, nkvhead, d, dv, scale);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu