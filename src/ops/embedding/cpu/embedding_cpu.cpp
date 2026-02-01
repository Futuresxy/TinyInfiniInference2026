#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void embedding_( T *out, const int64_t *index, const T *weight, size_t index_numel, const long int * stride) {
   
    for (size_t i = 0; i < index_numel; i++) {
        int64_t idx = index[i];
        const T *src = weight + idx * stride[0];
        T *dst = out + i * stride[1];
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            for (size_t j = 0; j < static_cast<size_t>(stride[1]); j++) {
                dst[j] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(src[j]));
            }
        } else{
            for (size_t j = 0; j < static_cast<size_t>(stride[1]); j++) {
                dst[j] = src[j];
        }
        }
        
    }
} 
         

namespace llaisys::ops::cpu {
    
void embedding(std::byte * out, std::byte * index, const std::byte * weight, const llaisysDataType_t type, size_t index_numel, const long int * stride) {

    switch (type) {
        case LLAISYS_DTYPE_F32:
            embedding_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), index_numel, stride);
        break;
        case LLAISYS_DTYPE_BF16:
            embedding_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::bf16_t *>(weight), index_numel, stride);
        break;
        case LLAISYS_DTYPE_F16:
            embedding_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::fp16_t *>(weight), index_numel, stride);
        break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    
    
}
} // namespace llaisys::ops::cpu
