#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <algorithm>

namespace llaisys::ops::cpu {
template <typename T>
void swiglu_(T *out , const T* gate ,const T* up , size_t slen ,size_t inter_size ){
    float temp_swiglu = 0.0;
    for (size_t i = 0; i < slen; i++)
    {
        for (size_t j = 0; j < inter_size ;j++)
        {
            size_t index =  i*inter_size+j;
            float gatenorm = 1.0 /(1+std::exp(-1.0 *llaisys::utils::cast<float>(gate[index])));
            temp_swiglu = llaisys::utils::cast<float>(up[index]) * llaisys::utils::cast<float>(gate[index]) * gatenorm;
            out[index] = llaisys::utils::cast<T>(temp_swiglu);
        }
        
    }
}

void swiglu(std::byte *out ,std::byte *gate , std::byte *up ,llaisysDataType_t type , size_t slen ,size_t inter_size){

// 根据数据类型分发模板
    switch (type) {
        case LLAISYS_DTYPE_F32:
            swiglu_<float>((float*)out, (const float*)gate, (const float*)up,  slen ,inter_size);
            break;
        case LLAISYS_DTYPE_BF16:
            swiglu_<llaisys::bf16_t>((llaisys::bf16_t*)out, (const llaisys::bf16_t*)gate, 
                                            (const llaisys::bf16_t*)up,  slen ,inter_size);
            break;
        case LLAISYS_DTYPE_F16:
            swiglu_<llaisys::fp16_t>((llaisys::fp16_t*)out, (const llaisys::fp16_t*)gate, 
                                            (const llaisys::fp16_t*)up, slen ,inter_size);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }



}


}