#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias ,llaisysDataType_t type, const size_t M,const size_t N, const size_t K, const long int * in_stride , const long int * weight_stride );
}