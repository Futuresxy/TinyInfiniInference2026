#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, const size_t dimM,const size_t dimk, const long int* stride_W, float eps ,llaisysDataType_t type);
}