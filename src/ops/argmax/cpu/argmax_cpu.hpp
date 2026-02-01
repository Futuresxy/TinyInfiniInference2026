#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx,  std::byte *vals, const std::byte *max_val, llaisysDataType_t type, size_t numel);
}