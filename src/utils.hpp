#pragma once
#include "utils/check.hpp"
#include "utils/types.hpp"
#ifdef _WIN32
    using stride_t = long long;
#else
    using stride_t = long;
#endif