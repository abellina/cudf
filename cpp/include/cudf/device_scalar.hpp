#pragma once 
#include <rmm/device_scalar.hpp>
#include <cudf/utilities/device_uvector.hpp>

namespace cudf { 

template <typename T>
using device_scalar = rmm::device_scalar<T, cudf::device_uvector<T>>;
}