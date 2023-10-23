#pragma once

//#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/nvtx/nvtx3.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/cuda_util.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/logging_assert.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/device/detail/coalescing_free_list.hpp>

#include <rmm/detail/thrust_namespace.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/optional.h>

#include <fmt/core.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

//#include "cuda_runtime.h"

namespace cudf { namespace detail { 
    class pinned_value_storage {
    public:
        pinned_value_storage() {
            cudaMallocHost(&_storage, 1024);
        }

        ~pinned_value_storage() {
            cudaFreeHost(_storage);
        }

        template <class T>
        T* get() const {
            return reinterpret_cast<T*>(_storage);
        }

        template <class T>
        void get(T& out_ref, const rmm::device_scalar<T>& ds, rmm::cuda_stream_view stream) const {
            //nvtxRangePush("pinned::get");
            //T* res = get<T>();
            //ds.value(stream, *res); // d2h pinned + sync
            out_ref = ds.value(stream);
            //out_ref = *res; // copy h2h
            //nvtxRangePop();
        }

    private:
        void* _storage;
    };

static thread_local pinned_value_storage cudf_pinned_value_storage;


} // cudf
} // detail