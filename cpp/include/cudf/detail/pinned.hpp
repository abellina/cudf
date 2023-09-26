#pragma once

//#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/nvtx/nvtx3.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/detail/stack_trace.hpp>
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
            out_ref = ds.value(stream);
            //rmm::detail::stack_trace st;
            //nvtxRangePush("pinned::get");
            //T* res = get<T>();
            //ds.value(stream, *res); // d2h pinned + sync
            //out_ref = *res; // copy h2h
            //nvtxRangePop();
            //std::cout << st << std::endl;
        }

    private:
        void* _storage;
    };

static thread_local pinned_value_storage cudf_pinned_value_storage;

} // cudf
} // detail