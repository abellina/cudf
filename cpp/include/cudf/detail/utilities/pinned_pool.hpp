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

template <typename T>
struct crtp {
  [[nodiscard]] T& underlying() { return static_cast<T&>(*this); }
  [[nodiscard]] T const& underlying() const { return static_cast<T const&>(*this); }
};


/**
 * @brief A coalescing best-fit suballocator which uses a pool of memory allocated from
 *        an upstream memory_resource.
 *
 * Allocation (do_allocate()) and deallocation (do_deallocate()) are thread-safe. Also,
 * this class is compatible with CUDA per-thread default stream.
 *
 * @tparam UpstreamResource memory_resource to use for allocating the pool. Implements
 *                          rmm::mr::device_memory_resource interface.
 */
class pool_memory_resource : public crtp<rmm::mr::pinned_memory_resource> {
 public:
  using free_list  = rmm::mr::detail::coalescing_free_list;  ///< The free list implementation
  using block_type = free_list::block_type;         ///< The type of block returned by the free list
  using lock_guard = std::lock_guard<std::mutex>;  ///< Type of lock used to synchronize access
  using split_block = std::pair<block_type, block_type>;

  free_list blocks;
  std::mutex mtx_;  // mutex for thread-safe access

  /**
   * @brief Construct a `pool_memory_resource` and allocate the initial device memory pool using
   * `upstream_mr`.
   *
   * @throws rmm::logic_error if `upstream_mr == nullptr`
   * @throws rmm::logic_error if `initial_pool_size` is neither the default nor aligned to a
   * multiple of pool_memory_resource::allocation_alignment bytes.
   * @throws rmm::logic_error if `maximum_pool_size` is neither the default nor aligned to a
   * multiple of pool_memory_resource::allocation_alignment bytes.
   *
   * @param upstream_mr The memory_resource from which to allocate blocks for the pool.
   * @param initial_pool_size Minimum size, in bytes, of the initial pool. Defaults to half of the
   * available memory on the current device.
   * @param maximum_pool_size Maximum size, in bytes, that the pool can grow to. Defaults to all
   * of the available memory on the current device.
   */
  explicit pool_memory_resource(rmm::mr::pinned_memory_resource* upstream_mr,
                                thrust::optional<std::size_t> initial_pool_size = thrust::nullopt,
                                thrust::optional<std::size_t> maximum_pool_size = thrust::nullopt)
    : upstream_mr_{[upstream_mr]() {
        RMM_EXPECTS(nullptr != upstream_mr, "Unexpected null upstream pointer.");
        return upstream_mr;
      }()}
  {
    RMM_EXPECTS(rmm::detail::is_aligned(initial_pool_size.value_or(0),
                                        rmm::detail::CUDA_ALLOCATION_ALIGNMENT),
                "Error, Initial pool size required to be a multiple of 256 bytes");
    RMM_EXPECTS(rmm::detail::is_aligned(maximum_pool_size.value_or(0),
                                        rmm::detail::CUDA_ALLOCATION_ALIGNMENT),
                "Error, Maximum pool size required to be a multiple of 256 bytes");

    initialize_pool(initial_pool_size, maximum_pool_size);
  }

  /**
   * @brief Destroy the `pool_memory_resource` and deallocate all memory it allocated using
   * the upstream resource.
   */
  ~pool_memory_resource() { release(); }

  pool_memory_resource()                                       = delete;
  pool_memory_resource(pool_memory_resource const&)            = delete;
  pool_memory_resource(pool_memory_resource&&)                 = delete;
  pool_memory_resource& operator=(pool_memory_resource const&) = delete;
  pool_memory_resource& operator=(pool_memory_resource&&)      = delete;

  void* allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t));

  void deallocate(void* ptr, std::size_t size);

  block_type get_block(std::size_t size, free_list& blocks);

  block_type get_block(std::size_t size);

  block_type allocate_and_insert_remainder(block_type block, std::size_t size, free_list& blocks);

  /**
   * @brief Get the upstream memory_resource object.
   *
   * @return UpstreamResource* the upstream memory resource.
   */
  rmm::mr::pinned_memory_resource* get_upstream() const noexcept { return upstream_mr_; }

  /**
   * @brief Computes the size of the current pool
   *
   * Includes allocated as well as free memory.
   *
   * @return std::size_t The total size of the currently allocated pool.
   */
  [[nodiscard]] std::size_t pool_size() const noexcept { return current_pool_size_; }


  /**
   * @brief Get the maximum size of allocations supported by this memory resource
   *
   * Note this does not depend on the memory size of the device. It simply returns the maximum
   * value of `std::size_t`
   *
   * @return std::size_t The maximum size of a single allocation supported by this memory resource
   */
  [[nodiscard]] std::size_t get_maximum_allocation_size() const
  {
    return std::numeric_limits<std::size_t>::max();
  }

  block_type try_to_expand(std::size_t try_size, std::size_t min_size);

  /**
   * @brief Allocate initial memory for the pool
   *
   * If initial_size is unset, then queries the upstream memory resource for available memory if
   * upstream supports `get_mem_info`, or queries the device (using CUDA API) for available memory
   * if not. Then attempts to initialize to half the available memory.
   *
   * If initial_size is set, then tries to initialize the pool to that size.
   *
   * @param initial_size The optional initial size for the pool
   * @param maximum_size The optional maximum size for the pool
   */
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  void initialize_pool(thrust::optional<std::size_t> initial_size,
                       thrust::optional<std::size_t> maximum_size);

  /**
   * @brief Allocate space from upstream to supply the suballocation pool and return
   * a sufficiently sized block.
   *
   * @param size The minimum size to allocate
   * @param blocks The free list (ignored in this implementation)
   * @param stream The stream on which the memory is to be used.
   * @return block_type a block of at least `size` bytes
   */
  block_type expand_pool(std::size_t size, free_list& blocks);

  /**
   * @brief Given a minimum size, computes an appropriate size to grow the pool.
   *
   * Strategy is to try to grow the pool by half the difference between the configured maximum
   * pool size and the current pool size, if the maximum pool size is set. If it is not set, try
   * to double the current pool size.
   *
   * Returns 0 if the requested size cannot be satisfied.
   *
   * @param size The size of the minimum allocation immediately needed
   * @return std::size_t The computed size to grow the pool.
   */
  [[nodiscard]] std::size_t size_to_grow(std::size_t size) const;

  /**
   * @brief Allocate a block from upstream to expand the suballocation pool.
   *
   * @param size The size in bytes to allocate from the upstream resource
   * @param stream The stream on which the memory is to be used.
   * @return block_type The allocated block
   */
  thrust::optional<block_type> block_from_upstream(std::size_t size);

  /**
   * @brief Splits `block` if necessary to return a pointer to memory of `size` bytes.
   *
   * If the block is split, the remainder is returned to the pool.
   *
   * @param block The block to allocate from.
   * @param size The size in bytes of the requested allocation.
   * @return A pair comprising the allocated pointer and any unallocated remainder of the input
   * block.
   */
  split_block allocate_from_block(block_type const& block, std::size_t size);

  /**
   * @brief Finds, frees and returns the block associated with pointer `ptr`.
   *
   * @param ptr The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @return The (now freed) block associated with `p`. The caller is expected to return the block
   * to the pool.
   */
  block_type free_block(void* ptr, std::size_t size) noexcept;

  /**
   * @brief Free all memory allocated from the upstream memory_resource.
   *
   */
  void release();

  /**
   * @brief Get the largest available block size and total free size in the specified free list
   *
   * This is intended only for debugging
   *
   * @param blocks The free list from which to return the summary
   * @return std::pair<std::size_t, std::size_t> Pair of largest available block, total free size
   */
  std::pair<std::size_t, std::size_t> free_list_summary(free_list const& blocks);

 private:
  rmm::mr::pinned_memory_resource* upstream_mr_;  // The "heap" to allocate the pool from
  std::size_t current_pool_size_{};
  thrust::optional<std::size_t> maximum_pool_size_{};

  // blocks allocated from upstream
  std::set<block_type, rmm::mr::detail::compare_blocks<block_type>> upstream_blocks_;
};

using pinned_pool_t = pool_memory_resource;
using pinned_resource_t = rmm::mr::pinned_memory_resource;

class static_pinned_pool {
  public:
    static_pinned_pool()
      : pinned_(new pinned_resource_t()),
        pool_(new pinned_pool_t(pinned_, 16L*1024*1024*1024, 16L*1024*1024*1024))
    {
    }

    ~static_pinned_pool() {
      delete pool_;
      delete pinned_;
    }

    void allocate(void** ptr, std::size_t sz) {
      *ptr = pool_->allocate(sz);
    }

    void free(void* ptr, std::size_t sz) {
      pool_->deallocate(ptr, sz);
    }

    static static_pinned_pool* get_instance() { 
      if (pinned_pool_ == nullptr) { 
        std::lock_guard<std::mutex> lock(mtx_);
        if (pinned_pool_ == nullptr) {
          pinned_pool_ = new static_pinned_pool();
        }
      } 
      return pinned_pool_;
    }
    static static_pinned_pool* pinned_pool_;

  private:
    pinned_resource_t* pinned_;
    pinned_pool_t* pool_;
    static std::mutex mtx_;  // mutex for thread-safe access
};



} // cudf
} // detail