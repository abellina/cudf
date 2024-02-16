/*
 *  Copyright 2024 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <cudf/utilities/resource_ref.hpp>

#include <cstddef>
#include <limits>
#include <new>  // for bad_alloc

#include <cudf/utilities/error.hpp>
#include <thrust/host_vector.h>

namespace cudf::detail {

/*! \p rmm_host_allocator is a CUDA-specific host memory allocator
 *  that employs \c a `cudf::host_resource_ref` for allocation.
 *
 * This implementation is ported from pinned_host_vector in cudf. 
 *
 *  \see https://en.cppreference.com/w/cpp/memory/allocator
 */
template <typename T>
class rmm_host_allocator;

/*! \p rmm_host_allocator is a CUDA-specific host memory allocator
 *  that employs \c an `cudf::host_resource_ref` for allocation.
 *
 * This implementation is ported from pinned_host_vector in cudf. 
 *
 *  \see https://en.cppreference.com/w/cpp/memory/allocator
 */
template <>
class rmm_host_allocator<void> {
 public:
  using value_type      = void;            ///< The type of the elements in the allocator
  using pointer         = void*;           ///< The type returned by address() / allocate()
  using const_pointer   = void const*;     ///< The type returned by address()
  using size_type       = std::size_t;     ///< The type used for the size of the allocation
  using difference_type = std::ptrdiff_t;  ///< The type of the distance between two pointers

  /**
   * @brief converts a `rmm_host_allocator<void>` to `rmm_host_allocator<U>`
   */
  template <typename U>
  struct rebind {
    using other = rmm_host_allocator<U>;  ///< The rebound type
  };
};

/*! \p rmm_host_allocator is a CUDA-specific host memory allocator
 *  that employs \c `rmm::mr::host_memory_resource` for allocation.
 *
 * This implementation is ported from pinned_host_vector in cudf. 
 *
 *  \see https://en.cppreference.com/w/cpp/memory/allocator
 */
template <typename T>
class rmm_host_allocator {
 public:
  using value_type      = T;               ///< The type of the elements in the allocator
  using pointer         = T*;              ///< The type returned by address() / allocate()
  using const_pointer   = T const*;        ///< The type returned by address()
  using reference       = T&;              ///< The parameter type for address()
  using const_reference = T const&;        ///< The parameter type for address()
  using size_type       = std::size_t;     ///< The type used for the size of the allocation
  using difference_type = std::ptrdiff_t;  ///< The type of the distance between two pointers

  typedef cuda::std::true_type propagate_on_container_move_assignment;

  /**
   * @brief converts a `rmm_host_allocator<T>` to `rmm_host_allocator<U>`
   */
  template <typename U>
  struct rebind {
    using other = rmm_host_allocator<U>;  ///< The rebound type
  };

  /**
   * @brief Cannot declare an empty host allocator.
   */
  rmm_host_allocator() = delete;

  /**
   * @brief Construct from a `cudf::host_resource_ref`
   */
  rmm_host_allocator(cudf::host_resource_ref _mr) : mr(_mr){}

  /**
   * @brief Copy constructor
   */
  rmm_host_allocator(rmm_host_allocator const& other) : mr(other.mr){}

  /**
   * @brief Move constructor
   */
  rmm_host_allocator(rmm_host_allocator&& other) : mr(std::move(other.mr)){}
  
  /**
   * @brief Move operator
   */
  /*
  rmm_host_allocator& operator=(rmm_host_allocator&& col)
  {
    mr = std::move(col.mr);
    return *this;
  }
  */
  
  /**
   * @brief Assignment operator
   */
  rmm_host_allocator& operator=(rmm_host_allocator const& col)
  {
    mr = col.mr;
    return *this;
  }  

  /**
   * @brief rmm_host_allocator's null destructor does nothing.
   */
  __host__ __device__ inline ~rmm_host_allocator() {}

  /**
   * @brief This method returns the address of a \c reference of
   *  interest.
   *
   *  @param r The \c reference of interest.
   *  @return \c r's address.
   */
  __host__ __device__ inline pointer address(reference r) { return &r; }

  /**
   * @brief This method returns the address of a \c const_reference
   *  of interest.
   *
   *  @param r The \c const_reference of interest.
   *  @return \c r's address.
   */
  __host__ __device__ inline const_pointer address(const_reference r) { return &r; }

  /**
   * @brief This method allocates storage for objects in host memory.
   *
   *  @param cnt The number of objects to allocate.
   *  @return a \c pointer to the newly allocated objects.
   *  @note The second parameter to this function is meant as a
   *        hint pointer to a nearby memory location, but is
   *        not used by this allocator.
   *  @note This method does not invoke \p value_type's constructor.
   *        It is the responsibility of the caller to initialize the
   *        objects at the returned \c pointer.
   */
  __host__ inline pointer allocate(size_type cnt, const_pointer /*hint*/ = 0)
  {
    if (cnt > this->max_size()) { throw std::bad_alloc(); }  // end if
    return static_cast<pointer>(mr.allocate(cnt * sizeof(value_type)));
  }

  /**
   * @brief This method deallocates host memory previously allocated
   *  with this \c rmm_host_allocator.
   *
   *  @param p A \c pointer to the previously allocated memory.
   *  @note The second parameter is the number of objects previously allocated.
   *  @note This method does not invoke \p value_type's destructor.
   *        It is the responsibility of the caller to destroy
   *        the objects stored at \p p.
   */
  __host__ inline void deallocate(pointer p, size_type cnt)
  {
    mr.deallocate(p, cnt * sizeof(value_type));
  }

  /**
   * @brief This method returns the maximum size of the \c cnt parameter
   *  accepted by the \p allocate() method.
   *
   *  @return The maximum number of objects that may be allocated
   *          by a single call to \p allocate().
   */
  inline size_type max_size() const { return (std::numeric_limits<size_type>::max)() / sizeof(T); }

  /**
   * @brief This method tests this \p rmm_host_allocator for equality to
   *  another.
   *
   *  @param x The other \p rmm_host_allocator of interest.
   *  @return This method always returns \c true.
   */
  __host__ __device__ inline bool operator==(rmm_host_allocator const& x) const
  {
    return x.mr == mr;
  }

  /**
   * @brief This method tests this \p rmm_host_allocator for inequality
   *  to another.
   *
   *  @param x The other \p rmm_host_allocator of interest.
   *  @return This method always returns \c false.
   */
  __host__ __device__ inline bool operator!=(rmm_host_allocator const& x) const
  {
    return !operator==(x);
  }
private:
  cudf::host_resource_ref mr;
};

/**
 * @brief A vector class with rmm host memory allocator
 */
template <typename T>
using rmm_host_vector = thrust::host_vector<T, rmm_host_allocator<T>>;

}  // namespace cudf::detail
