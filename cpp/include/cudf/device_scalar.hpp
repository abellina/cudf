#pragma once 
#include <rmm/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/device_uvector.hpp>

namespace cudf { 
    
#if 0
template <typename T>
using device_scalar = rmm::device_scalar<T, rmm::device_uvector<T>>;
#endif

template <typename T>
class device_scalar {
 public:
  static_assert(std::is_trivially_copyable<T>::value, 
    "Scalar type must be trivially copyable");

  using value_type = T;             ///< T, the type of the scalar element
  using size_type  = std::size_t;   ///< The type used for the size
  using reference  = T&;            ///< value_type&
  using const_reference = T const&; ///< const value_type&
  using pointer = T*;               ///< The type of the pointer returned by data()
  using const_pointer = T const*;   ///< The type of the iterator
                                    ///< returned by data() const

  ~device_scalar() {
    _stream.synchronize();
    if (_device_ptr != nullptr) {
      _device_mr.deallocate_async(_device_ptr, sizeof(value_type), _stream);
    }
    if (_pinned_ptr != nullptr) {
      _host_mr.deallocate_async(_pinned_ptr, sizeof(value_type), _stream);
    }
    _device_ptr = nullptr;
    _pinned_ptr = nullptr;
  }

  device_scalar(device_scalar&& other) noexcept
    : _device_mr{other._device_mr},
      _host_mr{other._host_mr},
      _device_ptr{other._device_ptr},
      _pinned_ptr{other._pinned_ptr}
  {
    other._device_ptr = nullptr;
    other._pinned_ptr = nullptr;
  }

  /**
   * @brief Default move assignment operator
   *
   * @return device_scalar& A reference to the assigned-to object
   */
  device_scalar& operator=(device_scalar&& other) noexcept {
    CUDF_FUNC_RANGE();
    if (&other != this) {
      _stream.synchronize();
      _device_mr.deallocate_async(_device_ptr, sizeof(value_type), _stream);
      _host_mr.deallocate_async(_pinned_ptr, sizeof(value_type), _stream);
      _device_ptr = other._device_ptr;
      _pinned_ptr = other._pinned_ptr;
      other._device_ptr = nullptr;
      other._pinned_ptr = nullptr;
    }
    return this;
  };

  /**
   * @brief Copy ctor is deleted as it doesn't allow a stream argument
   */
  device_scalar(device_scalar const&) = delete;

  /**
   * @brief Copy assignment is deleted as it doesn't allow a stream argument
   */
  device_scalar& operator=(device_scalar const&) = delete;

  /**
   * @brief Default constructor is deleted as it doesn't allow a stream argument
   */
  device_scalar() = delete;

  /**
   * @brief Construct a new uninitialized `device_scalar`.
   *
   * Does not synchronize the stream.
   *
   * @note This device_scalar is only safe to access in kernels and copies on the specified CUDA
   * stream, or on another stream only if a dependency is enforced (e.g. using
   * `cudaStreamWaitEvent()`).
   *
   * @throws rmm::bad_alloc if allocating the device memory fails.
   *
   * @param stream Stream on which to perform asynchronous allocation.
   * @param mr Optional, resource with which to allocate.
   */
  explicit device_scalar(rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref device_mr = mr::get_current_device_resource(),
                         rmm::host_async_resource_ref host_mr = cudf::get_pinned_memory_resource())
    : _device_mr(device_mr),
      _host_mr(host_mr),
      _device_ptr(nullptr),
      _pinned_ptr(nullptr),
      _stream(stream)
  {
    init_storage();
  }

  /**
   * @brief Construct a new `device_scalar` with an initial value.
   *
   * Does not synchronize the stream.
   *
   * @note This device_scalar is only safe to access in kernels and copies on the specified CUDA
   * stream, or on another stream only if a dependency is enforced (e.g. using
   * `cudaStreamWaitEvent()`).
   *
   * @throws rmm::bad_alloc if allocating the device memory for `initial_value` fails.
   * @throws rmm::cuda_error if copying `initial_value` to device memory fails.
   *
   * @param initial_value The initial value of the object in device memory.
   * @param stream Optional, stream on which to perform allocation and copy.
   * @param mr Optional, resource with which to allocate.
   */
  explicit device_scalar(value_type const& initial_value,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref device_mr = mr::get_current_device_resource(),
                         rmm::host_async_resource_ref host_mr = cudf::get_pinned_memory_resource()) 
    : _device_mr(device_mr),
      _host_mr(host_mr),
      _device_ptr(nullptr),
      _pinned_ptr(nullptr),
      _stream(stream)
  {
    init_storage();
    set_value_async(initial_value, stream);
  }

  /**
   * @brief Construct a new `device_scalar` by deep copying the contents of
   * another `device_scalar`, using the specified stream and memory
   * resource.
   *
   * @throws rmm::bad_alloc If creating the new allocation fails.
   * @throws rmm::cuda_error if copying from `other` fails.
   *
   * @param other The `device_scalar` whose contents will be copied
   * @param stream The stream to use for the allocation and copy
   * @param mr The resource to use for allocating the new `device_scalar`
   */
  device_scalar(device_scalar const& other,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref device_mr = mr::get_current_device_resource(),
                rmm::host_async_resource_ref host_mr = cudf::get_pinned_memory_resource()) 
    : _device_mr(device_mr),
      _host_mr(host_mr),
      _device_ptr(nullptr),
      _pinned_ptr(nullptr),
      _stream(stream)
  {
    init_storage();
    const_pointer device_ptr = other.data();
    cudaMemcpyAsync(
        data(), 
        device_ptr, 
        sizeof(value_type),
        cudaMemcpyDefault, 
        stream.value());
  }

  void init_storage() {
    _device_ptr = reinterpret_cast<value_type*>(
      _device_mr.allocate_async(sizeof(value_type), _stream));
    _pinned_ptr = reinterpret_cast<value_type*>(
      _host_mr.allocate_async(sizeof(value_type), _stream));
  }

  /**
   * @brief Copies the value from device to host, synchronizes, and returns the value.
   *
   * Synchronizes `stream` after copying the data from device to host.
   *
   * @note If the stream specified to this function is different from the stream specified
   * to the constructor, then an appropriate dependency must be inserted between the streams
   * (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`) before calling this function,
   * otherwise there may be a race condition.
   *
   * @throws rmm::cuda_error If the copy fails.
   * @throws rmm::cuda_error If synchronizing `stream` fails.
   *
   * @return T The value of the scalar.
   * @param stream CUDA stream on which to perform the copy and synchronize.
   */
  [[nodiscard]] value_type value(rmm::cuda_stream_view stream) const
  {
    if (sizeof(value_type) == 0) {
      return value_type{};
    }
    const_pointer d_data = data();
    cuda_memcpy_async(
        _pinned_ptr,
        d_data,
        sizeof(value_type),
        cudf::detail::host_memory_kind::PINNED,
        stream);

    stream.synchronize();

    return *_pinned_ptr;
  }

  /**
   * @brief Sets the value of the `device_scalar` to the value of `v`.
   *
   * This specialization for fundamental types is optimized to use `cudaMemsetAsync` when
   * `v` is zero.
   *
   * @note If the stream specified to this function is different from the stream specified
   * to the constructor, then appropriate dependencies must be inserted between the streams
   * (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`) before and after calling
   * this function, otherwise there may be a race condition.
   *
   * This function does not synchronize `stream` before returning. Therefore, the object
   * referenced by `v` should not be destroyed or modified until `stream` has been
   * synchronized. Otherwise, behavior is undefined.
   *
   * @note: This function incurs a host to device memcpy or device memset and should be used
   * carefully.
   *
   * Example:
   * \code{cpp}
   * rmm::device_scalar<int32_t> s;
   *
   * int v{42};
   *
   * // Copies 42 to device storage on `stream`. Does _not_ synchronize
   * vec.set_value_async(v, stream);
   * ...
   * cudaStreamSynchronize(stream);
   * // Synchronization is required before `v` can be modified
   * v = 13;
   * \endcode
   *
   * @throws rmm::cuda_error if copying @p value to device memory fails.
   *
   * @param value The host value which will be copied to device
   * @param stream CUDA stream on which to perform the copy
   */
  void set_value_async(value_type const& value, cuda_stream_view stream)
  {
    if constexpr (std::is_same<value_type, bool>::value) {
      RMM_CUDA_TRY(
        cudaMemsetAsync(data(), value, sizeof(value), stream.value()));
      return;
    }
    memcpy(_pinned_ptr, &value, sizeof(value));
    cuda_memcpy_async(
        data(),
        _pinned_ptr,
        sizeof(value),
        cudf::detail::host_memory_kind::PINNED,
        stream);
  }

  // Disallow passing literals to set_value to avoid race conditions where the memory holding the
  // literal can be freed before the async memcpy / memset executes.
  void set_value_async(value_type&&, cuda_stream_view) = delete;

  /**
   * @brief Sets the value of the `device_scalar` to zero on the specified stream.
   *
   * @note If the stream specified to this function is different from the stream specified
   * to the constructor, then appropriate dependencies must be inserted between the streams
   * (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`) before and after calling
   * this function, otherwise there may be a race condition.
   *
   * This function does not synchronize `stream` before returning.
   *
   * @note: This function incurs a device memset and should be used carefully.
   *
   * @param stream CUDA stream on which to perform the copy
   */
  void set_value_to_zero_async(cuda_stream_view stream)
  {
    cudaMemsetAsync(data(), 0, sizeof(value_type), stream.value());
  }

  /**
   * @brief Returns pointer to object in device memory.
   *
   * @note If the returned device pointer is used on a CUDA stream different from the stream
   * specified to the constructor, then appropriate dependencies must be inserted between the
   * streams (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`), otherwise there may
   * be a race condition.
   *
   * @return Pointer to underlying device memory
   */
  [[nodiscard]] pointer data() noexcept { 
    return _device_ptr;
  }

  /**
   * @brief Returns const pointer to object in device memory.
   *
   * @note If the returned device pointer is used on a CUDA stream different from the stream
   * specified to the constructor, then appropriate dependencies must be inserted between the
   * streams (e.g. using `cudaStreamWaitEvent()` or `cudaStreamSynchronize()`), otherwise there may
   * be a race condition.
   *
   * @return Const pointer to underlying device memory
   */
  [[nodiscard]] const_pointer data() const noexcept
  {
    return static_cast<const_pointer>(_device_ptr);
  }

  /**
   * @briefreturn{The size of the scalar: always 1}
   */
  [[nodiscard]] constexpr size_type size() const noexcept { return 1; }

  /**
   * @briefreturn{Stream associated with the device memory allocation}
   */
  [[nodiscard]] rmm::cuda_stream_view stream() const noexcept { return _stream; }

  /**
   * @brief Sets the stream to be used for deallocation
   *
   * @param stream Stream to be used for deallocation
   */
  void set_stream(rmm::cuda_stream_view stream) noexcept { _stream = stream; }

 private:
  rmm::device_async_resource_ref _device_mr;
  rmm::host_async_resource_ref _host_mr;
  pointer _device_ptr;
  pointer _pinned_ptr;
  rmm::cuda_stream_view _stream;
};

}