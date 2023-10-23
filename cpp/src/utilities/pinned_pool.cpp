#include "cudf/detail/utilities/pinned_pool.hpp"

namespace cudf { namespace detail {

void* pool_memory_resource::allocate(std::size_t size, std::size_t alignment)
{
  if (size <= 0) { return nullptr; }
  return malloc(size);

  lock_guard lock(mtx_);

  size = rmm::detail::align_up(size, rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
  RMM_EXPECTS(size <= get_maximum_allocation_size(),
              "Maximum allocation size exceeded",
              rmm::out_of_memory);
  auto const block = get_block(size);

  return block.pointer();
}

void pool_memory_resource::deallocate(void* ptr, std::size_t size)
{
  if (size <= 0 || ptr == nullptr) { return; }
  free(ptr);
  return;

  lock_guard lock(mtx_);

  size             = rmm::detail::align_up(size, rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
  auto const block = free_block(ptr, size);

  blocks.insert(block);
}

pool_memory_resource::block_type pool_memory_resource::get_block(std::size_t size, free_list& blocks)
{
  pool_memory_resource::block_type const block = blocks.get_block(size);

  if (block.is_valid()) {
    return block;
  } else {
  }
  return pool_memory_resource::block_type{};
}

pool_memory_resource::block_type pool_memory_resource::get_block(std::size_t size)
{
  {
    pool_memory_resource::block_type const block = get_block(size, blocks);
    if (block.is_valid()) { 
      return allocate_and_insert_remainder(block, size, blocks); 
    }
  }

  // no large enough blocks available after merging, so grow the pool
  pool_memory_resource::block_type const block = expand_pool(size, blocks);

  return allocate_and_insert_remainder(block, size, blocks);
}

pool_memory_resource::block_type pool_memory_resource::allocate_and_insert_remainder(pool_memory_resource::block_type block, std::size_t size, free_list& blocks)
{
  auto const [allocated, remainder] = allocate_from_block(block, size);
  if (remainder.is_valid()) { blocks.insert(remainder); }
  return allocated;
}



pool_memory_resource::block_type pool_memory_resource::try_to_expand(std::size_t try_size, std::size_t min_size)
{
  while (try_size >= min_size) {
    auto block = block_from_upstream(try_size);
    if (block.has_value()) {
      current_pool_size_ += block.value().size();
      return block.value();
    }
    if (try_size == min_size) {
      break;  // only try `size` once
    }
    try_size = std::max(min_size, try_size / 2);
  }
  RMM_FAIL("Maximum pool size exceeded", rmm::out_of_memory);
}

void pool_memory_resource::initialize_pool(
    thrust::optional<std::size_t> initial_size,
    thrust::optional<std::size_t> maximum_size)
{
  auto const try_size = [&]() {
    return initial_size.value();
  }();

  current_pool_size_ = 0;  // try_to_expand will set this if it succeeds
  maximum_pool_size_ = maximum_size;

  if (try_size > 0) {
    auto const block = try_to_expand(try_size, try_size);
    blocks.insert(block);
  }
}

pool_memory_resource::block_type pool_memory_resource::expand_pool(std::size_t size, free_list& blocks)
{
  // Strategy: If maximum_pool_size_ is set, then grow geometrically, e.g. by halfway to the
  // limit each time. If it is not set, grow exponentially, e.g. by doubling the pool size each
  // time. Upon failure, attempt to back off exponentially, e.g. by half the attempted size,
  // until either success or the attempt is less than the requested size.
  return try_to_expand(size_to_grow(size), size);
}

std::size_t pool_memory_resource::size_to_grow(std::size_t size) const
{
  if (maximum_pool_size_.has_value()) {
    auto const unaligned_remaining = maximum_pool_size_.value() - pool_size();
    using rmm::detail::align_up;
    auto const remaining = align_up(unaligned_remaining, rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
    auto const aligned_size = align_up(size, rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
    return (aligned_size <= remaining) ? std::max(aligned_size, remaining / 2) : 0;
  }
  return std::max(size, pool_size());
};

thrust::optional<pool_memory_resource::block_type> pool_memory_resource::block_from_upstream(std::size_t size)
{
  try {
    void* ptr = get_upstream()->allocate(size);
    return thrust::optional<pool_memory_resource::block_type>{
      *upstream_blocks_.emplace(static_cast<char*>(ptr), size, true).first};
  } catch (std::exception const& e) {
    return thrust::nullopt;
  }
}

pool_memory_resource::split_block pool_memory_resource::allocate_from_block(pool_memory_resource::block_type const& block, std::size_t size)
{
  block_type const alloc{block.pointer(), size, block.is_head()};

  auto rest = (block.size() > size)
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                ? block_type{block.pointer() + size, block.size() - size, false}
                : block_type{};
  return {alloc, rest};
}

pool_memory_resource::block_type pool_memory_resource::free_block(void* ptr, std::size_t size) noexcept
{
  auto const iter = upstream_blocks_.find(static_cast<char*>(ptr));
  return block_type{static_cast<char*>(ptr), size, (iter != upstream_blocks_.end())};
}

void pool_memory_resource::release()
{
  lock_guard lock(mtx_);

  for (auto block : upstream_blocks_) {
    get_upstream()->deallocate(block.pointer(), block.size());
  }
  upstream_blocks_.clear();
  current_pool_size_ = 0;
}

std::pair<std::size_t, std::size_t> pool_memory_resource::free_list_summary(free_list const& blocks)
{
  std::size_t largest{};
  std::size_t total{};
  std::for_each(blocks.cbegin(), blocks.cend(), [&largest, &total](auto const& block) {
    total += block.size();
    largest = std::max(largest, block.size());
  });
  return {largest, total};
}

static_pinned_pool* static_pinned_pool::pinned_pool_ = nullptr;
std::mutex static_pinned_pool::mtx_ = std::mutex{};

}}