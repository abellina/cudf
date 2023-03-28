/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cstddef>
#include <numeric>
#include <optional>

#include "contiguous_split_chunked.cuh"

namespace cudf {
namespace chunked {
namespace {

// align all column size allocations to this boundary so that all output column buffers
// start at that alignment.
static constexpr std::size_t split_align = 64;

/**
 * @brief Struct which contains information on a source buffer.
 *
 * The definition of "buffer" used throughout this module is a component piece of a
 * cudf column. So for example, a fixed-width column with validity would have 2 associated
 * buffers : the data itself and the validity buffer.  contiguous_split operates by breaking
 * each column up into it's individual components and copying each one as a separate kernel
 * block.
 */
struct src_buf_info {
  src_buf_info(cudf::type_id _type,
               const int* _offsets,
               int _offset_stack_pos,
               int _parent_offsets_index,
               bool _is_validity,
               size_type _column_offset)
    : type(_type),
      offsets(_offsets),
      offset_stack_pos(_offset_stack_pos),
      parent_offsets_index(_parent_offsets_index),
      is_validity(_is_validity),
      column_offset(_column_offset)
  {
  }

  cudf::type_id type;
  const int* offsets;        // a pointer to device memory offsets if I am an offset buffer
  int offset_stack_pos;      // position in the offset stack buffer
  int parent_offsets_index;  // immediate parent that has offsets, or -1 if none
  bool is_validity;          // if I am a validity buffer
  size_type column_offset;   // offset in the case of a sliced column
};

/**
 * @brief Struct which contains information on a destination buffer.
 *
 * Similar to src_buf_info, dst_buf_info contains information on a destination buffer we
 * are going to copy to.  If we have N input buffers (which come from X columns), and
 * M partitions, then we have N*M destination buffers.
 */
struct dst_buf_info {
  // constant across all copy commands for this buffer
  std::size_t buf_size;  // total size of buffer, including padding
  int num_elements;      // # of elements to be copied
  int element_size;      // size of each element in bytes
  int num_rows;  // # of rows to be copied(which may be different from num_elements in the case of
                 // validity or offset buffers)

  int src_element_index;   // element index to start reading from from my associated source buffer
  std::size_t dst_offset;  // my offset into the per-partition allocation
  int value_shift;         // amount to shift values down by (for offset buffers)
  int bit_shift;           // # of bits to shift right by (for validity buffers)
  size_type valid_count;   // validity count for this block of work

  int src_buf_index;  // source buffer index
  int dst_buf_index;  // destination buffer index
};

/**
 * @brief Copy a single buffer of column data, shifting values (for offset columns),
 * and validity (for validity buffers) as necessary.
 *
 * Copies a single partition of a source column buffer to a destination buffer. Shifts
 * element values by value_shift in the case of a buffer of offsets (value_shift will
 * only ever be > 0 in that case).  Shifts elements bitwise by bit_shift in the case of
 * a validity buffer (bif_shift will only ever be > 0 in that case).  This function assumes
 * value_shift and bit_shift will never be > 0 at the same time.
 *
 * This function expects:
 * - src may be a misaligned address
 * - dst must be an aligned address
 *
 * This function always does the ALU work related to value_shift and bit_shift because it is
 * entirely memory-bandwidth bound.
 *
 * @param dst Destination buffer
 * @param src Source buffer
 * @param t Thread index
 * @param num_elements Number of elements to copy
 * @param element_size Size of each element in bytes
 * @param src_element_index Element index to start copying at
 * @param stride Size of the kernel block
 * @param value_shift Shift incoming 4-byte offset values down by this amount
 * @param bit_shift Shift incoming data right by this many bits
 * @param num_rows Number of rows being copied
 * @param valid_count Optional pointer to a value to store count of set bits
 */
template <int block_size>
__device__ void copy_buffer(uint8_t* __restrict__ dst,
                            uint8_t const* __restrict__ src,
                            int t,
                            std::size_t num_elements,
                            std::size_t element_size,
                            std::size_t src_element_index,
                            uint32_t stride,
                            int value_shift,
                            int bit_shift,
                            std::size_t num_rows,
                            size_type* valid_count)
{
  src += (src_element_index * element_size);

  size_type thread_valid_count = 0;

  // handle misalignment. read 16 bytes in 4 byte reads. write in a single 16 byte store.
  std::size_t const num_bytes = num_elements * element_size;
  // how many bytes we're misaligned from 4-byte alignment
  uint32_t const ofs = reinterpret_cast<uintptr_t>(src) % 4;
  std::size_t pos    = t * 16;
  stride *= 16;
  while (pos + 20 <= num_bytes) {
    // read from the nearest aligned address.
    const uint32_t* in32 = reinterpret_cast<const uint32_t*>((src + pos) - ofs);
    uint4 v              = uint4{in32[0], in32[1], in32[2], in32[3]};
    if (ofs || bit_shift) {
      v.x = __funnelshift_r(v.x, v.y, ofs * 8 + bit_shift);
      v.y = __funnelshift_r(v.y, v.z, ofs * 8 + bit_shift);
      v.z = __funnelshift_r(v.z, v.w, ofs * 8 + bit_shift);
      v.w = __funnelshift_r(v.w, in32[4], ofs * 8 + bit_shift);
    }
    v.x -= value_shift;
    v.y -= value_shift;
    v.z -= value_shift;
    v.w -= value_shift;
    reinterpret_cast<uint4*>(dst)[pos / 16] = v;
    if (valid_count) {
      thread_valid_count += (__popc(v.x) + __popc(v.y) + __popc(v.z) + __popc(v.w));
    }
    pos += stride;
  }

  // copy trailing bytes
  if (t == 0) {
    std::size_t remainder;
    if (num_bytes < 16) {
      remainder = num_bytes;
    } else {
      std::size_t const last_bracket = (num_bytes / 16) * 16;
      remainder                      = num_bytes - last_bracket;
      if (remainder < 4) {
        // we had less than 20 bytes for the last possible 16 byte copy, so copy 16 + the extra
        remainder += 16;
      }
    }

    // if we're performing a value shift (offsets), or a bit shift (validity) the # of bytes and
    // alignment must be a multiple of 4. value shifting and bit shifting are mutually exclusive
    // and will never both be true at the same time.
    if (value_shift || bit_shift) {
      std::size_t idx = (num_bytes - remainder) / 4;
      uint32_t v = remainder > 0 ? (reinterpret_cast<uint32_t const*>(src)[idx] - value_shift) : 0;

      constexpr size_type rows_per_element = 32;
      auto const have_trailing_bits = ((num_elements * rows_per_element) - num_rows) < bit_shift;
      while (remainder) {
        // if we're at the very last word of a validity copy, we do not always need to read the next
        // word to get the final trailing bits.
        auto const read_trailing_bits = bit_shift > 0 && remainder == 4 && have_trailing_bits;
        uint32_t const next           = (read_trailing_bits || remainder > 4)
                                          ? (reinterpret_cast<uint32_t const*>(src)[idx + 1] - value_shift)
                                          : 0;

        uint32_t const val = (v >> bit_shift) | (next << (32 - bit_shift));
        if (valid_count) { thread_valid_count += __popc(val); }
        reinterpret_cast<uint32_t*>(dst)[idx] = val;
        v                                     = next;
        idx++;
        remainder -= 4;
      }
    } else {
      while (remainder) {
        std::size_t const idx = num_bytes - remainder--;
        uint32_t const val    = reinterpret_cast<uint8_t const*>(src)[idx];
        if (valid_count) { thread_valid_count += __popc(val); }
        reinterpret_cast<uint8_t*>(dst)[idx] = val;
      }
    }
  }

  if (valid_count) {
    if (num_bytes == 0) {
      if (!t) { *valid_count = 0; }
    } else {
      using BlockReduce = cub::BlockReduce<size_type, block_size>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      size_type block_valid_count{BlockReduce(temp_storage).Sum(thread_valid_count)};
      if (!t) {
        // we may have copied more bits than there are actual rows in the output.
        // so we need to subtract off the count of any bits that shouldn't have been
        // considered during the copy step.
        std::size_t const max_row    = (num_bytes * 8);
        std::size_t const slack_bits = max_row > num_rows ? max_row - num_rows : 0;
        auto const slack_mask        = set_most_significant_bits(slack_bits);
        if (slack_mask > 0) {
          uint32_t const last_word = reinterpret_cast<uint32_t*>(dst + (num_bytes - 4))[0];
          block_valid_count -= __popc(last_word & slack_mask);
        }
        *valid_count = block_valid_count;
      }
    }
  }
}

/**
 * @brief Kernel which copies data from multiple source buffers to multiple
 * destination buffers.
 *
 * When doing a contiguous_split on X columns comprising N total internal buffers
 * with M splits, we end up having to copy N*M source/destination buffer pairs.
 * These logical copies are further subdivided to distribute the amount of work
 * to be done as evenly as possible across the multiprocessors on the device.
 * This kernel is arranged such that each block copies 1 source/destination pair.
 *
 * @param src_bufs Input source buffers
 * @param dst_bufs Destination buffers
 * @param buf_info Information on the range of values to be copied for each destination buffer.
 */
template <int block_size>
__global__ void copy_partitions(uint8_t const** src_bufs,
                                uint8_t** dst_bufs,
                                dst_buf_info* buf_info)
{
  auto const buf_index     = blockIdx.x;
  auto const src_buf_index = buf_info[buf_index].src_buf_index;
  auto const dst_buf_index = buf_info[buf_index].dst_buf_index;

  // copy, shifting offsets and validity bits as needed
  copy_buffer<block_size>(
    dst_bufs[dst_buf_index] + buf_info[buf_index].dst_offset,
    src_bufs[src_buf_index],
    threadIdx.x,
    buf_info[buf_index].num_elements,
    buf_info[buf_index].element_size,
    buf_info[buf_index].src_element_index,
    blockDim.x,
    buf_info[buf_index].value_shift,
    buf_info[buf_index].bit_shift,
    buf_info[buf_index].num_rows,
    buf_info[buf_index].valid_count > 0 ? &buf_info[buf_index].valid_count : nullptr);
}

// The block of functions below are all related:
//
// compute_offset_stack_size()
// setup_src_buf_data()
// count_src_bufs()
// setup_source_buf_info()
// build_output_columns()
//
// Critically, they all traverse the hierarchy of source columns and their children
// in a specific order to guarantee they produce various outputs in a consistent
// way.  For example, setup_src_buf_info() produces a series of information
// structs that must appear in the same order that setup_src_buf_data() produces
// buffers.
//
// So please be careful if you change the way in which these functions and
// functors traverse the hierarchy.

/**
 * @brief Returns whether or not the specified type is a column that contains offsets.
 */
bool is_offset_type(type_id id) { return (id == type_id::STRING or id == type_id::LIST); }

/**
 * @brief Compute total device memory stack size needed to process nested
 * offsets per-output buffer.
 *
 * When determining the range of rows to be copied for each output buffer
 * we have to recursively apply the stack of offsets from our parent columns
 * (lists or strings).  We want to do this computation on the gpu because offsets
 * are stored in device memory.  However we don't want to do recursion on the gpu, so
 * each destination buffer gets a "stack" of space to work with equal in size to
 * it's offset nesting depth.  This function computes the total size of all of those
 * stacks.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param offset_depth Current offset nesting depth
 *
 * @returns Total offset stack size needed for this range of columns.
 */
template <typename InputIter>
std::size_t compute_offset_stack_size(InputIter begin, InputIter end, int offset_depth = 0)
{
  return std::accumulate(begin, end, 0, [offset_depth](auto stack_size, column_view const& col) {
    auto const num_buffers = 1 + (col.nullable() ? 1 : 0);
    return stack_size + (offset_depth * num_buffers) +
           compute_offset_stack_size(
             col.child_begin(), col.child_end(), offset_depth + is_offset_type(col.type().id()));
  });
}

/**
 * @brief Retrieve all buffers for a range of source columns.
 *
 * Retrieve the individual buffers that make up a range of input columns.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param out_buf Iterator into output buffer infos
 *
 * @returns next output buffer iterator
 */
template <typename InputIter, typename OutputIter>
OutputIter setup_src_buf_data(InputIter begin, InputIter end, OutputIter out_buf)
{
  std::for_each(begin, end, [&out_buf](column_view const& col) {
    if (col.nullable()) {
      *out_buf = reinterpret_cast<uint8_t const*>(col.null_mask());
      out_buf++;
    }
    // NOTE: we're always returning the base pointer here.  column-level offset is accounted
    // for later. Also, for some column types (string, list, struct) this pointer will be null
    // because there is no associated data with the root column.
    *out_buf = col.head<uint8_t>();
    out_buf++;

    out_buf = setup_src_buf_data(col.child_begin(), col.child_end(), out_buf);
  });
  return out_buf;
}

/**
 * @brief Count the total number of source buffers we will be copying
 * from.
 *
 * This count includes buffers for all input columns. For example a
 * fixed-width column with validity would be 2 buffers (data, validity).
 * A string column with validity would be 3 buffers (chars, offsets, validity).
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 *
 * @returns total number of source buffers for this range of columns
 */
template <typename InputIter>
size_type count_src_bufs(InputIter begin, InputIter end)
{
  auto buf_iter = thrust::make_transform_iterator(begin, [](column_view const& col) {
    return 1 + (col.nullable() ? 1 : 0) + count_src_bufs(col.child_begin(), col.child_end());
  });
  return std::accumulate(buf_iter, buf_iter + std::distance(begin, end), 0);
}

/**
 * @brief Computes source buffer information for the copy kernel.
 *
 * For each input column to be split we need to know several pieces of information
 * in the copy kernel.  This function traverses the input columns and prepares this
 * information for the gpu.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param head Beginning of source buffer info array
 * @param current Current source buffer info to be written to
 * @param offset_stack_pos Integer representing our current offset nesting depth
 * (how many list or string levels deep we are)
 * @param parent_offset_index Index into src_buf_info output array indicating our nearest
 * containing list parent. -1 if we have no list parent
 * @param offset_depth Current offset nesting depth (how many list levels deep we are)
 *
 * @returns next src_buf_output after processing this range of input columns
 */
// setup source buf info
template <typename InputIter>
std::pair<src_buf_info*, size_type> setup_source_buf_info(InputIter begin,
                                                          InputIter end,
                                                          src_buf_info* head,
                                                          src_buf_info* current,
                                                          int offset_stack_pos    = 0,
                                                          int parent_offset_index = -1,
                                                          int offset_depth        = 0);

/**
 * @brief Functor that builds source buffer information based on input columns.
 *
 * Called by setup_source_buf_info to build information for a single source column.  This function
 * will recursively call setup_source_buf_info in the case of nested types.
 */
struct buf_info_functor {
  src_buf_info* head;

  template <typename T>
  std::pair<src_buf_info*, size_type> operator()(column_view const& col,
                                                 src_buf_info* current,
                                                 int offset_stack_pos,
                                                 int parent_offset_index,
                                                 int offset_depth)
  {
    if (col.nullable()) {
      std::tie(current, offset_stack_pos) =
        add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
    }

    // info for the data buffer
    *current = src_buf_info(
      col.type().id(), nullptr, offset_stack_pos, parent_offset_index, false, col.offset());

    return {current + 1, offset_stack_pos + offset_depth};
  }

  template <typename T, typename... Args>
  std::enable_if_t<std::is_same_v<T, cudf::dictionary32>, std::pair<src_buf_info*, size_type>>
  operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported type");
  }

 private:
  std::pair<src_buf_info*, size_type> add_null_buffer(column_view const& col,
                                                      src_buf_info* current,
                                                      int offset_stack_pos,
                                                      int parent_offset_index,
                                                      int offset_depth)
  {
    // info for the validity buffer
    *current = src_buf_info(
      type_id::INT32, nullptr, offset_stack_pos, parent_offset_index, true, col.offset());

    return {current + 1, offset_stack_pos + offset_depth};
  }
};

template <>
std::pair<src_buf_info*, size_type> buf_info_functor::operator()<cudf::string_view>(
  column_view const& col,
  src_buf_info* current,
  int offset_stack_pos,
  int parent_offset_index,
  int offset_depth)
{
  if (col.nullable()) {
    std::tie(current, offset_stack_pos) =
      add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // string columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *current = src_buf_info(
    type_id::STRING, nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
  current++;
  offset_stack_pos += offset_depth;

  // string columns don't necessarily have children
  if (col.num_children() > 0) {
    CUDF_EXPECTS(col.num_children() == 2, "Encountered malformed string column");
    strings_column_view scv(col);

    // info for the offsets buffer
    auto offset_col = current;
    CUDF_EXPECTS(not scv.offsets().nullable(), "Encountered nullable string offsets column");
    *current = src_buf_info(type_id::INT32,
                            // note: offsets can be null in the case where the string column
                            // has been created with empty_like().
                            scv.offsets().begin<cudf::id_to_type<type_id::INT32>>(),
                            offset_stack_pos,
                            parent_offset_index,
                            false,
                            col.offset());

    current++;
    offset_stack_pos += offset_depth;

    // since we are crossing an offset boundary, calculate our new depth and parent offset index.
    offset_depth++;
    parent_offset_index = offset_col - head;

    // prevent appending buf_info for non-existent chars buffer
    CUDF_EXPECTS(not scv.chars().nullable(), "Encountered nullable string chars column");

    // info for the chars buffer
    *current = src_buf_info(
      type_id::INT8, nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
    current++;
    offset_stack_pos += offset_depth;
  }

  return {current, offset_stack_pos};
}

template <>
std::pair<src_buf_info*, size_type> buf_info_functor::operator()<cudf::list_view>(
  column_view const& col,
  src_buf_info* current,
  int offset_stack_pos,
  int parent_offset_index,
  int offset_depth)
{
  lists_column_view lcv(col);

  if (col.nullable()) {
    std::tie(current, offset_stack_pos) =
      add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // list columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *current = src_buf_info(
    type_id::LIST, nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
  current++;
  offset_stack_pos += offset_depth;

  CUDF_EXPECTS(col.num_children() == 2, "Encountered malformed list column");

  // info for the offsets buffer
  auto offset_col = current;
  *current        = src_buf_info(type_id::INT32,
                          // note: offsets can be null in the case where the lists column
                          // has been created with empty_like().
                          lcv.offsets().begin<cudf::id_to_type<type_id::INT32>>(),
                          offset_stack_pos,
                          parent_offset_index,
                          false,
                          col.offset());
  current++;
  offset_stack_pos += offset_depth;

  // since we are crossing an offset boundary, calculate our new depth and parent offset index.
  offset_depth++;
  parent_offset_index = offset_col - head;

  return setup_source_buf_info(col.child_begin() + 1,
                               col.child_end(),
                               head,
                               current,
                               offset_stack_pos,
                               parent_offset_index,
                               offset_depth);
}

template <>
std::pair<src_buf_info*, size_type> buf_info_functor::operator()<cudf::struct_view>(
  column_view const& col,
  src_buf_info* current,
  int offset_stack_pos,
  int parent_offset_index,
  int offset_depth)
{
  if (col.nullable()) {
    std::tie(current, offset_stack_pos) =
      add_null_buffer(col, current, offset_stack_pos, parent_offset_index, offset_depth);
  }

  // struct columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *current = src_buf_info(
    type_id::STRUCT, nullptr, offset_stack_pos, parent_offset_index, false, col.offset());
  current++;
  offset_stack_pos += offset_depth;

  // recurse on children
  cudf::structs_column_view scv(col);
  std::vector<column_view> sliced_children;
  sliced_children.reserve(scv.num_children());
  std::transform(thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator(scv.num_children()),
                 std::back_inserter(sliced_children),
                 [&scv](size_type child_index) { return scv.get_sliced_child(child_index); });
  return setup_source_buf_info(sliced_children.begin(),
                               sliced_children.end(),
                               head,
                               current,
                               offset_stack_pos,
                               parent_offset_index,
                               offset_depth);
}

template <typename InputIter>
std::pair<src_buf_info*, size_type> setup_source_buf_info(InputIter begin,
                                                          InputIter end,
                                                          src_buf_info* head,
                                                          src_buf_info* current,
                                                          int offset_stack_pos,
                                                          int parent_offset_index,
                                                          int offset_depth)
{
  std::for_each(begin, end, [&](column_view const& col) {
    std::tie(current, offset_stack_pos) = cudf::type_dispatcher(col.type(),
                                                                buf_info_functor{head},
                                                                col,
                                                                current,
                                                                offset_stack_pos,
                                                                parent_offset_index,
                                                                offset_depth);
  });
  return {current, offset_stack_pos};
}

/**
 * @brief Given a set of input columns, processed split buffers, and a metadata_builder,
 * append column metadata using the builder.
 *
 * After performing the split we are left with 1 large buffer per incoming split
 * partition.  We need to traverse this buffer and distribute the individual
 * subpieces that represent individual columns and children to produce the final
 * output columns.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param info_begin Iterator of dst_buf_info structs containing information about each
 * copied buffer
 * @param mb packed column metadata builder
 *
 * @returns new dst_buf_info iterator after processing this range of input columns
 */
template <typename InputIter, typename BufInfo>
BufInfo build_output_columns(InputIter begin,
                             InputIter end,
                             BufInfo info_begin,
                             metadata_builder& mb)
{
  auto current_info = info_begin;
  std::for_each(begin, end, [&current_info, &mb](column_view const& src) {
    auto [bitmask_offset, null_count] = [&]() {
      if (src.nullable()) {
        auto const ptr =
          current_info->num_elements == 0
            ? 0 
            // TODO: ask: why is this ptr the same as data_ptr?
            : current_info->dst_offset;
        auto const null_count = current_info->num_elements == 0
                                  ? 0
                                  : (current_info->num_rows - current_info->valid_count);
        ++current_info;
        return std::pair((int64_t)ptr, null_count);
      }
      return std::pair((int64_t)0, 0);
    }();

    // size/data pointer for the column
    auto const size = current_info->num_elements;
    int64_t data_offset =
      size == 0 || src.head() == nullptr ? 0 : current_info->dst_offset;
    
    mb.add_column_to_meta(
      src.type(), 
      (size_type) size, 
      (size_type) null_count, 
      src.num_children() > 0 ? -1 : data_offset,
      src.nullable() == 0 ? -1 : bitmask_offset,
      src.num_children());

    ++current_info;

    // children
    current_info = build_output_columns(src.child_begin(), src.child_end(), current_info, mb);
  });

  return current_info;
}

/**
 * @brief Functor that retrieves the size of a destination buffer
 */
struct buf_size_functor {
  dst_buf_info const* ci;
  std::size_t operator() __device__(int index) { return ci[index].buf_size; }
};

/**
 * @brief Functor that retrieves the split "key" for a given output
 * buffer index.
 *
 * The key is simply the partition index.
 */
struct split_key_functor {
  int num_columns;
  int operator() __device__(int buf_index) { return buf_index / num_columns; }
};

/**
 * @brief Output iterator for writing values to the dst_offset field of the
 * dst_buf_info struct
 */
struct dst_offset_output_iterator {
  dst_buf_info* c;
  using value_type        = std::size_t;
  using difference_type   = std::size_t;
  using pointer           = std::size_t*;
  using reference         = std::size_t&;
  using iterator_category = thrust::output_device_iterator_tag;

  dst_offset_output_iterator operator+ __host__ __device__(int i) { return {c + i}; }

  void operator++ __host__ __device__() { c++; }

  reference operator[] __device__(int i) { return dereference(c + i); }
  reference operator* __device__() { return dereference(c); }

 private:
  reference __device__ dereference(dst_buf_info* c) { return c->dst_offset; }
};

/**
 * @brief Output iterator for writing values to the valid_count field of the
 * dst_buf_info struct
 */
struct dst_valid_count_output_iterator {
  dst_buf_info* c;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type*;
  using reference         = size_type&;
  using iterator_category = thrust::output_device_iterator_tag;

  dst_valid_count_output_iterator operator+ __host__ __device__(int i)
  {
    return dst_valid_count_output_iterator{c + i};
  }

  void operator++ __host__ __device__() { c++; }

  reference operator[] __device__(int i) { return dereference(c + i); }
  reference operator* __device__() { return dereference(c); }

 private:
  reference __device__ dereference(dst_buf_info* c) { return c->valid_count; }
};

/**
 * @brief Functor for computing size of data elements for a given cudf type.
 *
 * Note: columns types which themselves inherently have no data (strings, lists,
 * structs) return 0.
 */
struct size_of_helper {
  template <typename T>
  constexpr std::enable_if_t<not is_fixed_width<T>(), int> __device__ operator()() const
  {
    return 0;
  }

  template <typename T>
  constexpr std::enable_if_t<is_fixed_width<T>(), int> __device__ operator()() const noexcept
  {
    return sizeof(cudf::device_storage_type_t<T>);
  }
};

/**
 * @brief Functor for returning the number of chunks an input buffer is being
 * subdivided into during the repartitioning step.
 *
 * Note: columns types which themselves inherently have no data (strings, lists,
 * structs) return 0.
 */
struct num_chunks_func {
  thrust::pair<std::size_t, std::size_t> const* chunks;
  __device__ std::size_t operator()(size_type i) const { return thrust::get<0>(chunks[i]); }
};

/**
 * @brief Get the size in bytes of a chunk described by `dst_buf_info`.
*/
struct chunk_byte_size_function {
  __device__ std::size_t operator()(const dst_buf_info& i) const { 
    // TODO: ask: is this correct, or should we be using the chunk_size?
    // it seems that the chunk size is 1MB most of the time, but the num_elements * element_size is variable:
    //chunk_index: 0 chunk_size 1048576 so far 0 el size: 1048576
    //chunk_index: 1 chunk_size 1048576 so far 0 el size: 201424
    //chunk_index: 2 chunk_size 0 so far 0 el size: 0
    //chunk_index: 3 chunk_size 1048576 so far 0 el size: 1048576
    //chunk_index: 4 chunk_size 1048576 so far 0 el size: 1048576

    const std::size_t bytes = (i.num_elements * i.element_size);
    return util::round_up_unsafe(bytes, split_align);
  }
};

struct chunk_byte_size_func {
  dst_buf_info const* d_dst_buf_info;
  __device__ std::size_t operator()(size_type i) const { 
    // TODO: ask: is this correct, or should we be using the chunk_size?
    // it seems that the chunk size is 1MB most of the time, but the num_elements * element_size is variable:
    //chunk_index: 0 chunk_size 1048576 so far 0 el size: 1048576
    //chunk_index: 1 chunk_size 1048576 so far 0 el size: 201424
    //chunk_index: 2 chunk_size 0 so far 0 el size: 0
    //chunk_index: 3 chunk_size 1048576 so far 0 el size: 1048576
    //chunk_index: 4 chunk_size 1048576 so far 0 el size: 1048576
    auto el = d_dst_buf_info[i];
    const std::size_t bytes = (el.num_elements * el.element_size);
    return util::round_up_unsafe(bytes, split_align);
  }
};

};  // anonymous namespace

namespace detail {

// CHUNKED A
// packed block of memory 1. split indices and src_buf_info structs
struct packed_split_indices_and_src_buf_info {
  explicit packed_split_indices_and_src_buf_info(cudf::table_view const& input,
                                        std::vector<size_type> const& splits,
                                        std::size_t num_partitions,
                                        cudf::size_type num_src_bufs,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
  {
    indices_size = cudf::util::round_up_safe((num_partitions + 1) * sizeof(size_type), split_align);
    src_buf_info_size = cudf::util::round_up_safe(num_src_bufs * sizeof(src_buf_info), split_align);

    // host-side
    h_indices_and_source_info = std::vector<uint8_t>(indices_size + src_buf_info_size);
    h_indices                 = reinterpret_cast<size_type*>(h_indices_and_source_info.data());
    h_src_buf_info =
      reinterpret_cast<src_buf_info*>(h_indices_and_source_info.data() + indices_size);

    // compute splits -> indices.
    // these are row numbers per split
    h_indices[0]              = 0;
    h_indices[num_partitions] = input.column(0).size();
    std::copy(splits.begin(), splits.end(), std::next(h_indices));

    // setup source buf info
    // TODO: ask: learn how this works
    setup_source_buf_info(input.begin(), input.end(), h_src_buf_info, h_src_buf_info);

    offset_stack_partition_size = compute_offset_stack_size(input.begin(), input.end());
    offset_stack_size           = offset_stack_partition_size * num_partitions * sizeof(size_type);
    // device-side
    // gpu-only : stack space needed for nested list offset calculation
    d_indices_and_source_info =
      rmm::device_buffer(indices_size + src_buf_info_size + offset_stack_size,
                         stream,
                         rmm::mr::get_current_device_resource());
    d_indices      = reinterpret_cast<size_type*>(d_indices_and_source_info.data());
    d_src_buf_info = reinterpret_cast<src_buf_info*>(
      reinterpret_cast<uint8_t*>(d_indices_and_source_info.data()) + indices_size);
    d_offset_stack =
      reinterpret_cast<size_type*>(reinterpret_cast<uint8_t*>(d_indices_and_source_info.data()) +
                                   indices_size + src_buf_info_size);

    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_indices, h_indices, indices_size + src_buf_info_size, cudaMemcpyDefault, stream.value()));
  }

  size_type indices_size;
  std::size_t src_buf_info_size;
  std::size_t offset_stack_size;

  std::vector<uint8_t> h_indices_and_source_info;
  rmm::device_buffer d_indices_and_source_info;

  size_type* h_indices;
  src_buf_info* h_src_buf_info;

  int offset_stack_partition_size;
  size_type* d_indices;
  src_buf_info* d_src_buf_info;
  size_type* d_offset_stack;
};

// packed block of memory 2. partition buffer sizes and dst_buf_info structs
struct packed_partition_buf_size_and_dst_buf_info {
  packed_partition_buf_size_and_dst_buf_info(
    std::size_t num_partitions,
    cudf::size_type num_src_bufs,
    std::size_t num_bufs,
    std::unique_ptr<packed_split_indices_and_src_buf_info> split_indices_and_src_buf_info,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
    : packed_partition_buf_size_and_dst_buf_info(
        num_partitions, 
        num_src_bufs, 
        num_bufs, 
        std::move(split_indices_and_src_buf_info), 
        nullptr, 
        0, 
        stream, 
        mr)
  {
  }

  packed_partition_buf_size_and_dst_buf_info(std::size_t num_partitions,
                                             cudf::size_type num_src_bufs,
                                             std::size_t num_bufs,
                                             std::unique_ptr<packed_split_indices_and_src_buf_info> split_indices_and_src_buf_info,
                                             uint8_t* user_provided_buffer,
                                             std::size_t user_provided_buffer_size,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
  {
    buf_sizes_size = cudf::util::round_up_safe(num_partitions * sizeof(std::size_t), split_align);
    dst_buf_info_size = cudf::util::round_up_safe(num_bufs * sizeof(dst_buf_info), split_align);
    // host-side
    h_buf_sizes_and_dst_info = std::vector<uint8_t>(buf_sizes_size + dst_buf_info_size);
    h_buf_sizes              = reinterpret_cast<std::size_t*>(h_buf_sizes_and_dst_info.data());
    h_dst_buf_info =
      reinterpret_cast<dst_buf_info*>(h_buf_sizes_and_dst_info.data() + buf_sizes_size);

    // device-side
    d_buf_sizes_and_dst_info = rmm::device_buffer(
      buf_sizes_size + dst_buf_info_size, stream, rmm::mr::get_current_device_resource());
    d_buf_sizes = reinterpret_cast<std::size_t*>(d_buf_sizes_and_dst_info.data());

    //// destination buffer info
    d_dst_buf_info = reinterpret_cast<dst_buf_info*>(
      static_cast<uint8_t*>(d_buf_sizes_and_dst_info.data()) + buf_sizes_size);

    initialize(num_src_bufs, num_bufs, split_indices_and_src_buf_info.get(), stream, mr);

    user_provided_out_buffer = user_provided_buffer;
    user_provided_out_buffer_size = user_provided_buffer_size;

    // allocate output partition buffers
    if (user_provided_buffer == nullptr) {
      out_buffers.reserve(num_partitions);
      std::transform(h_buf_sizes,
                    h_buf_sizes + num_partitions,
                    std::back_inserter(out_buffers),
                    [stream = stream, mr = mr](std::size_t bytes) {
                      return rmm::device_buffer{bytes, stream, mr};
                    });
    }
  }

  void initialize(cudf::size_type num_src_bufs,
                  std::size_t num_bufs,
                  packed_split_indices_and_src_buf_info* split_indices_and_src_buf_info,
                  rmm::cuda_stream_view stream,
                  rmm::mr::device_memory_resource* mr) {
    auto const d_src_buf_info        = split_indices_and_src_buf_info->d_src_buf_info;
    auto offset_stack_partition_size = split_indices_and_src_buf_info->offset_stack_partition_size;
    auto d_offset_stack              = split_indices_and_src_buf_info->d_offset_stack;
    auto d_indices                   = split_indices_and_src_buf_info->d_indices;

    // compute sizes of each column in each partition, including alignment.
    thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<std::size_t>(0),
      thrust::make_counting_iterator<std::size_t>(num_bufs),
      d_dst_buf_info,
      [d_src_buf_info,
       offset_stack_partition_size,
       d_offset_stack,
       d_indices,
       num_src_bufs] __device__(std::size_t t) {
        int const split_index   = t / num_src_bufs;
        int const src_buf_index = t % num_src_bufs;
        auto const& src_info    = d_src_buf_info[src_buf_index];

        // apply nested offsets (lists and string columns).
        //
        // We can't just use the incoming row indices to figure out where to read from in a
        // nested list situation.  We have to apply offsets every time we cross a boundary
        // (list or string).  This loop applies those offsets so that our incoming row_index_start
        // and row_index_end get transformed to our final values.
        //
        int const stack_pos =
          src_info.offset_stack_pos + (split_index * offset_stack_partition_size);
        size_type* offset_stack  = &(d_offset_stack[stack_pos]);
        int parent_offsets_index = src_info.parent_offsets_index;
        int stack_size           = 0;
        int root_column_offset   = src_info.column_offset;

        // TODO: ask: what is this loop doing
        while (parent_offsets_index >= 0) {
          offset_stack[stack_size++] = parent_offsets_index;
          root_column_offset         = d_src_buf_info[parent_offsets_index].column_offset;
          parent_offsets_index       = d_src_buf_info[parent_offsets_index].parent_offsets_index;
        }
        // make sure to include the -column- offset on the root column in our calculation.
        int row_start = d_indices[split_index] + root_column_offset;
        int row_end   = d_indices[split_index + 1] + root_column_offset;
        while (stack_size > 0) {
          stack_size--;
          auto const offsets = d_src_buf_info[offset_stack[stack_size]].offsets;
          // this case can happen when you have empty string or list columns constructed with
          // empty_like()
          if (offsets != nullptr) {
            row_start = offsets[row_start];
            row_end   = offsets[row_end];
          }
        }

        // final element indices and row count
        int const out_element_index = src_info.is_validity ? row_start / 32 : row_start;
        int const num_rows          = row_end - row_start;
        // if I am an offsets column, all my values need to be shifted
        int const value_shift = src_info.offsets == nullptr ? 0 : src_info.offsets[row_start];
        // if I am a validity column, we may need to shift bits
        int const bit_shift = src_info.is_validity ? row_start % 32 : 0;
        // # of rows isn't necessarily the same as # of elements to be copied.
        auto const num_elements = [&]() {
          if (src_info.offsets != nullptr && num_rows > 0) {
            return num_rows + 1;
          } else if (src_info.is_validity) {
            return (num_rows + 31) / 32;
          }
          return num_rows;
        }();
        int const element_size = cudf::type_dispatcher(data_type{src_info.type}, size_of_helper{});
        std::size_t const bytes =
          static_cast<std::size_t>(num_elements) * static_cast<std::size_t>(element_size);

        return dst_buf_info{util::round_up_unsafe(bytes, split_align),
                            num_elements,
                            element_size,
                            num_rows,
                            out_element_index,
                            0,
                            value_shift,
                            bit_shift,
                            src_info.is_validity ? 1 : 0,
                            src_buf_index,
                            split_index};
      });

    // compute total size of each partition
    // key is split index
    {
      auto keys = cudf::detail::make_counting_transform_iterator(
        0, split_key_functor{static_cast<int>(num_src_bufs)});
      auto values =
        cudf::detail::make_counting_transform_iterator(0, buf_size_functor{d_dst_buf_info});

      thrust::reduce_by_key(rmm::exec_policy(stream),
                            keys,
                            keys + num_bufs,
                            values,
                            thrust::make_discard_iterator(),
                            d_buf_sizes);
    }

    // compute num_rows          = row_end - row_start; start offset for each output buffer
    {
      auto keys = cudf::detail::make_counting_transform_iterator(
        0, split_key_functor{static_cast<int>(num_src_bufs)});
      auto values =
        cudf::detail::make_counting_transform_iterator(0, buf_size_functor{d_dst_buf_info});

      thrust::exclusive_scan_by_key(rmm::exec_policy(stream),
                                    keys,
                                    keys + num_bufs,
                                    values,
                                    dst_offset_output_iterator{d_dst_buf_info},
                                    std::size_t{0});
    }
  
    // DtoH buf sizes and col info back to the host
    CUDF_CUDA_TRY(cudaMemcpyAsync(h_buf_sizes,
                                  d_buf_sizes,
                                  buf_sizes_size + dst_buf_info_size,
                                  cudaMemcpyDefault,
                                  stream.value()));

    stream.synchronize();
  }

  // buffer sizes and destination info (used in chunked copies)
  std::size_t buf_sizes_size;
  std::size_t dst_buf_info_size;

  std::vector<uint8_t> h_buf_sizes_and_dst_info;
  std::size_t* h_buf_sizes;
  dst_buf_info* h_dst_buf_info;

  rmm::device_buffer d_buf_sizes_and_dst_info;
  std::size_t* d_buf_sizes;
  dst_buf_info* d_dst_buf_info;

  std::vector<rmm::device_buffer> out_buffers;
  uint8_t* user_provided_out_buffer;
  std::size_t user_provided_out_buffer_size;
};

// Packed block of memory 3:
// Pointers to source and destination buffers (and stack space on the
// gpu for offset computation)
struct packed_src_and_dst_pointers {
  packed_src_and_dst_pointers(cudf::table_view const& input,
                              std::size_t num_partitions,
                              cudf::size_type num_src_bufs,
                              packed_partition_buf_size_and_dst_buf_info* buf_info,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr):stream(stream)
  {
    src_bufs_size =
      cudf::util::round_up_safe(num_src_bufs * sizeof(uint8_t*), split_align);
    dst_bufs_size =
      cudf::util::round_up_safe(num_partitions * sizeof(uint8_t*), split_align);

    // TODO: ask why when we copy h_src_bufs to device we copy with src_bufs_size + dst_bufs_size...
    // host-side
    h_src_and_dst_buffers = std::vector<uint8_t>(src_bufs_size + dst_bufs_size);
    h_src_bufs = reinterpret_cast<uint8_t const**>(h_src_and_dst_buffers.data());
    h_dst_bufs = reinterpret_cast<uint8_t**>(h_src_and_dst_buffers.data() + src_bufs_size);

    //
    // CHUNKED B: in the chunked case, there will only ever be 1 destination buffer, passed in
    //            by the caller.
    //
    // setup dst buffers
    if (buf_info->user_provided_out_buffer == nullptr) {
      std::transform(buf_info->out_buffers.begin(), buf_info->out_buffers.end(), h_dst_bufs, [](auto& buf) {
        return static_cast<uint8_t*>(buf.data());
      });
    } else {
      h_dst_bufs[0] = buf_info->user_provided_out_buffer;
    }

    // TODO: ASK: why do we add offset_stack_size here
    //  --> not needed actually, we can remove
    // device-side
    d_src_and_dst_buffers = rmm::device_buffer(
      src_bufs_size + dst_bufs_size, stream, rmm::mr::get_current_device_resource());
    d_src_bufs = reinterpret_cast<uint8_t const**>(d_src_and_dst_buffers.data());
    d_dst_bufs = reinterpret_cast<uint8_t**>(
      reinterpret_cast<uint8_t*>(d_src_and_dst_buffers.data()) + src_bufs_size);

    // setup src buffers
    setup_src_buf_data(input.begin(), input.end(), h_src_bufs);
  }

  void copy_to_device() {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_src_bufs, h_src_bufs, src_bufs_size + dst_bufs_size, cudaMemcpyDefault, stream.value()));
  }

  const rmm::cuda_stream_view stream;

  std::vector<uint8_t> h_src_and_dst_buffers;
  rmm::device_buffer d_src_and_dst_buffers;
  std::size_t src_bufs_size; 
  std::size_t dst_bufs_size; 
  const uint8_t** h_src_bufs;
  const uint8_t** d_src_bufs;
  uint8_t** h_dst_bufs;
  uint8_t** d_dst_bufs;
};

struct iteration_state {
  iteration_state(rmm::device_uvector<dst_buf_info> _d_dst_buf_info, int num_expected_copies)
    : d_dst_buf_info(std::move(_d_dst_buf_info)),
      h_keys(num_expected_copies),
      h_num_buffs_per_key(num_expected_copies),
      h_size_of_buffs_per_key(num_expected_copies)
  {
  }

  rmm::device_uvector<dst_buf_info> d_dst_buf_info;
  std::vector<std::size_t> h_keys;
  std::vector<std::size_t> h_num_buffs_per_key;
  std::vector<std::size_t> h_size_of_buffs_per_key;
};

std::unique_ptr<iteration_state> get_dst_buf_info(
  rmm::device_uvector<thrust::pair<std::size_t, std::size_t>>& chunks,
  rmm::device_uvector<offset_type>& chunk_offsets,
  int num_chunks,
  int num_bufs,
  int num_src_bufs,
  dst_buf_info* _d_dst_buf_info,
  std::optional<std::size_t> user_buffer_size,
  rmm::cuda_stream_view stream) {

  auto out_to_in_index = [chunk_offsets = chunk_offsets.begin(), num_bufs] __device__(size_type i) {
    return static_cast<size_type>(
             thrust::upper_bound(thrust::seq, chunk_offsets, chunk_offsets + num_bufs + 1, i) -
             chunk_offsets) -
           1;
  };

  auto iter = thrust::make_counting_iterator(0);

  // load up the chunks as d_dst_buf_info
  rmm::device_uvector<dst_buf_info> d_dst_buf_info(num_chunks, stream);

  // TODO: ask: I think we can create the chunk d_dst_buf_info once in device memory
  // then just make sure we start at the right index, rather than create new d_dst_buf_info
  // for a particular bounce buffer copy
  thrust::for_each(
    rmm::exec_policy(stream),
    iter,
    iter + num_chunks,
    [_d_dst_buf_info,
     d_dst_buf_info = d_dst_buf_info.begin(),
     chunks         = chunks.begin(),
     chunk_offsets  = chunk_offsets.begin(),
     num_bufs,
     num_src_bufs,
     out_to_in_index] __device__(size_type i) {
      size_type const in_buf_index = out_to_in_index(i);
      size_type const chunk_index  = i - chunk_offsets[in_buf_index];
      auto const chunk_size        = thrust::get<1>(chunks[in_buf_index]);
      
      dst_buf_info const& in       = _d_dst_buf_info[in_buf_index];

      // adjust info
      dst_buf_info& out = d_dst_buf_info[i];
      out.element_size  = in.element_size;
      out.value_shift   = in.value_shift;
      out.bit_shift     = in.bit_shift;
      out.valid_count =
        in.valid_count;  // valid count will be set to 1 if this is a validity buffer
      out.src_buf_index = in.src_buf_index;
      out.dst_buf_index = in.dst_buf_index;

      size_type const elements_per_chunk =
        out.element_size == 0 ? 0 : chunk_size / out.element_size;
      out.num_elements = ((chunk_index + 1) * elements_per_chunk) > in.num_elements
                           ? in.num_elements - (chunk_index * elements_per_chunk)
                           : elements_per_chunk;

      size_type const rows_per_chunk =
        // if this is a validity buffer, each element is a bitmask_type, which
        // corresponds to 32 rows.
        out.valid_count > 0
          ? elements_per_chunk * static_cast<size_type>(cudf::detail::size_in_bits<bitmask_type>())
          : elements_per_chunk;
      out.num_rows = ((chunk_index + 1) * rows_per_chunk) > in.num_rows
                       ? in.num_rows - (chunk_index * rows_per_chunk)
                       : rows_per_chunk;

      out.src_element_index = in.src_element_index + (chunk_index * elements_per_chunk);
      out.dst_offset        = in.dst_offset + (chunk_index * chunk_size);

      // out.bytes and out.buf_size are unneeded here because they are only used to
      // calculate real output buffer sizes. the data we are generating here is
      // purely intermediate for the purposes of doing more uniform copying of data
      // underneath the final structure of the output
    });


  std::unique_ptr<iteration_state> istate; 
  if (user_buffer_size) {
    auto user_buffer_sz = user_buffer_size.value();

    // copy the chunk sizes back to host
    std::vector<std::size_t> h_sizes(num_chunks);
    {
      rmm::device_uvector<std::size_t> sizes(num_chunks, stream);
      thrust::transform(rmm::exec_policy(stream),
                        d_dst_buf_info.begin(),
                        d_dst_buf_info.end(),
                        sizes.begin(),
                        chunk_byte_size_function());
      
      CUDF_CUDA_TRY(cudaMemcpyAsync(h_sizes.data(),
                                    sizes.data(),
                                    sizeof(std::size_t) * sizes.size(),
                                    cudaMemcpyDefault,
                                    stream.value()));

      stream.synchronize();
    }

    // compute the chunks size and offsets
    std::vector<std::size_t> offset_per_chunk(num_chunks);
    std::vector<std::size_t> num_chunks_per_split;
    std::vector<std::size_t> size_of_chunks_per_split;
    std::vector<std::size_t> accum_size_per_split;
    {
      std::size_t current_split_num_chunks = 0;
      std::size_t current_split_size       = 0;
      std::size_t accum_size               = 0;

      int current_split = 0;
      for (int i = 0; i < h_sizes.size(); ++i) {
        auto curr_size = h_sizes[i];
        if (current_split_size + curr_size > user_buffer_sz) {
          num_chunks_per_split.push_back(current_split_num_chunks);
          size_of_chunks_per_split.push_back(current_split_size);
          accum_size_per_split.push_back(accum_size);
          current_split_num_chunks = 0;
          current_split_size       = 0;
          ++current_split;
        }
        offset_per_chunk[i] = current_split;
        current_split_size += curr_size;
        accum_size += curr_size;
        ++current_split_num_chunks;
      }
      if (current_split_num_chunks > 0) {
        num_chunks_per_split.push_back(current_split_num_chunks);
        size_of_chunks_per_split.push_back(current_split_size);
        accum_size_per_split.push_back(accum_size);
      }
    }

    // apply changed offset
    {
      rmm::device_uvector<std::size_t> d_offset_per_chunk(num_chunks, stream);
      rmm::device_uvector<std::size_t> d_accum_size_per_split(accum_size_per_split.size(), stream);

      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_offset_per_chunk.data(), offset_per_chunk.data(), num_chunks * sizeof(std::size_t), cudaMemcpyDefault, stream.value()));
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_accum_size_per_split.data(), accum_size_per_split.data(), accum_size_per_split.size() * sizeof(std::size_t), cudaMemcpyDefault, stream.value()));

      // we want to update the offset of chunks in the second to last copy
      auto num_chunks_in_first_split = num_chunks_per_split[0];
      auto iter = thrust::make_counting_iterator(num_chunks_in_first_split);
      thrust::for_each(rmm::exec_policy(stream),
                       iter,
                       iter + num_chunks - num_chunks_in_first_split,
                       [d_dst_buf_info = d_dst_buf_info.begin(),
                        d_accum_size_per_split = d_accum_size_per_split.begin(),
                        d_offset_per_chunk = d_offset_per_chunk.begin()] __device__(size_type i) {
                          auto split = d_offset_per_chunk[i];
                          d_dst_buf_info[i].dst_offset -= d_accum_size_per_split[split - 1];
                       });
    }
    istate = std::make_unique<iteration_state>(std::move(d_dst_buf_info), num_chunks_per_split.size());
    istate->h_num_buffs_per_key = std::move(num_chunks_per_split);
    istate->h_size_of_buffs_per_key = std::move(size_of_chunks_per_split);
  } else {
    auto size_iter = cudf::detail::make_counting_transform_iterator(
      0, chunk_byte_size_func{d_dst_buf_info.begin()});

    auto last_size = thrust::reduce(rmm::exec_policy(stream), size_iter, size_iter + num_chunks);

    istate = std::make_unique<iteration_state>(std::move(d_dst_buf_info), 1);
    istate->h_keys[0] = 0;
    istate->h_num_buffs_per_key[0] = num_chunks;
    istate->h_size_of_buffs_per_key[0] = last_size; 
  }

 return std::move(istate);
}

void copy_data(int num_chunks_to_copy,
               int starting_chunk,
               uint8_t const** d_src_bufs,
               uint8_t** d_dst_bufs,
               rmm::device_uvector<dst_buf_info>& d_dst_buf_info,
               rmm::cuda_stream_view stream)
{
  constexpr size_type block_size = 256;
  copy_partitions<block_size><<<num_chunks_to_copy, block_size, 0, stream.value()>>>(
    d_src_bufs, d_dst_bufs, d_dst_buf_info.data() + starting_chunk);
}

void copy_data_regular(rmm::device_uvector<thrust::pair<std::size_t, std::size_t>>& chunks,
                       rmm::device_uvector<offset_type>& chunk_offsets,
                       int num_bufs,
                       int num_chunks_to_copy,
                       int starting_chunk,
                       uint8_t const** d_src_bufs,
                       uint8_t** d_dst_bufs,
                       rmm::device_uvector<dst_buf_info>& d_dst_buf_info,
                       rmm::cuda_stream_view stream)
{
  constexpr size_type block_size = 256;
  copy_partitions<block_size><<<num_chunks_to_copy, block_size, 0, stream.value()>>>(
    d_src_bufs, d_dst_bufs, d_dst_buf_info.data() + starting_chunk);

  auto out_to_in_index = [chunk_offsets = chunk_offsets.begin(), num_bufs] __device__(size_type i) {
    return static_cast<size_type>(
             thrust::upper_bound(thrust::seq, chunk_offsets, chunk_offsets + num_bufs + 1, i) -
             chunk_offsets) -
           1;
  };

  //
  // TODO: we do need this for the regular contig split case
  // CHUNKED G:  this step is unnecessary for the chunked case, since with 0 partitions we
  //             can just use the null count of the original columns
  // postprocess valid_counts

  auto _d_dst_buf_info = d_dst_buf_info.data(); 
  auto keys = cudf::detail::make_counting_transform_iterator(
    0, [out_to_in_index] __device__(size_type i) { return out_to_in_index(i); });
  auto values = thrust::make_transform_iterator(
    d_dst_buf_info.begin(), [] __device__(dst_buf_info const& info) { return info.valid_count; });
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        keys,
                        keys + num_chunks_to_copy,
                        values,
                        thrust::make_discard_iterator(),
                        dst_valid_count_output_iterator{_d_dst_buf_info});
}

std::size_t get_num_partitions(std::vector<size_type> const& splits) {
  return splits.size() + 1;
}

struct chunk_infos {
  chunk_infos(rmm::device_uvector<thrust::pair<std::size_t, std::size_t>> _chunks,
              rmm::device_uvector<offset_type> _chunk_offsets)
    : chunks(std::move(_chunks)), chunk_offsets(std::move(_chunk_offsets))
  {
  }

  rmm::device_uvector<thrust::pair<std::size_t, std::size_t>> chunks;
  rmm::device_uvector<offset_type> chunk_offsets;
};

bool check_inputs(cudf::table_view const& input, std::vector<size_type> const& splits) 
{
  if (input.num_columns() == 0) {
    // TODO: does this work for no columns?
    return true;
  }
  if (splits.size() > 0) {
    CUDF_EXPECTS(splits.back() <= input.column(0).size(),
                 "splits can't exceed size of input columns");
  }
  {
    size_type begin = 0;
    for (std::size_t i = 0; i < splits.size(); i++) {
      size_type end = splits[i];
      CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.");
      CUDF_EXPECTS(end >= begin, "End index cannot be smaller than the starting index.");
      CUDF_EXPECTS(end <= input.column(0).size(), "Slice range out of bounds.");
      begin = end;
    }
  }
  // if inputs are empty, just return num_partitions empty tables
  return input.column(0).size() == 0;
}

struct contiguous_split_state {
  //
  // CHUNKED A:  This section of code gets refactored out and is called during the first 
  //             chunked pack call. Everything it computes will be valid across calls except
  //             the destination buffer data. all the data/vars here will need to be stored in
  //             some class/struct.
  //
  // compute # of source buffers (column data, validity, children), # of partitions
  // and total # of buffers

  // num_src_buffs = count_src_bufs(col1, .., colN)  -> result the count of the buffers in the columns
  //   and children flattened
  // num_partitions will be 1 (0 splits)

  contiguous_split_state(cudf::table_view const& input,
            std::vector<size_type> const& splits,
            std::unique_ptr<packed_partition_buf_size_and_dst_buf_info> _partition_buf_size_and_dst_buf_info,
            rmm::cuda_stream_view stream,
            rmm::mr::device_memory_resource* mr)
    : stream(stream),
      mr(mr),
      bytes_copied_so_far(0),
      starting_buf(0),
      buffs_to_copy(0),
      copy_iteration(0),
      partition_buf_size_and_dst_buf_info(std::move(_partition_buf_size_and_dst_buf_info))
  {
    is_empty = check_inputs(input, splits);
    num_partitions = get_num_partitions(splits);
    num_src_bufs = count_src_bufs(input.begin(), input.end());
    num_bufs = num_src_bufs * num_partitions;

    src_and_dst_pointers = std::make_unique<packed_src_and_dst_pointers>(
        input, num_partitions, num_src_bufs, partition_buf_size_and_dst_buf_info.get(), stream, mr);

    compute_chunks();
    packed_metadata = make_packed_column_metadata(input, num_partitions);
    prepare_chunked_copy();
  }  // pass the new packed structs here?

  void compute_chunks() {
    //
    // CHUNKED F:  So this is where the "real" array of destination buffers to be copied
    // (_d_dst_buf_info)
    //             is further partitioned into smaller chunks.  these more granular chunks are what
    //             gets passed to the copy kernel.  So ultimately what needs to happen is
    //               - refactor out this whole block so it can be called by the chunked packer
    //               seperately.
    //               - verify that the ordering of the chunks is linear with the overall output
    //                 buffer. that is,
    // TODO: I do not think I have succeeded at verifying this:
    //                 d_dst_buf_info[N] is copying to the destination location exactly where
    //                 d_dst_buf_info[N-1] ends.
    //               - the intermediate chunked struct would store d_dst_buf_info across calls as
    //                 well as the count of buffers we've processed so far.
    //               - when it's time to pack another chunk, we are given a buffer and a size from
    //                 the caller.  we search
    //                 forward from the last used pos in the d_dst_buf_info array until we cross the
    //                 output size (there's a good example of doing this quickly in
    //                 cpp/io/src/parquet/reader_impl_preprocess.cu -> find_splits. we call
    //                 copy_partitions on that subset of dst_buf_infos.
    //               - I -think- the only thing that will need to get updated for each chunk is
    //                 dst_buf_info::dst_offset.
    //                 Everything else that would need to be changed should be 0 (value_shift,
    //                 bit_shift, etc) because we're only outputting 1 "real" partition at the end
    //                 of the day
    //               - In the packed case, after computing d_dst_buf_info, do a scan on all the
    //                 sizes to generate
    //                 cumulative sizes so we can determine what chunks to read. This will get
    //                 stored in the intermediate data.
    //
    // Since we parallelize at one block per copy, we are vulnerable to situations where we
    // have small numbers of copies to do (a combination of small numbers of splits and/or columns),
    // so we will take the actual set of outgoing source/destination buffers and further partition
    // them into much smaller chunks in order to drive up the number of blocks and overall
    // occupancy.
    auto const desired_chunk_size = std::size_t{1 * 1024 * 1024};
    // TODO: should probably call this something differently instead of just chunks.
    rmm::device_uvector<thrust::pair<std::size_t, std::size_t>> chunks(num_bufs, stream);
    auto& d_dst_buf_info = partition_buf_size_and_dst_buf_info->d_dst_buf_info;
    thrust::transform(
      rmm::exec_policy(stream),
      d_dst_buf_info,
      d_dst_buf_info + num_bufs,
      chunks.begin(),
      [desired_chunk_size] __device__(
        dst_buf_info const& buf) -> thrust::pair<std::size_t, std::size_t> {
        // Total bytes for this incoming partition
        std::size_t const bytes =
          static_cast<std::size_t>(buf.num_elements) * static_cast<std::size_t>(buf.element_size);

        // This clause handles nested data types (e.g. list or string) that store no data in the row
        // columns, only in their children.
        if (bytes == 0) { return {1, 0}; }

        // The number of chunks we want to subdivide this buffer into
        std::size_t const num_chunks = std::max(
          std::size_t{1}, util::round_up_unsafe(bytes, desired_chunk_size) / desired_chunk_size);

        // NOTE: leaving chunk size as a separate parameter for future tuning
        // possibilities, even though in the current implementation it will be a
        // constant.
        return {num_chunks, desired_chunk_size};
      });


    std::size_t& my_num_bufs = num_bufs;
    rmm::device_uvector<offset_type> chunk_offsets(num_bufs + 1, stream);
    auto buf_count_iter = cudf::detail::make_counting_transform_iterator(
      0, [my_num_bufs, 
          num_chunks = num_chunks_func{chunks.begin()}] __device__(size_type i) {
        return i == my_num_bufs ? 0 : num_chunks(i);
      });

    thrust::exclusive_scan(rmm::exec_policy(stream),
                           buf_count_iter,
                           buf_count_iter + num_bufs + 1,
                           chunk_offsets.begin(),
                           0);

    // used during the copy
    computed_chunks = std::make_unique<chunk_infos>(std::move(chunks), std::move(chunk_offsets));
  }

  // TODO: could we merge this with the regular packed columns?
  std::vector<packed_columns::metadata> make_empty_column_metadata(
    cudf::table_view const& input,
    std::size_t num_partitions) {
    // TODO: ASK: why needed? sanitize the inputs (to handle corner cases like sliced tables)
    std::size_t empty_num_partitions   = num_partitions;
    std::vector<std::unique_ptr<column>> empty_columns;
    empty_columns.reserve(input.num_columns());
    std::transform(
      input.begin(), input.end(), std::back_inserter(empty_columns), [](column_view const& col) {
        return cudf::empty_like(col); // return new empty column
      });
    std::vector<cudf::column_view> empty_column_views;
    empty_column_views.reserve(input.num_columns());
    std::transform(empty_columns.begin(),
                   empty_columns.end(),
                   std::back_inserter(empty_column_views),
                   [](std::unique_ptr<column> const& col) { return col->view(); });
    table_view empty_inputs(empty_column_views);

    // build the empty results
    std::vector<packed_columns::metadata> result;
    result.reserve(empty_num_partitions);
    auto iter = thrust::make_counting_iterator(0);
    std::transform(iter,
                   iter + empty_num_partitions,
                   std::back_inserter(result),
                   [&empty_inputs](int partition_index) {
                     return cudf::pack_metadata(
                      empty_inputs, static_cast<uint8_t const*>(nullptr), 0);
                   });
    return result;
  }

  std::vector<packed_columns::metadata> make_packed_column_metadata(
    cudf::table_view const& input,
    std::size_t num_partitions) 
  {
    if (is_empty) {
      return make_empty_column_metadata(input, num_partitions);
    }
    //
    // CHUNKED E: This is a little ugly.  This is the code that produces the metadata, but it does
    // so
    //            by first wrapping everything in column_views and a table_view and then calling
    //            pack_metadata(). but in the chunked case, we're not going to have real pointers
    //            or even any backing allocation (which pack_metadata relies on).
    //
    // build the output.
    std::vector<packed_columns::metadata> result;
    result.reserve(num_partitions);

    auto& h_dst_buf_info = partition_buf_size_and_dst_buf_info->h_dst_buf_info;
    auto cur_dst_buf_info = h_dst_buf_info;
    for (std::size_t idx = 0; idx < num_partitions; idx++) {
      metadata_builder mb(input.num_columns());

      // traverse the buffers and build the columns.
      cur_dst_buf_info = build_output_columns(
        input.begin(), 
        input.end(), 
        cur_dst_buf_info, 
        mb); //h_dst_bufs[idx]); //TODO: h_dst_bufs also points to out_buffers

      // pack the columns
      result.push_back(mb.build());
    }

    return result;
  }

  void prepare_chunked_copy() {
    //
    // CHUNKED C: If we execute this for every chunk it is mildly wasteful since the "src" info will
    //            already be in place - only the dst buf info will have changed. this is fine
    //            though. the data is small.
    //
    // HtoD src and dest buffers
    src_and_dst_pointers->copy_to_device();
    
    auto out_buffer_size = partition_buf_size_and_dst_buf_info->user_provided_out_buffer != nullptr ? 
      std::optional<std::size_t>(partition_buf_size_and_dst_buf_info->user_provided_out_buffer_size)
      : std::nullopt;
    
    auto const num_chunks =
      cudf::detail::make_counting_transform_iterator(0, num_chunks_func{computed_chunks->chunks.begin()});
    size_type const new_buf_count =
      thrust::reduce(rmm::exec_policy(stream), num_chunks, num_chunks + computed_chunks->chunks.size());

    auto s = get_dst_buf_info(computed_chunks->chunks,
                              computed_chunks->chunk_offsets,
                              new_buf_count,
                              num_bufs,
                              num_src_bufs,
                              partition_buf_size_and_dst_buf_info->d_dst_buf_info,
                              out_buffer_size,
                              stream);
    state = std::move(s);
  }

  cudf::size_type perform_chunked_copy() {
    // TODO: perhaps this happens on intialization
    CUDF_EXPECTS(partition_buf_size_and_dst_buf_info->user_provided_out_buffer != nullptr,
      "Cannot perform chunked contiguous split without a user buffer");
    
    auto num_chunks_to_copy = state->h_num_buffs_per_key[copy_iteration];
    auto bytes_copied = state->h_size_of_buffs_per_key[copy_iteration];

    // perform the copy.
    copy_data(num_chunks_to_copy,
              starting_buf,
              src_and_dst_pointers->d_src_bufs,
              src_and_dst_pointers->d_dst_bufs,
              state->d_dst_buf_info,
              stream);

    ++copy_iteration;
    bytes_copied_so_far += bytes_copied;
    starting_buf += num_chunks_to_copy;

    // TODO: ask: the original code synchronized here before making the packed metadata result. Do we need to do that?
    // we are making the packed columns result ahead of time 
    return bytes_copied;
  }

  void perform_regular_copy() {
    //
    // CHUNKED C: If we execute this for every chunk it is mildly wasteful since the "src" info will
    //            already be in place - only the dst buf info will have changed. this is fine
    //            though. the data is small.
    //
    // HtoD src and dest buffers
    src_and_dst_pointers->copy_to_device();
    
    // apply the chunking.
    auto const num_chunks =
      cudf::detail::make_counting_transform_iterator(0, num_chunks_func{computed_chunks->chunks.begin()});
    size_type const new_buf_count =
      thrust::reduce(rmm::exec_policy(stream), num_chunks, num_chunks + computed_chunks->chunks.size());

    // perform the copy.
    copy_data_regular(
      computed_chunks->chunks,
      computed_chunks->chunk_offsets,
      num_bufs, 
      new_buf_count,
      0,
      src_and_dst_pointers->d_src_bufs, 
      src_and_dst_pointers->d_dst_bufs, 
      state->d_dst_buf_info, 
      stream);

    ++copy_iteration;
  }

  bool has_next() {
    return !is_empty && copy_iteration < state->h_keys.size(); 
  }

  std::vector<packed_columns::metadata> const& get_packed_metadata() {
    return packed_metadata;
  }

  std::vector<std::pair<packed_columns::metadata, rmm::device_buffer>> make_packed_columns() {
    std::vector<std::pair<packed_columns::metadata, rmm::device_buffer>> result;
    result.reserve(num_partitions);

    for (int idx = 0; idx < num_partitions; ++idx) {
      auto& out_buffer = partition_buf_size_and_dst_buf_info->out_buffers[idx];
      result.push_back(std::make_pair(packed_metadata[idx], std::move(out_buffer)));
    }

    return result;
  }

  rmm::cuda_stream_view stream;
  rmm::mr::device_memory_resource* mr;

  std::size_t num_partitions;
  
  // number of source buffers including children * number of splits
  std::size_t num_bufs;

  // number of source buffers including children
  size_type num_src_bufs;

  std::unique_ptr<packed_partition_buf_size_and_dst_buf_info> partition_buf_size_and_dst_buf_info;
  std::unique_ptr<packed_src_and_dst_pointers> src_and_dst_pointers;

  //
  // State around the iterator pattern
  //

  // the chunk data computed once on initialization
  std::unique_ptr<chunk_infos> computed_chunks;

  // whether the table was empty to begin with
  // TODO: ask: empty columns vs columns but no rows
  bool is_empty;

  // amount of bytes copied so far. This gets updated on subsequent calls to copy
  cudf::size_type bytes_copied_so_far;

  // the current starting buffer. This gets updated on subsequent calls to copy
  cudf::size_type starting_buf;
  
  // amount of buffer chunks we are fitting at most each time we call copy
  cudf::size_type buffs_to_copy;

  // we compute the packed_metata on initialization and store it here
  std::vector<packed_columns::metadata> packed_metadata;
  
  int copy_iteration;

  std::unique_ptr<iteration_state> state;
  };
};  // namespace detail

chunked_contiguous_split::chunked_contiguous_split(cudf::table_view const& input,
                                                   cudf::device_span<uint8_t> const & user_buffer,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  auto num_partitions = 1;
  auto num_src_bufs = count_src_bufs(input.begin(), input.end());
  auto num_bufs = num_src_bufs * num_partitions;
  auto no_splits = std::vector<size_type>();

  auto split_indices_and_src_buf_infos = 
    std::make_unique<detail::packed_split_indices_and_src_buf_info>(
      input, no_splits, num_partitions, num_src_bufs, stream, mr);

  auto partition_buf_size_and_dst_buf_info =
    std::make_unique<detail::packed_partition_buf_size_and_dst_buf_info>(
      num_partitions,
      num_src_bufs,
      num_bufs,
      std::move(split_indices_and_src_buf_infos),
      user_buffer.data(),
      user_buffer.size(),
      stream,
      mr);

  state = std::make_unique<detail::contiguous_split_state>(
    input, no_splits, std::move(partition_buf_size_and_dst_buf_info), stream, mr);
}

// required for the unique_ptr to work with a non-complete type (contiguous_split_state)
chunked_contiguous_split::~chunked_contiguous_split() = default;

bool chunked_contiguous_split::has_next() const {
  return state->has_next();
}

std::size_t chunked_contiguous_split::next() {
  CUDF_FUNC_RANGE()
  return state->perform_chunked_copy();
}

std::vector<packed_columns::metadata> const& 
chunked_contiguous_split::make_packed_columns() const {
  return state->get_packed_metadata();
}

contiguous_split::contiguous_split(
        cudf::table_view const& input,
        std::vector<size_type> const& splits,
        rmm::cuda_stream_view stream,
        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE()
  auto num_partitions = detail::get_num_partitions(splits);
  auto num_src_bufs = count_src_bufs(input.begin(), input.end());
  auto num_bufs = num_src_bufs * num_partitions;

  auto split_indices_and_src_buf_infos = 
    std::make_unique<detail::packed_split_indices_and_src_buf_info>(
      input, splits, num_partitions, num_src_bufs, stream, mr);

  auto partition_buf_size_and_dst_buf_info =
    std::make_unique<detail::packed_partition_buf_size_and_dst_buf_info>(
      num_partitions,
      num_src_bufs,
      num_bufs,
      std::move(split_indices_and_src_buf_infos),
      stream,
      mr);

  state = std::make_unique<detail::contiguous_split_state>(
    input, splits, std::move(partition_buf_size_and_dst_buf_info), stream, mr);
}

contiguous_split::~contiguous_split() = default;

std::vector<std::pair<packed_columns::metadata, rmm::device_buffer>> 
contiguous_split::make_packed_columns() {
  state->perform_regular_copy();
  return state->make_packed_columns();
}

}};  // namespace cudf
