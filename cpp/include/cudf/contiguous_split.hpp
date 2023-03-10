/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace cudf {

/**
 * @addtogroup column_copy
 * @{
 * @file
 * @brief Table APIs for contiguous_split, pack, unpack, and metadadata
 */

/**
 * @brief Column data in a serialized format
 *
 * @ingroup copy_split
 *
 * Contains data from an array of columns in two contiguous buffers: one on host, which contains
 * table metadata and one on device which contains the table data.
 */
struct packed_columns {
  /**
   * @brief Host-side metadata buffer used for reconstructing columns via unpack.
   *
   * @ingroup copy_split
   */
  struct metadata {
    metadata() = default;

    /**
     * @brief Construct a new metadata object
     *
     * @param v Host-side buffer containing metadata
     */
    metadata(std::vector<uint8_t>&& v) : data_(std::move(v)) {}

    /**
     * @brief Returns pointer to the host-side metadata buffer data
     *
     * @return Pointer to the host-side metadata buffer
     */
    [[nodiscard]] uint8_t const* data() const { return data_.data(); }

    /**
     * @brief Returns size of the metadata buffer
     *
     * @return Size of the metadata buffer
     */
    [[nodiscard]] size_t size() const { return data_.size(); }

   private:
    std::vector<uint8_t> data_;
  };

  packed_columns()
    : metadata_(std::make_unique<metadata>()), gpu_data(std::make_unique<rmm::device_buffer>())
  {
  }

  /**
   * @brief Construct a new packed columns object
   *
   * @param md Host-side metadata buffer
   * @param gd Device-side data buffer
   */
  packed_columns(std::unique_ptr<metadata>&& md, std::unique_ptr<rmm::device_buffer>&& gd)
    : metadata_(std::move(md)), gpu_data(std::move(gd))
  {
  }

  std::unique_ptr<metadata> metadata_;           ///< Host-side metadata buffer
  std::unique_ptr<rmm::device_buffer> gpu_data;  ///< Device-side data buffer
};

namespace detail {
  class metadata_builder_impl;
}

class metadata_builder {
  public:
    explicit metadata_builder(size_type num_root_columns);
    ~metadata_builder();

    void add_column_to_meta(data_type col_type,
                            size_type col_size,
                            size_type col_null_count,
                            int64_t data_offset,
                            int64_t null_mask_offset,
                            size_type num_children);

    packed_columns::metadata build();

  private:
    detail::metadata_builder_impl* impl;
};

/**
 * @brief The result(s) of a cudf::contiguous_split
 *
 * @ingroup copy_split
 *
 * Each table_view resulting from a split operation performed by contiguous_split,
 * will be returned wrapped in a `packed_table`.  The table_view and internal
 * column_views in this struct are not owned by a top level cudf::table or cudf::column.
 * The backing memory and metadata is instead owned by the `data` field and is in one
 * contiguous block.
 *
 * The user is responsible for assuring that the `table` or any derived table_views do
 * not outlive the memory owned by `data`
 */
struct packed_table {
  cudf::table_view table;  ///< Result table_view of a cudf::contiguous_split
  packed_columns data;     ///< Column data owned
};

/**
 * @brief Performs a deep-copy split of a `table_view` into a set of `table_view`s into a single
 * contiguous block of memory.
 *
 * @ingroup copy_split
 *
 * The memory for the output views is allocated in a single contiguous `rmm::device_buffer` returned
 * in the `packed_table`. There is no top-level owning table.
 *
 * The returned views of `input` are constructed from a vector of indices, that indicate
 * where each split should occur. The `i`th returned `table_view` is sliced as
 * `[0, splits[i])` if `i`=0, else `[splits[i], input.size())` if `i` is the last view and
 * `[splits[i-1], splits[i]]` otherwise.
 *
 * For all `i` it is expected `splits[i] <= splits[i+1] <= input.size()`
 * For a `splits` size N, there will always be N+1 splits in the output
 *
 * @note It is the caller's responsibility to ensure that the returned views
 * do not outlive the viewed device memory contained in the `all_data` field of the
 * returned packed_table.
 *
 * @code{.pseudo}
 * Example:
 * input:   [{10, 12, 14, 16, 18, 20, 22, 24, 26, 28},
 *           {50, 52, 54, 56, 58, 60, 62, 64, 66, 68}]
 * splits:  {2, 5, 9}
 * output:  [{{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}},
 *           {{50, 52}, {54, 56, 58}, {60, 62, 64, 66}, {68}}]
 * @endcode
 *
 *
 * @throws cudf::logic_error if `splits` has end index > size of `input`.
 * @throws cudf::logic_error When the value in `splits` is not in the range [0, input.size()).
 * @throws cudf::logic_error When the values in the `splits` are 'strictly decreasing'.
 *
 * @param input View of a table to split
 * @param splits A vector of indices where the view will be split
 * @param[in] mr Device memory resource used to allocate the returned result's device memory
 * @return The set of requested views of `input` indicated by the `splits` and the viewed memory
 * buffer.
 */
std::vector<packed_table> contiguous_split(
  cudf::table_view const& input,
  std::vector<size_type> const& splits,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

namespace detail {
  class metadata_builder_impl;
}

class metadata_builder {
  public:
    explicit metadata_builder(size_type num_root_columns);
    ~metadata_builder();

    void add_column_to_meta(column_view const& col, int64_t data_offset, int64_t null_mask_offset);

    void add_column_to_meta(data_type col_type,
                            size_type col_size,
                            size_type col_null_count,
                            int64_t data_offset,
                            int64_t null_mask_offset,
                            size_type num_children);

    packed_columns::metadata build();

  private:
    detail::metadata_builder_impl* impl;
};

namespace detail {
  struct contiguous_split_state;
};

class chunked_contiguous_split {
  public:
    explicit chunked_contiguous_split(
        cudf::table_view const& input,
        std::size_t user_buffer_size,
        rmm::cuda_stream_view stream,
        rmm::mr::device_memory_resource* mr);

    ~chunked_contiguous_split();

    [[nodiscard]] std::size_t get_total_contiguous_size() const;

    [[nodiscard]] bool has_next() const;

    [[nodiscard]] std::size_t next(cudf::device_span<uint8_t> const& user_buffer);

    [[nodiscard]] std::unique_ptr<packed_columns::metadata> make_packed_columns() const;

  private:
    // internal state of contiguous split
    std::unique_ptr<detail::contiguous_split_state> state;
  };

std::unique_ptr<chunked_contiguous_split> make_chunked_contiguous_split(
  cudf::table_view const& input,
  std::size_t user_buffer_size,
  rmm::mr::device_memory_resource* mr);

/**
 * @brief Deep-copy a `table_view` into a serialized contiguous memory format
 *
 * The metadata from the `table_view` is copied into a host vector of bytes and the data from the
 * `table_view` is copied into a `device_buffer`. Pass the output of this function into
 * `cudf::unpack` to deserialize.
 *
 * @param input View of the table to pack
 * @param[in] mr Optional, The resource to use for all returned device allocations
 * @return packed_columns A struct containing the serialized metadata and data in contiguous host
 *         and device memory respectively
 */
packed_columns pack(cudf::table_view const& input,
                    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Produce the metadata used for packing a table stored in a contiguous buffer.
 *
 * The metadata from the `table_view` is copied into a host vector of bytes which can be used to
 * construct a `packed_columns` or `packed_table` structure.  The caller is responsible for
 * guaranteeing that that all of the columns in the table point into `contiguous_buffer`.
 *
 * @param table View of the table to pack
 * @param contiguous_buffer A contiguous buffer of device memory which contains the data referenced
 * by the columns in `table`
 * @param buffer_size The size of `contiguous_buffer`
 * @return Vector of bytes representing the metadata used to `unpack` a packed_columns struct
 */
packed_columns::metadata pack_metadata(table_view const& table,
                                       uint8_t const* contiguous_buffer,
                                       size_t buffer_size);

/**
 * @brief Deserialize the result of `cudf::pack`
 *
 * Converts the result of a serialized table into a `table_view` that points to the data stored in
 * the contiguous device buffer contained in `input`.
 *
 * It is the caller's responsibility to ensure that the `table_view` in the output does not outlive
 * the data in the input.
 *
 * No new device memory is allocated in this function.
 *
 * @param input The packed columns to unpack
 * @return The unpacked `table_view`
 */
table_view unpack(packed_columns const& input);

/**
 * @brief Deserialize the result of `cudf::pack`
 *
 * Converts the result of a serialized table into a `table_view` that points to the data stored in
 * the contiguous device buffer contained in `gpu_data` using the metadata contained in the host
 * buffer `metadata`.
 *
 * It is the caller's responsibility to ensure that the `table_view` in the output does not outlive
 * the data in the input.
 *
 * No new device memory is allocated in this function.
 *
 * @param metadata The host-side metadata buffer resulting from the initial pack() call
 * @param gpu_data The device-side contiguous buffer storing the data that will be referenced by
 * the resulting `table_view`
 * @return The unpacked `table_view`
 */
table_view unpack(uint8_t const* metadata, uint8_t const* gpu_data);

/** @} */
}  // namespace cudf
