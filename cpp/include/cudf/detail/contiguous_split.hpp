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

#include <cudf/contiguous_split.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::contiguous_split
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 **/
std::vector<packed_table> contiguous_split(cudf::table_view const& input,
                                           std::vector<size_type> const& splits,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr);

/**
 * @copydoc cudf::pack
 *
 * @param stream Optional CUDA stream on which to execute kernels
 **/
packed_columns pack(cudf::table_view const& input,
                    rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource* mr);

// opaque implementation of `metadata_builder` since it needs to use
// `serialized_column`, which is only defined in pack.cpp
class metadata_builder_impl;

/**
 * @brief Helper class that creates packed_columns::metadata
 *
 * This class is an interface to the opaque metadata that is used to
 * describe `contiguous_split` and `pack` results.
 */
class metadata_builder {
 public:
  /** 
   * @brief Construct a new metadata_builder
   *
   * @param num_root_columns is the number of top-level columns (non-nested)
   */
  explicit metadata_builder(size_type num_root_columns);

  /**
   * @brief Add a column to this metadata builder
   *
   * @param col_type column data type
   * @param col_size column row count
   * @param col_null_count column null count
   * @param data_offset data offset from the column's base ptr,
   *                    or -1 for an empty column
   * @param null_mask_offset null mask offset from the column's base ptr,
   *                    or -1 for a column that isn't nullable
   * @param num_children number of chilren columns
   */
  void add_column_to_meta(data_type col_type,
                          size_type col_size,
                          size_type col_null_count,
                          int64_t data_offset,
                          int64_t null_mask_offset,
                          size_type num_children);

  /**
   * @brief Builds the opaque metadata for all added columns
   * @returns A packed_columns::metadata instance with serialized
   *          column metadata
   */
  packed_columns::metadata build();

 private:
  std::unique_ptr<metadata_builder_impl> impl;
};

}  // namespace detail
}  // namespace cudf
