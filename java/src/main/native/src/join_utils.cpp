/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "join_utils.hpp"

#include <cudf/column/column.hpp>
#include <rmm/device_buffer.hpp>

namespace cudf {
namespace jni {

jlongArray gather_maps_to_java(
  JNIEnv* env,
  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>> maps)
{
  // release the underlying device buffer to Java
  auto left_map_buffer  = std::make_unique<rmm::device_buffer>(maps.first->release());
  auto right_map_buffer = std::make_unique<rmm::device_buffer>(maps.second->release());
  cudf::jni::native_jlongArray result(env, 5);
  result[0] = static_cast<jlong>(left_map_buffer->size());
  result[1] = ptr_as_jlong(left_map_buffer->data());
  result[2] = release_as_jlong(left_map_buffer);
  result[3] = ptr_as_jlong(right_map_buffer->data());
  result[4] = release_as_jlong(right_map_buffer);
  return result.get_jArray();
}

jlongArray gather_map_to_java(JNIEnv* env,
                              std::unique_ptr<rmm::device_uvector<cudf::size_type>> map)
{
  // release the underlying device buffer to Java
  cudf::jni::native_jlongArray result(env, 3);
  result[0]              = static_cast<jlong>(map->size() * sizeof(cudf::size_type));
  auto gather_map_buffer = std::make_unique<rmm::device_buffer>(map->release());
  result[1]              = ptr_as_jlong(gather_map_buffer->data());
  result[2]              = release_as_jlong(gather_map_buffer);
  return result.get_jArray();
}

std::pair<std::size_t, cudf::device_span<cudf::size_type const>> get_mixed_size_info(
  JNIEnv* env, jlong j_output_row_count, jlong j_matches_view)
{
  auto const row_count = static_cast<std::size_t>(j_output_row_count);
  auto const matches   = reinterpret_cast<cudf::column_view const*>(j_matches_view);
  return std::make_pair(row_count,
                        cudf::device_span<cudf::size_type const>(
                          matches->template data<cudf::size_type>(), matches->size()));
}

}  // namespace jni
}  // namespace cudf 