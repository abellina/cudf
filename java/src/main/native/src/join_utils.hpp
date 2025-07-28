/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "cudf_jni_apis.hpp"
#include "jni_compiled_expr.hpp"
#include "jni_utils.hpp"

#include <cudf/join/conditional_join.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <jni.h>

#include <memory>
#include <utility>

namespace cudf {
namespace jni {

/**
 * @brief Convert a pair of cudf gather maps into the form that Java expects
 * 
 * The resulting Java long array contains the following at each index:
 *   0: Size of the left gather map in bytes
 *   1: Device address of the left gather map
 *   2: Host address of the rmm::device_buffer instance that owns the left gather map data
 *   3: Device address of the right gather map
 *   4: Host address of the rmm::device_buffer instance that owns the right gather map data
 */
jlongArray gather_maps_to_java(
  JNIEnv* env,
  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>> maps);

/**
 * @brief Convert a cudf gather map into the form that Java expects
 * 
 * The resulting Java long array contains the following at each index:
 *   0: Size of the gather map in bytes
 *   1: Device address of the gather map
 *   2: Host address of the rmm::device_buffer instance that owns the gather map data
 */
jlongArray gather_map_to_java(JNIEnv* env,
                              std::unique_ptr<rmm::device_uvector<cudf::size_type>> map);

/**
 * @brief Extract size information for mixed joins
 */
std::pair<std::size_t, cudf::device_span<cudf::size_type const>> get_mixed_size_info(
  JNIEnv* env, jlong j_output_row_count, jlong j_matches_view);

/**
 * @brief Generate gather maps needed to manifest the result of an equi-join between two tables.
 */
template <typename T>
jlongArray join_gather_maps(
  JNIEnv* env, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal, T join_func)
{
  JNI_NULL_CHECK(env, j_left_keys, "left_table is null", NULL);
  JNI_NULL_CHECK(env, j_right_keys, "right_table is null", NULL);
  try {
    cudf::jni::auto_set_device(env);
    auto left_keys  = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto right_keys = reinterpret_cast<cudf::table_view const*>(j_right_keys);
    auto nulleq = compare_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    return gather_maps_to_java(env, join_func(*left_keys, *right_keys, nulleq));
  }
  CATCH_STD(env, NULL);
}

/**
 * @brief Generate gather maps needed to manifest the result of an equi-join between a left table and
 * a hash table built from the join's right table.
 */
template <typename T>
jlongArray hash_join_gather_maps(JNIEnv* env,
                                 jlong j_left_keys,
                                 jlong j_right_hash_join,
                                 T join_func)
{
  JNI_NULL_CHECK(env, j_left_keys, "left table is null", NULL);
  JNI_NULL_CHECK(env, j_right_hash_join, "hash join is null", NULL);
  try {
    cudf::jni::auto_set_device(env);
    auto left_keys = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto hash_join = reinterpret_cast<cudf::hash_join const*>(j_right_hash_join);
    return gather_maps_to_java(env, join_func(*left_keys, *hash_join));
  }
  CATCH_STD(env, NULL);
}

/**
 * @brief Generate gather maps needed to manifest the result of a conditional join between two tables.
 */
template <typename T>
jlongArray cond_join_gather_maps(
  JNIEnv* env, jlong j_left_table, jlong j_right_table, jlong j_condition, T join_func)
{
  JNI_NULL_CHECK(env, j_left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, j_right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, j_condition, "condition is null", NULL);
  try {
    cudf::jni::auto_set_device(env);
    auto left_table  = reinterpret_cast<cudf::table_view const*>(j_left_table);
    auto right_table = reinterpret_cast<cudf::table_view const*>(j_right_table);
    auto condition   = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);
    return gather_maps_to_java(
      env, join_func(*left_table, *right_table, condition->get_top_expression()));
  }
  CATCH_STD(env, NULL);
}

/**
 * @brief Generate a gather map needed to manifest the result of a semi/anti join between two tables.
 */
template <typename T>
jlongArray join_gather_single_map(
  JNIEnv* env, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal, T join_func)
{
  JNI_NULL_CHECK(env, j_left_keys, "left_table is null", NULL);
  JNI_NULL_CHECK(env, j_right_keys, "right_table is null", NULL);
  try {
    cudf::jni::auto_set_device(env);
    auto left_keys  = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto right_keys = reinterpret_cast<cudf::table_view const*>(j_right_keys);
    auto nulleq = compare_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    return gather_map_to_java(env, join_func(*left_keys, *right_keys, nulleq));
  }
  CATCH_STD(env, NULL);
}

/**
 * @brief Generate a gather map needed to manifest the result of a conditional semi/anti join
 * between two tables.
 */
template <typename T>
jlongArray cond_join_gather_single_map(
  JNIEnv* env, jlong j_left_table, jlong j_right_table, jlong j_condition, T join_func)
{
  JNI_NULL_CHECK(env, j_left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, j_right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, j_condition, "condition is null", NULL);
  try {
    cudf::jni::auto_set_device(env);
    auto left_table  = reinterpret_cast<cudf::table_view const*>(j_left_table);
    auto right_table = reinterpret_cast<cudf::table_view const*>(j_right_table);
    auto condition   = reinterpret_cast<cudf::jni::ast::compiled_expr*>(j_condition);
    return gather_map_to_java(
      env, join_func(*left_table, *right_table, condition->get_top_expression()));
  }
  CATCH_STD(env, NULL);
}

/**
 * @brief Generate size information for mixed joins
 */
template <typename T>
jlongArray mixed_join_size(JNIEnv* env,
                           jlong j_left_keys,
                           jlong j_right_keys,
                           jlong j_left_condition,
                           jlong j_right_condition,
                           jlong j_condition,
                           jboolean j_nulls_equal,
                           T join_size_func)
{
  JNI_NULL_CHECK(env, j_left_keys, "left keys table is null", 0);
  JNI_NULL_CHECK(env, j_right_keys, "right keys table is null", 0);
  JNI_NULL_CHECK(env, j_left_condition, "left condition table is null", 0);
  JNI_NULL_CHECK(env, j_right_condition, "right condition table is null", 0);
  JNI_NULL_CHECK(env, j_condition, "condition is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const left_keys       = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto const right_keys      = reinterpret_cast<cudf::table_view const*>(j_right_keys);
    auto const left_condition  = reinterpret_cast<cudf::table_view const*>(j_left_condition);
    auto const right_condition = reinterpret_cast<cudf::table_view const*>(j_right_condition);
    auto const condition = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    auto [join_size, matches_per_row] = join_size_func(*left_keys,
                                                       *right_keys,
                                                       *left_condition,
                                                       *right_condition,
                                                       condition->get_top_expression(),
                                                       nulls_equal);
    if (matches_per_row->size() > std::numeric_limits<cudf::size_type>::max()) {
      throw std::runtime_error("Too many values in device buffer to convert into a column");
    }
    auto col_size = static_cast<size_type>(matches_per_row->size());
    auto col_data = matches_per_row->release();
    cudf::jni::native_jlongArray result(env, 2);
    result[0] = static_cast<jlong>(join_size);
    result[1] = ptr_as_jlong(new cudf::column{cudf::data_type{cudf::type_id::INT32},
                                              col_size,
                                              std::move(col_data),
                                              rmm::device_buffer{},
                                              0});
    return result.get_jArray();
  }
  CATCH_STD(env, NULL);
}

/**
 * @brief Generate gather maps for mixed joins
 */
template <typename T>
jlongArray mixed_join_gather_maps(JNIEnv* env,
                                  jlong j_left_keys,
                                  jlong j_right_keys,
                                  jlong j_left_condition,
                                  jlong j_right_condition,
                                  jlong j_condition,
                                  jboolean j_nulls_equal,
                                  T join_func)
{
  JNI_NULL_CHECK(env, j_left_keys, "left keys table is null", 0);
  JNI_NULL_CHECK(env, j_right_keys, "right keys table is null", 0);
  JNI_NULL_CHECK(env, j_left_condition, "left condition table is null", 0);
  JNI_NULL_CHECK(env, j_right_condition, "right condition table is null", 0);
  JNI_NULL_CHECK(env, j_condition, "condition is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const left_keys       = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto const right_keys      = reinterpret_cast<cudf::table_view const*>(j_right_keys);
    auto const left_condition  = reinterpret_cast<cudf::table_view const*>(j_left_condition);
    auto const right_condition = reinterpret_cast<cudf::table_view const*>(j_right_condition);
    auto const condition = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    return gather_maps_to_java(env,
                               join_func(*left_keys,
                                         *right_keys,
                                         *left_condition,
                                         *right_condition,
                                         condition->get_top_expression(),
                                         nulls_equal));
  }
  CATCH_STD(env, NULL);
}

/**
 * @brief Generate a single gather map for mixed joins (semi/anti joins)
 */
template <typename T>
jlongArray mixed_join_gather_single_map(JNIEnv* env,
                                        jlong j_left_keys,
                                        jlong j_right_keys,
                                        jlong j_left_condition,
                                        jlong j_right_condition,
                                        jlong j_condition,
                                        jboolean j_nulls_equal,
                                        T join_func)
{
  JNI_NULL_CHECK(env, j_left_keys, "left keys table is null", 0);
  JNI_NULL_CHECK(env, j_right_keys, "right keys table is null", 0);
  JNI_NULL_CHECK(env, j_left_condition, "left condition table is null", 0);
  JNI_NULL_CHECK(env, j_right_condition, "right condition table is null", 0);
  JNI_NULL_CHECK(env, j_condition, "condition is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const left_keys       = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto const right_keys      = reinterpret_cast<cudf::table_view const*>(j_right_keys);
    auto const left_condition  = reinterpret_cast<cudf::table_view const*>(j_left_condition);
    auto const right_condition = reinterpret_cast<cudf::table_view const*>(j_right_condition);
    auto const condition = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    return gather_map_to_java(env,
                              join_func(*left_keys,
                                        *right_keys,
                                        *left_condition,
                                        *right_condition,
                                        condition->get_top_expression(),
                                        nulls_equal));
  }
  CATCH_STD(env, NULL);
}

}  // namespace jni
}  // namespace cudf 