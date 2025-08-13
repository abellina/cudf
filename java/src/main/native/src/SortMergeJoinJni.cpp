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

#include "cudf_jni_apis.hpp"
#include "join_utils.hpp"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/sort_merge_join.hpp>

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_SortMergeJoin_create(JNIEnv* env,
                                                                 jclass,
                                                                 jlong j_build_table,
                                                                 jboolean j_build_table_sorted,
                                                                 jboolean j_compare_nulls_equal)
{
  JNI_NULL_CHECK(env, j_build_table, "build table handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto build_table = reinterpret_cast<cudf::table_view const*>(j_build_table);
    auto null_equality = j_compare_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    auto join_obj = new cudf::sort_merge_join(
        *build_table, 
        j_build_table_sorted ? cudf::sorted::YES : cudf::sorted::NO,
        null_equality);
    return reinterpret_cast<jlong>(join_obj);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_SortMergeJoin_destroy(JNIEnv* env, jclass, jlong j_handle)
{
  try {
    cudf::jni::auto_set_device(env);
    auto join_obj = reinterpret_cast<cudf::sort_merge_join*>(j_handle);
    delete join_obj;
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_SortMergeJoin_makePartitionContext(
  JNIEnv* env, jclass, jlong j_join_obj, jlong j_stream_table, jboolean j_stream_table_sorted)
{
  JNI_NULL_CHECK(env, j_join_obj, "join object handle is null", 0);
  JNI_NULL_CHECK(env, j_stream_table, "stream table handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto stream_table = reinterpret_cast<cudf::table_view const*>(j_stream_table);
    auto join_obj = reinterpret_cast<cudf::sort_merge_join*>(j_join_obj);
    auto context = join_obj->inner_join_match_context(*stream_table, 
        j_stream_table_sorted ? cudf::sorted::YES : cudf::sorted::NO);
    auto partition_ctx = new cudf::sort_merge_join::partition_context{
      std::move(context), 
      0, 
      0
    };
    return reinterpret_cast<jlong>(partition_ctx);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_SortMergeJoin_destroyPartitionContext(JNIEnv* env, jclass, jlong j_partition_context)
{
  try {
    cudf::jni::auto_set_device(env);
    auto context = reinterpret_cast<cudf::sort_merge_join::partition_context*>(j_partition_context);
    delete context;
  }
  CATCH_STD(env, );
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_SortMergeJoin_getNumRows(
  JNIEnv* env, jclass, jlong j_partition_context)
{
  JNI_NULL_CHECK(env, j_partition_context, "partition context handle is null", nullptr);
  try {
    cudf::jni::auto_set_device(env);
    auto context = reinterpret_cast<cudf::sort_merge_join::partition_context*>(j_partition_context);
    auto host_vec = cudf::detail::make_std_vector(*context->left_table_context._match_counts, cudf::get_default_stream());
    cudf::jni::native_jlongArray ret(env, host_vec.size());
    for (size_t i = 0; i < host_vec.size(); ++i) {
      ret[i] = static_cast<jlong>(host_vec[i]);
    }
    return ret.get_jArray();
  }
  CATCH_STD(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_SortMergeJoin_partitionedJoin(
  JNIEnv* env, jclass, jlong j_join_obj, jlong j_partition_context, jlong start_row, jlong num_rows)
{
  JNI_NULL_CHECK(env, j_join_obj, "join object handle is null", nullptr);
  JNI_NULL_CHECK(env, j_partition_context, "partition context handle is null", nullptr);
  try {
    cudf::jni::auto_set_device(env);
    auto join_obj = reinterpret_cast<cudf::sort_merge_join*>(j_join_obj);
    auto context = reinterpret_cast<cudf::sort_merge_join::partition_context*>(j_partition_context);
    context->left_start_idx = static_cast<cudf::size_type>(start_row);
    context->left_end_idx = context->left_start_idx + static_cast<cudf::size_type>(num_rows);
    auto left_right_indices = join_obj->partitioned_inner_join(*context);
    return cudf::jni::gather_maps_to_java(env, std::move(left_right_indices));
  }
  CATCH_STD(env, nullptr);
}

}  // extern "C" 