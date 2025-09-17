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
#include "jni_utils.hpp"

#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <vector>
#include <algorithm>

namespace {
struct JoinMatchContextHolder {
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> match_counts;
  cudf::table_view left_view;
};
} // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_JoinMatchContext_createFromHashJoinInner(
  JNIEnv* env, jclass, jlong hash_join_handle, jlong left_table_view_handle)
{
  JNI_ARG_CHECK(env, hash_join_handle != 0, "hash_join handle is null", 0);
  JNI_ARG_CHECK(env, left_table_view_handle != 0, "left table_view handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto* hash_join = reinterpret_cast<cudf::hash_join const*>(hash_join_handle);
    auto const& left_view = *reinterpret_cast<cudf::table_view const*>(left_table_view_handle);

    auto ctx = hash_join->inner_join_match_context(left_view);

    auto holder = std::make_unique<JoinMatchContextHolder>();
    holder->left_view = ctx._left_table;
    holder->match_counts = std::move(ctx._match_counts);

    return reinterpret_cast<jlong>(holder.release());
  } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_JoinMatchContext_createFromHashJoinLeft(
  JNIEnv* env, jclass, jlong hash_join_handle, jlong left_table_view_handle)
{
  JNI_ARG_CHECK(env, hash_join_handle != 0, "hash_join handle is null", 0);
  JNI_ARG_CHECK(env, left_table_view_handle != 0, "left table_view handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto* hash_join = reinterpret_cast<cudf::hash_join const*>(hash_join_handle);
    auto const& left_view = *reinterpret_cast<cudf::table_view const*>(left_table_view_handle);

    auto ctx = hash_join->left_join_match_context(left_view);

    auto holder = std::make_unique<JoinMatchContextHolder>();
    holder->left_view = ctx._left_table;
    holder->match_counts = std::move(ctx._match_counts);

    return reinterpret_cast<jlong>(holder.release());
  } CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_JoinMatchContext_createFromHashJoinFull(
  JNIEnv* env, jclass, jlong hash_join_handle, jlong left_table_view_handle)
{
  JNI_ARG_CHECK(env, hash_join_handle != 0, "hash_join handle is null", 0);
  JNI_ARG_CHECK(env, left_table_view_handle != 0, "left table_view handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto* hash_join = reinterpret_cast<cudf::hash_join const*>(hash_join_handle);
    auto const& left_view = *reinterpret_cast<cudf::table_view const*>(left_table_view_handle);

    auto ctx = hash_join->full_join_match_context(left_view);

    auto holder = std::make_unique<JoinMatchContextHolder>();
    holder->left_view = ctx._left_table;
    holder->match_counts = std::move(ctx._match_counts);

    return reinterpret_cast<jlong>(holder.release());
  } CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_JoinMatchContext_exportMatchCounts(
  JNIEnv* env, jclass, jlong ctx_handle)
{
  JNI_ARG_CHECK(env, ctx_handle != 0, "JoinMatchContext handle is null", nullptr);
  try {
    cudf::jni::auto_set_device(env);
    auto* holder = reinterpret_cast<JoinMatchContextHolder*>(ctx_handle);

    auto const n = holder->match_counts ? static_cast<jsize>(holder->match_counts->size()) : 0;
    cudf::jni::native_jlongArray out(env, n);

    if (n > 0) {
      std::vector<int32_t> host_counts(n);
      CUDF_CUDA_TRY(cudaMemcpy(host_counts.data(),
                               holder->match_counts->data(),
                               sizeof(int32_t) * n,
                               cudaMemcpyDeviceToHost));
      std::transform(host_counts.begin(), host_counts.end(), out.begin(),
                     [](int32_t v) { return static_cast<jlong>(v); });

      holder->match_counts.reset();
    }

    return out.get_jArray();
  } CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_JoinMatchContext_closeNative(
  JNIEnv* env, jclass, jlong ctx_handle)
{
  if (ctx_handle == 0) { return; }
  try {
    cudf::jni::auto_set_device(env);
    auto* holder = reinterpret_cast<JoinMatchContextHolder*>(ctx_handle);
    delete holder;
  } CATCH_STD(env, );
}

} // extern "C"
