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

#include "cudf_jni_apis.hpp"

extern "C" {
JNIEXPORT void JNICALL Java_ai_rapids_cudf_ChunkedContiguousSplit_destroyChunkedContiguousSplit(
    JNIEnv *env, jclass, jlong chunked_contig_split) {
  try {
    cudf::jni::auto_set_device(env);
    auto cs = reinterpret_cast<cudf::chunked_contiguous_split*>(chunked_contig_split);
    delete cs;
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ChunkedContiguousSplit_chunkedContiguousSplitSize(
    JNIEnv *env, jclass, jlong chunked_contig_split) {
  try {
    cudf::jni::auto_set_device(env);
    auto cs = reinterpret_cast<cudf::chunked_contiguous_split*>(chunked_contig_split);
    return cs->get_total_contiguous_size();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_ChunkedContiguousSplit_chunkedContiguousSplitHasNext(
    JNIEnv *env, jclass, jlong chunked_contig_split) {
  try {
    cudf::jni::auto_set_device(env);
    auto cs = reinterpret_cast<cudf::chunked_contiguous_split*>(chunked_contig_split);
    return cs->has_next();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ChunkedContiguousSplit_chunkedContiguousSplitNext(
    JNIEnv *env, jclass, jlong chunked_contig_split, jlong user_ptr, jlong user_ptr_size) {
  try {
    cudf::jni::auto_set_device(env);
    auto cs = reinterpret_cast<cudf::chunked_contiguous_split*>(chunked_contig_split);
    auto user_buffer_span = cudf::device_span<uint8_t>(
      reinterpret_cast<uint8_t*>(user_ptr), 
      static_cast<std::size_t>(user_ptr_size));
    return cs->next(user_buffer_span);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_ChunkedContiguousSplit_chunkedContiguousSplitMakePackedColumns(
    JNIEnv *env, jclass, jlong chunked_contig_split) {
  try {
    cudf::jni::auto_set_device(env);
    auto cs = reinterpret_cast<cudf::chunked_contiguous_split*>(chunked_contig_split);
    std::unique_ptr<cudf::packed_columns::metadata> result = cs->make_packed_columns();
    return cudf::jni::packed_column_metadata_from(env, std::move(result));
  }
  CATCH_STD(env, NULL);
}

} // extern "C"
