/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_PackedColumnMetadata_createMetadataDirectBuffer(
  JNIEnv* env, jclass, jlong j_metadata_ptr)
{
  JNI_NULL_CHECK(env, j_metadata_ptr, "metadata is null", nullptr);
  try {
    auto metadata = reinterpret_cast<std::vector<uint8_t>*>(j_metadata_ptr);
    return env->NewDirectByteBuffer(const_cast<uint8_t*>(metadata->data()), metadata->size());
  }
  CATCH_STD(env, nullptr);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_PackedColumnMetadata_createMetadataHandle(
  JNIEnv* env, jclass, jobject j_metadata)
{
  try {
    auto vec_ptr = new std::vector<uint8_t>();
    size_t size = env->GetDirectBufferCapacity(j_metadata);
    vec_ptr->reserve(size);
    auto data = reinterpret_cast<uint8_t*>(env->GetDirectBufferAddress(j_metadata));
    std::copy(data, data + size, std::back_inserter(*vec_ptr));
    return reinterpret_cast<uint64_t>(vec_ptr);
  }
  CATCH_STD(env, 0);
}


JNIEXPORT void JNICALL Java_ai_rapids_cudf_PackedColumnMetadata_closeMetadata(JNIEnv* env,
                                                                              jclass,
                                                                              jlong j_metadata_ptr)
{
  JNI_NULL_CHECK(env, j_metadata_ptr, "metadata is null", );
  try {
    auto metadata = reinterpret_cast<std::vector<uint8_t>*>(j_metadata_ptr);
    delete metadata;
  }
  CATCH_STD(env, );
}

}  // extern "C"
