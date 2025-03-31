/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

namespace {

#define CONTIGUOUS_TABLE_CLASS                  "ai/rapids/cudf/ContiguousTable"
#define CONTIGUOUS_TABLE_FACTORY_SIG(param_sig) "(" param_sig ")L" CONTIGUOUS_TABLE_CLASS ";"

#define CONTIGUOUS_TABLES_CLASS                  "ai/rapids/cudf/ContiguousTables"
#define CONTIGUOUS_TABLES_FACTORY_SIG(param_sig) "(" param_sig ")L" CONTIGUOUS_TABLES_CLASS ";"

jclass Contiguous_table_jclass;
jmethodID From_packed_table_method;
jclass Contiguous_tables_jclass;
jmethodID From_packed_tables_method;

#define GROUP_BY_RESULT_CLASS "ai/rapids/cudf/ContigSplitGroupByResult"
jclass Contig_split_group_by_result_jclass;
jfieldID Contig_split_group_by_result_groups_field;
jfieldID Contig_split_group_by_result_uniq_key_columns_field;

}  // anonymous namespace

namespace cudf {
namespace jni {

bool cache_contiguous_table_jni(JNIEnv* env)
{
  {
    jclass cls = env->FindClass(CONTIGUOUS_TABLE_CLASS);
    if (cls == nullptr) { return false; }

    From_packed_table_method =
      env->GetStaticMethodID(cls, "fromPackedTable", CONTIGUOUS_TABLE_FACTORY_SIG("JJJJJ"));
    if (From_packed_table_method == nullptr) { return false; }

    // Convert local reference to global so it cannot be garbage collected.
    Contiguous_table_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
    if (Contiguous_table_jclass == nullptr) { return false; }
  }

  {
    jclass cls = env->FindClass(CONTIGUOUS_TABLES_CLASS);
    if (cls == nullptr) { return false; }

    From_packed_tables_method =
      env->GetStaticMethodID(cls, "fromPackedTables", CONTIGUOUS_TABLES_FACTORY_SIG("[J[J[J[JJJJ"));
    if (From_packed_tables_method == nullptr) { return false; }

    // Convert local reference to global so it cannot be garbage collected.
    Contiguous_tables_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
    if (Contiguous_tables_jclass == nullptr) { return false; }
  }

  return true;
}

void release_contiguous_table_jni(JNIEnv* env)
{
  Contiguous_table_jclass = cudf::jni::del_global_ref(env, Contiguous_table_jclass);
  Contiguous_tables_jclass = cudf::jni::del_global_ref(env, Contiguous_tables_jclass);
}

bool cache_contig_split_group_by_result_jni(JNIEnv* env)
{
  jclass cls = env->FindClass(GROUP_BY_RESULT_CLASS);
  if (cls == nullptr) { return false; }

  Contig_split_group_by_result_groups_field =
    env->GetFieldID(cls, "groups", "[Lai/rapids/cudf/ContiguousTable;");
  if (Contig_split_group_by_result_groups_field == nullptr) { return false; }
  Contig_split_group_by_result_uniq_key_columns_field =
    env->GetFieldID(cls, "uniqKeyColumns", "[J");
  if (Contig_split_group_by_result_uniq_key_columns_field == nullptr) { return false; }

  // Convert local reference to global so it cannot be garbage collected.
  Contig_split_group_by_result_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (Contig_split_group_by_result_jclass == nullptr) { return false; }
  return true;
}

void release_contig_split_group_by_result_jni(JNIEnv* env)
{
  Contig_split_group_by_result_jclass = del_global_ref(env, Contig_split_group_by_result_jclass);
}

jobject contig_split_group_by_result_from(JNIEnv* env, jobjectArray& groups)
{
  jobject gbr = env->AllocObject(Contig_split_group_by_result_jclass);
  env->SetObjectField(gbr, Contig_split_group_by_result_groups_field, groups);
  return gbr;
}

jobject contig_split_group_by_result_from(JNIEnv* env,
                                          jobjectArray& groups,
                                          jlongArray& uniq_key_columns)
{
  jobject gbr = env->AllocObject(Contig_split_group_by_result_jclass);
  env->SetObjectField(gbr, Contig_split_group_by_result_groups_field, groups);
  env->SetObjectField(gbr, Contig_split_group_by_result_uniq_key_columns_field, uniq_key_columns);
  return gbr;
}



jobject contiguous_tables_contiguously(
  JNIEnv* env, 
  std::vector<cudf::packed_table>& result, 
  rmm::device_buffer* buff
)
{
  auto base_addr = reinterpret_cast<uint64_t>(buff->data());
  auto data_length = buff->size();
  jlong rmm_buffer_address = reinterpret_cast<jlong>(buff);

  auto num_splits = result.size();
  cudf::jni::native_jlongArray jmetadata_addresses(env, num_splits);
  cudf::jni::native_jlongArray jdata_offsets(env, num_splits);
  cudf::jni::native_jlongArray jdata_sizes(env, num_splits);
  cudf::jni::native_jlongArray jrow_counts(env, num_splits);

  for (size_t i = 0; i < result.size(); i++) {
    auto& split = result[i].data;
    jmetadata_addresses[i] = reinterpret_cast<jlong>(split.metadata.get());
    jdata_offsets[i] = reinterpret_cast<jlong>(split.gpu_data->data()) - base_addr;
    jdata_sizes[i] = static_cast<jlong>(split.gpu_data->size());
    jrow_counts[i] = result[i].table.num_rows();
  }

  jmetadata_addresses.commit();
  jdata_offsets.commit();
  jdata_sizes.commit();
  jrow_counts.commit();

  jobject res = env->CallStaticObjectMethod(Contiguous_tables_jclass,
                                     From_packed_tables_method,
                                     jmetadata_addresses.get_jArray(),
                                     jdata_offsets.get_jArray(),
                                     jdata_sizes.get_jArray(),
                                     jrow_counts.get_jArray(),
                                     base_addr,
                                     data_length,
                                     rmm_buffer_address);
  for (size_t i = 0; i < result.size(); i++) {
    auto& split = result[i].data;
    split.metadata.release();
    split.gpu_data.release();
  }
  return res;
}

jobject contiguous_table_from(JNIEnv* env, cudf::packed_columns& split, long row_count)
{
  jlong metadata_address   = reinterpret_cast<jlong>(split.metadata.get());
  jlong data_address       = reinterpret_cast<jlong>(split.gpu_data->data());
  jlong data_size          = static_cast<jlong>(split.gpu_data->size());
  jlong rmm_buffer_address = reinterpret_cast<jlong>(split.gpu_data.get());

  jobject contig_table_obj = env->CallStaticObjectMethod(Contiguous_table_jclass,
                                                         From_packed_table_method,
                                                         metadata_address,
                                                         data_address,
                                                         data_size,
                                                         rmm_buffer_address,
                                                         row_count);

  if (contig_table_obj != nullptr) {
    split.metadata.release();
    split.gpu_data.release();
  }

  return contig_table_obj;
}

native_jobjectArray<jobject> contiguous_table_array(JNIEnv* env, jsize length)
{
  return native_jobjectArray<jobject>(
    env, env->NewObjectArray(length, Contiguous_table_jclass, nullptr));
}

}  // namespace jni
}  // namespace cudf

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ContiguousTable_createPackedMetadata(
  JNIEnv* env, jclass, jlong j_table, jlong j_buffer_addr, jlong j_buffer_length)
{
  JNI_NULL_CHECK(env, j_table, "input table is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto table        = reinterpret_cast<cudf::table_view const*>(j_table);
    auto data_addr    = reinterpret_cast<uint8_t const*>(j_buffer_addr);
    auto data_size    = static_cast<size_t>(j_buffer_length);
    auto metadata_ptr = new std::vector<uint8_t>(cudf::pack_metadata(*table, data_addr, data_size));
    return reinterpret_cast<jlong>(metadata_ptr);
  }
  CATCH_STD(env, 0);
}

}  // extern "C"
