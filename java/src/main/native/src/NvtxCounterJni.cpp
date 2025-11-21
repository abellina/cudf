/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jni_utils.hpp"
#include "nvtx_common.hpp"

#include <nvtx3/nvtx3.hpp>
#include <nvtx3/nvToolsExtCounters.h>

namespace {

/**
 * Helper function to get the Java domain handle for NVTX operations.
 */
nvtxDomainHandle_t get_java_domain()
{
  return nvtx3::domain::get<cudf::jni::java_domain>();
}

}  // anonymous namespace

extern "C" {

/**
 * Register a counter with the NVTX profiler.
 * 
 * @param env JNI environment
 * @param clazz Java class
 * @param j_name Counter name
 * @param j_description Counter description (may be null)
 * @param schema_id Schema ID for counter data layout
 * @param scope_id Scope identifier
 * @param counter_id Static counter ID or NVTX_COUNTER_ID_NONE
 * @return Unique counter ID
 */
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_NvtxCounter_register(JNIEnv* env,
                                                                  jclass clazz,
                                                                  jstring j_name,
                                                                  jstring j_description,
                                                                  jlong schema_id,
                                                                  jlong scope_id,
                                                                  jlong counter_id)
{
  JNI_TRY
  {
    if (j_name == nullptr) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "Counter name cannot be null", 0);
    }

    cudf::jni::native_jstring name(env, j_name);
    cudf::jni::native_jstring description(env, j_description);

    nvtxCounterAttr_t attr{};
    attr.structSize   = sizeof(nvtxCounterAttr_t);
    attr.name         = name.get();
    attr.description  = (j_description != nullptr) ? description.get() : nullptr;
    attr.schemaId     = static_cast<uint64_t>(schema_id);
    attr.scopeId      = static_cast<uint64_t>(scope_id);
    attr.counterId    = static_cast<uint64_t>(counter_id);
    attr.semantics    = nullptr;  // Using default semantics

    uint64_t id = nvtxCounterRegister(get_java_domain(), &attr);
    return static_cast<jlong>(id);
  }
  JNI_CATCH(env, 0);
}

/**
 * Unregister a counter.
 * Note: NVTX counter extension does not provide an explicit unregister API.
 * Counters exist for the lifetime of the application. This is a no-op for
 * compatibility with the Java AutoCloseable pattern.
 * 
 * @param env JNI environment
 * @param clazz Java class
 * @param counter_id Counter identifier
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_NvtxCounter_unregister(JNIEnv* env,
                                                                   jclass clazz,
                                                                   jlong counter_id)
{
  // No-op: NVTX counters do not have an explicit unregister API.
  // The Java side will simply stop sampling this counter.
}

/**
 * Sample a 64-bit integer counter.
 * 
 * @param env JNI environment
 * @param clazz Java class
 * @param counter_id Counter identifier
 * @param value Counter value
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_NvtxCounter_sampleInt64Native(JNIEnv* env,
                                                                           jclass clazz,
                                                                           jlong counter_id,
                                                                           jlong value)
{
  JNI_TRY
  {
    nvtxCounterSampleInt64(get_java_domain(),
                           static_cast<uint64_t>(counter_id),
                           static_cast<int64_t>(value));
  }
  JNI_CATCH(env, );
}

/**
 * Sample a 64-bit floating-point counter.
 * 
 * @param env JNI environment
 * @param clazz Java class
 * @param counter_id Counter identifier
 * @param value Counter value
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_NvtxCounter_sampleFloat64Native(JNIEnv* env,
                                                                             jclass clazz,
                                                                             jlong counter_id,
                                                                             jdouble value)
{
  JNI_TRY
  {
    nvtxCounterSampleFloat64(get_java_domain(),
                             static_cast<uint64_t>(counter_id),
                             static_cast<double>(value));
  }
  JNI_CATCH(env, );
}

/**
 * Sample a counter without a value.
 * 
 * @param env JNI environment
 * @param clazz Java class
 * @param counter_id Counter identifier
 * @param reason Reason for missing value
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_NvtxCounter_sampleNoValueNative(JNIEnv* env,
                                                                             jclass clazz,
                                                                             jlong counter_id,
                                                                             jbyte reason)
{
  JNI_TRY
  {
    nvtxCounterSampleNoValue(get_java_domain(),
                             static_cast<uint64_t>(counter_id),
                             static_cast<uint8_t>(reason));
  }
  JNI_CATCH(env, );
}

/**
 * Submit a batch of counter samples.
 * 
 * @param env JNI environment
 * @param clazz Java class
 * @param counter_id Counter identifier
 * @param j_counters Byte array containing counter samples
 * @param flags Batch flags
 * @param j_timestamps Timestamp array (may be null)
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_NvtxCounter_submitBatchNative(JNIEnv* env,
                                                                           jclass clazz,
                                                                           jlong counter_id,
                                                                           jbyteArray j_counters,
                                                                           jlong flags,
                                                                           jlongArray j_timestamps)
{
  JNI_TRY
  {
    if (j_counters == nullptr) {
      JNI_THROW_NEW(
        env, "java/lang/IllegalArgumentException", "Counters array cannot be null", );
    }

    // Get counter data
    cudf::jni::native_jbyteArray counters(env, j_counters);
    jsize counters_size = env->GetArrayLength(j_counters);

    // Get timestamp data if provided
    std::unique_ptr<cudf::jni::native_jlongArray> timestamps;
    jsize timestamps_size = 0;
    int64_t const* timestamps_ptr = nullptr;

    if (j_timestamps != nullptr) {
      timestamps = std::make_unique<cudf::jni::native_jlongArray>(env, j_timestamps);
      timestamps_size = env->GetArrayLength(j_timestamps);
      timestamps_ptr  = reinterpret_cast<int64_t const*>(timestamps->data());
    }

    // Create and submit the batch
    nvtxCounterBatch_t batch{};
    batch.counterId      = static_cast<uint64_t>(counter_id);
    batch.counters       = counters.data();
    batch.countersSize   = static_cast<size_t>(counters_size);
    batch.flags          = static_cast<uint64_t>(flags);
    batch.timestamps     = timestamps_ptr;
    batch.timestampsSize = static_cast<size_t>(timestamps_size * sizeof(int64_t));

    nvtxCounterBatchSubmit(get_java_domain(), &batch);
  }
  JNI_CATCH(env, );
}

}  // extern "C"

