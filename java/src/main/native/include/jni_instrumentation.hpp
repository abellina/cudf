/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file jni_instrumentation.hpp
 * @brief Configurable instrumentation for cudf JNI calls
 *
 * This file provides NVTX range instrumentation and optional sleep delays
 * for JNI API calls. This is useful for profiling and debugging.
 *
 * To enable instrumentation, set CUDF_JNI_ENABLE_INSTRUMENTATION to 1 below.
 * To disable it, set it to 0 (all instrumentation code will be compiled out).
 */

// ============================================================================
// CONFIGURATION - Change these values to enable/disable instrumentation
// ============================================================================

// Set to 1 to enable NVTX ranges and sleep for JNI calls, 0 to disable
#define CUDF_JNI_ENABLE_INSTRUMENTATION 0

// Sleep duration in microseconds (10us = 10 microseconds)
// Set to 0 to disable sleep but keep NVTX ranges
#define CUDF_JNI_INSTRUMENTATION_SLEEP_US 10

// ============================================================================
// Implementation
// ============================================================================

#if CUDF_JNI_ENABLE_INSTRUMENTATION

#include "nvtx_common.hpp"

#include <nvtx3/nvtx3.hpp>

#include <chrono>
#include <thread>

namespace cudf::jni {

/**
 * @brief Instrument a JNI call with an NVTX range around a sleep
 *
 * This function pushes an NVTX range named "nanosleep_nvtx", sleeps, then pops the range.
 * The NVTX range wraps only the sleep call, not the entire JNI function.
 */
inline void jni_instrumentation_marker()
{
  // Push NVTX range
  nvtx3::color range_color(0xFFFFA500);  // Orange color
  nvtx3::event_attributes attr{range_color, "nanosleep_nvtx"};
  nvtxDomainRangePushEx(nvtx3::domain::get<cudf::jni::java_domain>(), attr.get());

  // Sleep inside the NVTX range
#if CUDF_JNI_INSTRUMENTATION_SLEEP_US > 0
  std::this_thread::sleep_for(std::chrono::microseconds(CUDF_JNI_INSTRUMENTATION_SLEEP_US));
#endif

  // Pop NVTX range
  nvtxDomainRangePop(nvtx3::domain::get<cudf::jni::java_domain>());
}

}  // namespace cudf::jni

/**
 * @brief Macro to add JNI instrumentation marker
 *
 * This emits an NVTX range named "nanosleep_nvtx" that wraps a short sleep.
 */
#define JNI_INSTRUMENTATION_SCOPE \
  cudf::jni::jni_instrumentation_marker()

#else  // CUDF_JNI_ENABLE_INSTRUMENTATION == 0

// When instrumentation is disabled, this macro becomes a no-op
#define JNI_INSTRUMENTATION_SCOPE ((void)0)

#endif  // CUDF_JNI_ENABLE_INSTRUMENTATION
