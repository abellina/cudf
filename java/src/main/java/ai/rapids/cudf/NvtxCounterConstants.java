/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * Constants for NVTX Counter Extension.
 * 
 * This class contains various constants used by the NVTX counter extension,
 * including counter IDs, sample reasons, batch flags, and predefined schema IDs.
 */
public final class NvtxCounterConstants {
  
  // Prevent instantiation
  private NvtxCounterConstants() {}

  // ===== Counter IDs =====
  
  /**
   * The counter ID is not specified (let the tool generate a dynamic ID).
   */
  public static final long COUNTER_ID_NONE = 0L;

  /**
   * Starting value for static (user-provided, feed-forward) counter IDs.
   * Static counter IDs must be >= this value and < COUNTER_ID_DYNAMIC_START.
   */
  public static final long COUNTER_ID_STATIC_START = 1L << 24;

  /**
   * Starting value for dynamically (tool) generated counter IDs.
   */
  public static final long COUNTER_ID_DYNAMIC_START = 1L << 32;

  // ===== Sample Reasons =====
  
  /**
   * Counter sample value is zero.
   */
  public static final byte COUNTER_SAMPLE_ZERO = 0;

  /**
   * Counter sample value is unchanged from previous sample.
   */
  public static final byte COUNTER_SAMPLE_UNCHANGED = 1;

  /**
   * Counter sample is unavailable (failed to get a counter sample).
   */
  public static final byte COUNTER_SAMPLE_UNAVAILABLE = 2;

  // ===== Batch Flags =====
  
  /**
   * Indicates that timestamps array contains begin-time interval pairs.
   * Each sample has two timestamps: begin and end.
   */
  public static final long COUNTER_BATCH_FLAG_BEGINTIME_INTERVAL_PAIR = 1L << 32;

  /**
   * Indicates that timestamps array contains end-time interval pairs.
   * Each sample has two timestamps: begin and end.
   */
  public static final long COUNTER_BATCH_FLAG_ENDTIME_INTERVAL_PAIR = 2L << 32;

  // ===== Scope IDs =====
  
  /**
   * No specific scope (global scope).
   */
  public static final long SCOPE_NONE = 0L;

  // ===== Predefined Schema IDs for common counter types =====
  
  /**
   * Schema for a single 64-bit signed integer counter.
   */
  public static final long SCHEMA_ID_INT64 = 1L;

  /**
   * Schema for a single 64-bit floating-point counter.
   */
  public static final long SCHEMA_ID_FLOAT64 = 2L;

  /**
   * Schema for a single 32-bit signed integer counter.
   */
  public static final long SCHEMA_ID_INT32 = 3L;

  /**
   * Schema for a single 32-bit floating-point counter.
   */
  public static final long SCHEMA_ID_FLOAT32 = 4L;
}

