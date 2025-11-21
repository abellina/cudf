/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * Java wrapper for NVTX Counter Extension APIs.
 * 
 * This class provides an interface to the NVTX (NVIDIA Tools Extension Library) counter
 * extension, which allows applications to register and sample custom counters that can be
 * visualized in profiling tools like Nsight Systems.
 * 
 * Counters can be used to track various metrics like memory usage, queue sizes, cache hits,
 * or any other numeric value that changes over time during application execution.
 * 
 * Example usage:
 * <pre>
 * {@code
 * // Register a counter (use try-with-resources to ensure cleanup)
 * try (NvtxCounter counter = NvtxCounter.create("Memory Usage", "Current memory usage in bytes")) {
 *     // Sample the counter
 *     counter.sampleInt64(memoryUsage);
 *     
 *     // Or sample as double
 *     counter.sampleFloat64(cpuTemperature);
 * }
 * }
 * </pre>
 */
public class NvtxCounter implements AutoCloseable {
  private static final boolean isEnabled = Boolean.getBoolean("ai.rapids.cudf.nvtx.enabled");

  static {
    if (isEnabled) {
      NativeDepsLoader.loadNativeDeps();
    }
  }

  private final long counterId;
  private boolean closed = false;

  /**
   * Private constructor - use factory methods to create instances.
   */
  private NvtxCounter(long counterId) {
    this.counterId = counterId;
  }

  /**
   * Create a counter with default attributes.
   * 
   * @param name Name of the counter
   * @return A new NvtxCounter instance, or a no-op counter if NVTX is disabled
   */
  public static NvtxCounter create(String name) {
    return create(name, null);
  }

  /**
   * Create a counter with name and description.
   * 
   * @param name Name of the counter
   * @param description Optional description of the counter
   * @return A new NvtxCounter instance, or a no-op counter if NVTX is disabled
   */
  public static NvtxCounter create(String name, String description) {
    NvtxCounterAttr attr = new NvtxCounterAttr.Builder(name)
        .description(description)
        .build();
    return create(attr);
  }

  /**
   * Create a counter with full attributes.
   * 
   * @param attr Attributes of the counter to register
   * @return A new NvtxCounter instance, or a no-op counter if NVTX is disabled
   */
  public static NvtxCounter create(NvtxCounterAttr attr) {
    long id = NvtxCounterConstants.COUNTER_ID_NONE;
    if (isEnabled) {
      id = register(attr.getName(), attr.getDescription(), attr.getSchemaId(), 
                   attr.getScopeId(), attr.getCounterId());
    }
    return new NvtxCounter(id);
  }

  /**
   * Sample this counter with an integer value (the profiler determines the timestamp).
   * 
   * @param value 64-bit integer counter value
   */
  public void sampleInt64(long value) {
    if (closed) {
      throw new IllegalStateException("Cannot sample a closed counter");
    }
    if (isEnabled) {
      sampleInt64Native(counterId, value);
    }
  }

  /**
   * Sample this counter with a floating-point value (the profiler determines the timestamp).
   * 
   * @param value 64-bit floating-point counter value
   */
  public void sampleFloat64(double value) {
    if (closed) {
      throw new IllegalStateException("Cannot sample a closed counter");
    }
    if (isEnabled) {
      sampleFloat64Native(counterId, value);
    }
  }

  /**
   * Sample this counter without a value (e.g., to indicate unchanged or unavailable data).
   * 
   * @param reason Reason for the missing sample value (see NvtxCounterConstants.COUNTER_SAMPLE_*)
   */
  public void sampleNoValue(byte reason) {
    if (closed) {
      throw new IllegalStateException("Cannot sample a closed counter");
    }
    if (isEnabled) {
      sampleNoValueNative(counterId, reason);
    }
  }

  /**
   * Submit a batch of counter samples for this counter.
   * 
   * @param counters Byte array containing counter samples
   * @param flags Batch flags (timestamp ordering, timestamp style, etc.)
   * @param timestamps Optional array of timestamps or timestamp/interval pairs
   */
  public void submitBatch(byte[] counters, long flags, long[] timestamps) {
    if (closed) {
      throw new IllegalStateException("Cannot submit batch for a closed counter");
    }
    if (isEnabled) {
      submitBatchNative(counterId, counters, flags, timestamps);
    }
  }

  /**
   * Submit a batch of counter samples for this counter.
   * 
   * @param batch The counter batch to submit (counter ID is ignored, this counter's ID is used)
   */
  public void submitBatch(NvtxCounterBatch batch) {
    submitBatch(batch.getCounters(), batch.getFlags(), batch.getTimestamps());
  }

  /**
   * Close this counter and free associated resources.
   * After closing, the counter cannot be sampled anymore.
   */
  @Override
  public synchronized void close() {
    if (closed) {
      return; // Already closed, silently return
    }
    closed = true;
    if (isEnabled && counterId != NvtxCounterConstants.COUNTER_ID_NONE) {
      unregister(counterId);
    }
  }

  /**
   * @return true if this counter has been closed
   */
  public boolean isClosed() {
    return closed;
  }

  // Native methods
  private static native long register(String name, String description, long schemaId,
                                      long scopeId, long counterId);
  private static native void unregister(long counterId);
  private static native void sampleInt64Native(long counterId, long value);
  private static native void sampleFloat64Native(long counterId, double value);
  private static native void sampleNoValueNative(long counterId, byte reason);
  private static native void submitBatchNative(long counterId, byte[] counters, long flags,
                                               long[] timestamps);
}

