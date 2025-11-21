/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * Helper class to submit a batch of counter samples.
 * 
 * This class allows efficient submission of multiple counter samples at once,
 * which can be more performant than individual sample calls when dealing with
 * high-frequency counter updates.
 * 
 * Example usage:
 * <pre>
 * {@code
 * try (NvtxCounter counter = NvtxCounter.create("Latency")) {
 *     // Prepare counter data as byte array (e.g., multiple int64 values)
 *     byte[] counterData = new byte[8 * numSamples];
 *     // ... fill counterData with values ...
 *     
 *     // Prepare timestamps (one per sample)
 *     long[] timestamps = new long[numSamples];
 *     // ... fill timestamps ...
 *     
 *     NvtxCounterBatch batch = new NvtxCounterBatch.Builder()
 *         .counters(counterData)
 *         .timestamps(timestamps)
 *         .build();
 *     
 *     counter.submitBatch(batch);
 * }
 * }
 * </pre>
 */
public class NvtxCounterBatch {
  private final byte[] counters;
  private final long flags;
  private final long[] timestamps;

  private NvtxCounterBatch(Builder builder) {
    this.counters = builder.counters;
    this.flags = builder.flags;
    this.timestamps = builder.timestamps;
  }

  /**
   * @return Batch of counter samples as byte array
   */
  public byte[] getCounters() {
    return counters;
  }

  /**
   * @return Batch flags (timestamp ordering, timestamp style, etc.)
   */
  public long getFlags() {
    return flags;
  }

  /**
   * @return Array of timestamps or timestamp/interval pairs (may be null)
   */
  public long[] getTimestamps() {
    return timestamps;
  }

  /**
   * Builder for creating NvtxCounterBatch instances.
   */
  public static class Builder {
    private byte[] counters;
    private long flags = 0;
    private long[] timestamps = null;

    /**
     * Create a new builder.
     */
    public Builder() {
    }

    /**
     * Set the batch of counter samples.
     * 
     * @param counters Byte array containing counter samples
     * @return This builder instance
     */
    public Builder counters(byte[] counters) {
      if (counters == null || counters.length == 0) {
        throw new IllegalArgumentException("Counters array cannot be null or empty");
      }
      this.counters = counters;
      return this;
    }

    /**
     * Set batch flags (e.g., timestamp ordering, interval pairs).
     * 
     * @param flags Batch flags (see NvtxCounterConstants.COUNTER_BATCH_FLAG_*)
     * @return This builder instance
     */
    public Builder flags(long flags) {
      this.flags = flags;
      return this;
    }

    /**
     * Set the timestamps array for the samples.
     * By default, one timestamp per sample is assumed unless interval pair flags are set.
     * 
     * @param timestamps Array of timestamps or timestamp/interval pairs
     * @return This builder instance
     */
    public Builder timestamps(long[] timestamps) {
      this.timestamps = timestamps;
      return this;
    }

    /**
     * Build the NvtxCounterBatch instance.
     * 
     * @return A new NvtxCounterBatch instance
     */
    public NvtxCounterBatch build() {
      if (counters == null) {
        throw new IllegalStateException("Counters array must be set before building");
      }
      return new NvtxCounterBatch(this);
    }
  }
}

