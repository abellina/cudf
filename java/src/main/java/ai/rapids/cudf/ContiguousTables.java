/*
 *
 *  Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

/**
 * A table that is backed by a single contiguous device buffer. This makes transfers of the data
 * much simpler.
 */
public final class ContiguousTables implements AutoCloseable {
  private long[] dataOffsets;
  private long[] dataLengths;
  private PackedColumnMetadata[] metas;
  private final long[] rowCounts;
  private DeviceMemoryBuffer buffer;

  // This method is invoked by JNI
  static ContiguousTables fromPackedTables(
    long[] metadataHandles,
    long[] dataOffsets,
    long[] dataLengths,
    long[] rowCounts,
    long baseAddress,
    long baseLength,
    long rmmBufferAddress) {
    DeviceMemoryBuffer buffer = DeviceMemoryBuffer.fromRmm(baseAddress, baseLength, rmmBufferAddress);
    ContiguousTables res = new ContiguousTables(metadataHandles, dataOffsets, dataLengths, rowCounts, buffer);
    return res;
  }

  /** Construct a contiguous table instance given a table and the device buffer backing it. */
  ContiguousTables(
    long[] metadataHandles, 
    long[] dataOffsets, 
    long[] dataLengths, 
    long[] rowCounts, 
    DeviceMemoryBuffer buffer) {
    this.metas = new PackedColumnMetadata[metadataHandles.length];
    for (int i = 0; i < metadataHandles.length; i++) {
      this.metas[i] = new PackedColumnMetadata(metadataHandles[i]);
    }
    this.dataOffsets = dataOffsets;
    this.dataLengths = dataLengths;
    this.rowCounts = rowCounts;
    this.buffer = buffer;
  }

  /** Get the device buffer backing the contiguous table data. */
  public DeviceMemoryBuffer getBuffer() {
    return buffer;
  }

  public long[] getOffsets() {
    return dataOffsets;
  }

  public long[] getLengths() {
    return dataLengths;
  }

  public long[] getRowCounts() {
    return rowCounts;
  }

  public PackedColumnMetadata[] releaseMeta() {
    PackedColumnMetadata[] ret = metas;
    metas = null;
    return ret;
  }
  
  /** Close the contiguous table instance and its underlying resources. */
  @Override
  public void close() {
    if (buffer != null) {
      buffer.close();
      buffer = null;
    }
  }
}
