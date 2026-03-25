/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import ai.rapids.cudf.ast.CompiledExpression;

/**
 * Experimental Parquet hybrid scan reader that optimally reads parquet files subject to
 * highly selective filters.
 *
 * This reader performs multi-phase filtering:
 * 1. Parse footer and get row groups
 * 2. Filter row groups using statistics
 * 3. Filter row groups using dictionary pages
 * 4. Filter using bloom filters
 *
 * The reader returns filtered row group indices that can be used to read only the
 * necessary data from the parquet file.
 */
public class HybridScanReader implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private long nativeHandle;
  private boolean closed = false;

  /**
   * Create a hybrid scan reader from footer bytes.
   *
   * @param footerBuffer Host memory buffer containing parquet footer bytes
   * @param filter Compiled AST filter expression, or null for no filter
   * @param columns Column names to read, or null/empty for all columns
   */
  public HybridScanReader(HostMemoryBuffer footerBuffer, CompiledExpression filter,
                          String[] columns) {
    long filterHandle = (filter != null) ? filter.getNativeHandle() : 0;
    this.nativeHandle = createFromFooter(
        footerBuffer.getAddress(),
        footerBuffer.getLength(),
        filterHandle,
        columns);
  }

  /**
   * Get all available row group indices from the parquet file.
   *
   * @return Array of row group indices
   */
  public int[] getAllRowGroups() {
    assertNotClosed();
    return getAllRowGroups(nativeHandle);
  }

  /**
   * Filter row groups using column chunk statistics from the footer.
   *
   * @param rowGroupIndices Input row group indices to filter
   * @return Filtered row group indices (subset of input)
   */
  public int[] filterRowGroupsWithStats(int[] rowGroupIndices) {
    assertNotClosed();
    return filterRowGroupsWithStats(nativeHandle, rowGroupIndices);
  }

  /**
   * Get byte ranges for dictionary pages needed for dictionary-based filtering.
   * Returns pairs of (offset, length) for each dictionary page.
   *
   * @param rowGroupIndices Row group indices to get dictionary ranges for
   * @return Array of [offset, length] pairs, or empty if no dictionaries
   */
  public long[] getDictionaryPageByteRanges(int[] rowGroupIndices) {
    assertNotClosed();
    return getDictionaryPageByteRanges(nativeHandle, rowGroupIndices);
  }

  /**
   * Filter row groups using dictionary pages.
   *
   * @param dictionaryData Device buffers containing dictionary page data
   * @param rowGroupIndices Input row group indices to filter
   * @return Filtered row group indices
   */
  public int[] filterRowGroupsWithDictionaries(DeviceMemoryBuffer[] dictionaryData,
                                               int[] rowGroupIndices) {
    assertNotClosed();
    long[] bufferAddresses = new long[dictionaryData.length];
    long[] bufferLengths = new long[dictionaryData.length];
    for (int i = 0; i < dictionaryData.length; i++) {
      bufferAddresses[i] = dictionaryData[i].getAddress();
      bufferLengths[i] = dictionaryData[i].getLength();
    }
    return filterRowGroupsWithDictionaries(nativeHandle, bufferAddresses, bufferLengths,
        rowGroupIndices);
  }

  /**
   * Get the total number of rows in the specified row groups.
   *
   * @param rowGroupIndices Row group indices
   * @return Total number of rows
   */
  public long getTotalRowsInRowGroups(int[] rowGroupIndices) {
    assertNotClosed();
    return getTotalRowsInRowGroups(nativeHandle, rowGroupIndices);
  }

  /**
   * Get byte ranges for filter column chunks.
   * Returns pairs of (offset, length) for each column chunk.
   *
   * @param rowGroupIndices Row group indices
   * @return Array of [offset, length] pairs
   */
  public long[] getFilterColumnChunkByteRanges(int[] rowGroupIndices) {
    assertNotClosed();
    return getFilterColumnChunkByteRanges(nativeHandle, rowGroupIndices);
  }

  /**
   * Get byte ranges for payload column chunks.
   * Returns pairs of (offset, length) for each column chunk.
   *
   * @param rowGroupIndices Row group indices
   * @return Array of [offset, length] pairs
   */
  public long[] getPayloadColumnChunkByteRanges(int[] rowGroupIndices) {
    assertNotClosed();
    return getPayloadColumnChunkByteRanges(nativeHandle, rowGroupIndices);
  }

  /**
   * Get byte ranges for ALL column chunks (filter + payload combined).
   * Used for single-stage materialization where all columns are read at once.
   * Returns pairs of (offset, length) for each column chunk.
   *
   * @param rowGroupIndices Row group indices
   * @return Array of [offset, length] pairs
   */
  public long[] getAllColumnChunkByteRanges(int[] rowGroupIndices) {
    assertNotClosed();
    return getAllColumnChunkByteRanges(nativeHandle, rowGroupIndices);
  }

  /**
   * Materialize all columns from the given column chunk data into a Table.
   * Materialize all columns from the given column chunk data (no filtering).
   *
   * Reads all data without filtering:
   * 1. Get byte ranges via getPayloadColumnChunkByteRanges()
   * 2. Read data from file into device buffers
   * 3. Call this method to decode into a Table
   *
   * @param rowGroupIndices Row group indices to read
   * @param columnChunkData Device buffers containing column chunk data in the same order
   *                        as returned by getPayloadColumnChunkByteRanges()
   * @return Table containing the decoded data
   */
  public Table materialize(int[] rowGroupIndices, DeviceMemoryBuffer[] columnChunkData) {
    assertNotClosed();
    long[] bufferAddresses = new long[columnChunkData.length];
    long[] bufferLengths = new long[columnChunkData.length];
    for (int i = 0; i < columnChunkData.length; i++) {
      bufferAddresses[i] = columnChunkData[i].getAddress();
      bufferLengths[i] = columnChunkData[i].getLength();
    }
    long[] columnHandles = materializePayloadColumns(nativeHandle, rowGroupIndices,
        bufferAddresses, bufferLengths);
    return new Table(columnHandles);
  }

  /**
   * Convenience method to materialize all row groups from pre-loaded data buffer.
   * Convenience method to materialize all row groups from a pre-loaded host buffer.
   *
   * @param fileData The complete parquet file data as a host buffer
   * @return Table containing the decoded data
   */
  public Table materializeFromBuffer(HostMemoryBuffer fileData) {
    assertNotClosed();
    long[] columnHandles = materializeFromHostBuffer(nativeHandle, 
        fileData.getAddress(), fileData.getLength());
    return new Table(columnHandles);
  }

  /**
   * Materialize columns from host memory buffers containing column chunk data.
   * This avoids the extra device-to-device copy that happens when using DeviceMemoryBuffer.
   * 
   * The data is copied directly from host to device in the JNI layer, then passed to cuDF.
   *
   * @param rowGroupIndices Row group indices to read
   * @param columnChunkData Host buffers containing column chunk data in the same order
   *                        as returned by getPayloadColumnChunkByteRanges()
   * @return Table containing the decoded data
   */
  /**
   * Materialize columns from host memory buffers (two-stage mode).
   * This is the optimized path that copies directly from host to device.
   *
   * @param rowGroupIndices the row groups to materialize
   * @param filterColumnData host buffers containing filter column chunk data (can be null/empty)
   * @param payloadColumnData host buffers containing payload column chunk data
   * @return a Table with the materialized columns in the originally requested order
   */
  public Table materializeFromHostBuffers(int[] rowGroupIndices,
                                          HostMemoryBuffer[] filterColumnData,
                                          HostMemoryBuffer[] payloadColumnData) {
    assertNotClosed();
    
    // Prepare filter column buffers (may be null/empty)
    long[] filterAddresses = null;
    long[] filterLengths = null;
    if (filterColumnData != null && filterColumnData.length > 0) {
      filterAddresses = new long[filterColumnData.length];
      filterLengths = new long[filterColumnData.length];
      for (int i = 0; i < filterColumnData.length; i++) {
        filterAddresses[i] = filterColumnData[i].getAddress();
        filterLengths[i] = filterColumnData[i].getLength();
      }
    }
    
    // Prepare payload column buffers
    long[] payloadAddresses = new long[payloadColumnData.length];
    long[] payloadLengths = new long[payloadColumnData.length];
    for (int i = 0; i < payloadColumnData.length; i++) {
      payloadAddresses[i] = payloadColumnData[i].getAddress();
      payloadLengths[i] = payloadColumnData[i].getLength();
    }
    
    long[] columnHandles = materializeFromHostBuffers(nativeHandle, rowGroupIndices,
        filterAddresses, filterLengths, payloadAddresses, payloadLengths);
    return new Table(columnHandles);
  }

  /**
   * Materialize ALL columns in single-stage mode from host memory buffers.
   * Reads all column data at once and applies filter after decoding.
   * Best for high selectivity filters or few/small payload columns.
   *
   * @param rowGroupIndices the row groups to materialize
   * @param allColumnData host buffers containing ALL column chunk data in the order
   *                      returned by getAllColumnChunkByteRanges()
   * @return a Table with all materialized columns
   */
  public Table materializeAllColumnsFromHostBuffers(int[] rowGroupIndices,
                                                    HostMemoryBuffer[] allColumnData) {
    assertNotClosed();
    
    long[] addresses = new long[allColumnData.length];
    long[] lengths = new long[allColumnData.length];
    for (int i = 0; i < allColumnData.length; i++) {
      addresses[i] = allColumnData[i].getAddress();
      lengths[i] = allColumnData[i].getLength();
    }
    
    long[] columnHandles = materializeAllColumnsFromHostBuffers(nativeHandle, rowGroupIndices,
        addresses, lengths);
    return new Table(columnHandles);
  }

  private void assertNotClosed() {
    if (closed) {
      throw new IllegalStateException("HybridScanReader has been closed");
    }
  }

  @Override
  public void close() {
    if (!closed) {
      destroy(nativeHandle);
      nativeHandle = 0;
      closed = true;
    }
  }

  // Native methods
  private static native long createFromFooter(long footerAddress, long footerLength,
                                              long filterHandle, String[] columns);
  private static native int[] getAllRowGroups(long handle);
  private static native int[] filterRowGroupsWithStats(long handle, int[] rowGroupIndices);
  private static native long[] getDictionaryPageByteRanges(long handle, int[] rowGroupIndices);
  private static native int[] filterRowGroupsWithDictionaries(long handle, long[] bufferAddresses,
                                                              long[] bufferLengths,
                                                              int[] rowGroupIndices);
  private static native long getTotalRowsInRowGroups(long handle, int[] rowGroupIndices);
  private static native long[] getFilterColumnChunkByteRanges(long handle, int[] rowGroupIndices);
  private static native long[] getPayloadColumnChunkByteRanges(long handle, int[] rowGroupIndices);
  private static native long[] getAllColumnChunkByteRanges(long handle, int[] rowGroupIndices);
  private static native long[] materializePayloadColumns(long handle, int[] rowGroupIndices,
                                                         long[] bufferAddresses,
                                                         long[] bufferLengths);
  private static native long[] materializeFromHostBuffer(long handle, long bufferAddress,
                                                         long bufferLength);
  private static native long[] materializeFromHostBuffers(long handle, int[] rowGroupIndices,
                                                          long[] filterBufferAddresses,
                                                          long[] filterBufferLengths,
                                                          long[] payloadBufferAddresses,
                                                          long[] payloadBufferLengths);
  private static native long[] materializeAllColumnsFromHostBuffers(long handle,
                                                                    int[] rowGroupIndices,
                                                                    long[] bufferAddresses,
                                                                    long[] bufferLengths);
  private static native void destroy(long handle);
}

