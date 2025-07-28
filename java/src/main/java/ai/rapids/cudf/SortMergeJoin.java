/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class represents a sort-merge join handle that can be used for performing
 * sort-merge join operations. The handle manages the native resources associated
 * with the sort-merge join and provides a reference counted interface to ensure
 * proper resource cleanup.
 */
public class SortMergeJoin implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger log = LoggerFactory.getLogger(SortMergeJoin.class);

  private boolean isClosed = false;
  private long handle = 0;

  public SortMergeJoin(Table buildTable, boolean buildTableSorted) {
    handle = create(buildTable.getNativeView(), buildTableSorted);
  }

  @Override
  public synchronized void close() {
    if (!isClosed) {
      destroy(handle);
      isClosed = true;
      handle = 0;
    }
  }

  long getNativeHandle() {
    return handle;
  }

  /**
   * Create a partition context for performing partitioned joins with a stream table.
   * @param streamTable the stream table to join with
   * @param streamTableSorted true if the stream table is already sorted otherwise false
   * @return a PartitionContext that can be used for partitioned joins
   */
  public PartitionContext makePartitionContext(Table streamTable, boolean streamTableSorted) {
    long contextHandle = makePartitionContext(handle, streamTable.getNativeView(), streamTableSorted);
    return new PartitionContext(contextHandle);
  }

  /**
   * A context object for managing partition-based sort merge joins.
   * This class manages the native resources associated with a partition context.
   */
  public static class PartitionContext implements AutoCloseable {
    private long contextHandle;
    private boolean isClosed = false;

    PartitionContext(long contextHandle) {
      this.contextHandle = contextHandle;
    }

    @Override
    public synchronized void close() {
      if (!isClosed) {
        destroyPartitionContext(contextHandle);
        isClosed = true;
        contextHandle = 0;
      }
    }

    /**
     * Get the number of rows that would result from the join.
     * @return array of row counts
     */
    public long[] getNumRows() {
      return SortMergeJoin.getNumRows(contextHandle);
    }

    /**
     * Perform a partitioned join for a specific range of rows.
     * @param joinObj the sort merge join object
     * @param startRow the starting row index
     * @param numRows the number of rows to process
     * @return gather maps for the join result
     */
    public long[] partitionedJoin(SortMergeJoin joinObj, long startRow, long numRows) {
      return SortMergeJoin.partitionedJoin(joinObj.getNativeHandle(), contextHandle, startRow, numRows);
    }

    long getContextHandle() {
      return contextHandle;
    }
  }

  private static native long create(long buildTableView, boolean buildTableSorted);
  private static native void destroy(long handle);
  private static native long makePartitionContext(long joinObj, long streamTableView, boolean streamTableSorted);
  private static native void destroyPartitionContext(long partitionContext);
  private static native long[] getNumRows(long partitionContext);
  private static native long[] partitionedJoin(long joinObj, long partitionContext, long startRow, long numRows);
} 