/*
 *
 *  Copyright (c) 2023, NVIDIA CORPORATION.
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

public class ChunkedContiguousSplit implements AutoCloseable {
  long nativePtr;
  public ChunkedContiguousSplit(long nativePtr) {
    this.nativePtr = nativePtr;
  }
  public long getSize() {
    return chunkedContiguousSplitSize(nativePtr);
  }

  public boolean hasNext() {
    return chunkedContiguousSplitHasNext(nativePtr);
  }

  public long next(DeviceMemoryBuffer userPtr) {
    return chunkedContiguousSplitNext(nativePtr, userPtr.getAddress(), userPtr.getLength());
  }

  public PackedColumnMetadata getPackedColumnMetadata() {
    return chunkedContiguousSplitMakePackedColumns(nativePtr);
  }

  @Override
  public void close() {
    destroyChunkedContiguousSplit(nativePtr);
  }

  private static native long chunkedContiguousSplitSize(long nativePtr);
  private static native boolean chunkedContiguousSplitHasNext(long nativePtr);
  private static native long chunkedContiguousSplitNext(long nativePtr, long userPtr, long userPtrSize);
  private static native PackedColumnMetadata chunkedContiguousSplitMakePackedColumns(long nativePtr);
  private static native void destroyChunkedContiguousSplit(long nativePtr);
}
