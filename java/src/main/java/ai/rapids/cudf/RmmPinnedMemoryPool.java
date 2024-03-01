/*
 *
 *  Copyright (c) 2019-2024, NVIDIA CORPORATION.
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


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Comparator;
import java.util.Iterator;
import java.util.Objects;
import java.util.Optional;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * This provides a pool of pinned memory similar to what RMM does for device memory.
 */
public final class PinnedMemoryPool implements AutoCloseable {
  private static final Logger log = LoggerFactory.getLogger(PinnedMemoryPool.class);
  private static final long ALIGNMENT = ColumnView.hostPaddingSizeInBytes();

  // These static fields should only ever be accessed when class-synchronized.
  // Do NOT use singleton_ directly!  Use the getSingleton accessor instead.
  private static volatile PinnedMemoryPool singleton_ = null;
  private static Future<PinnedMemoryPool> initFuture = null;

  private final long totalPoolSize;
  private RmmPoolHostMemoryResource<RmmPinnedHostMemoryResource> rmmPool;
  
  private static final class PinnedHostBufferCleanerNew extends MemoryBuffer.MemoryBufferCleaner {
    private final long origLength;
    private long ptr;

    PinnedHostBufferCleanerNew(long ptr, long length) {
      this.ptr = ptr;
      origLength = length;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long origAddress = 0;
      if (ptr != -1) {
        try {
          PinnedMemoryPool.freeInternal(ptr, origLength);
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          ptr = -1;
        }
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("A PINNED HOST BUFFER WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress) + ")");
        logRefCountDebug("Leaked pinned host buffer");
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return ptr == -1;
    }
  }

  private static PinnedMemoryPool getSingleton() {
    if (singleton_ == null) {
      if (initFuture == null) {
        return null;
      }

      synchronized (PinnedMemoryPool.class) {
        if (singleton_ == null) {
          try {
            singleton_ = initFuture.get();
          } catch (Exception e) {
            throw new RuntimeException("Error initializing pinned memory pool", e);
          }
          initFuture = null;
        }
      }
    }
    return singleton_;
  }

  private static void freeInternal(long ptr, long size) {
    Objects.requireNonNull(getSingleton()).free(ptr, size);
  }

  /**
   * Initialize the pool.
   *
   * @param poolSize size of the pool to initialize.
   */
  public static synchronized void initialize(long poolSize) {
    initialize(poolSize, -1, false);
  }

  /**
   * Initialize the pool.
   *
   * @param poolSize size of the pool to initialize.
   * @param gpuId    gpu id to set to get memory pool from, -1 means to use default
   */
  public static synchronized void initialize(long poolSize, int gpuId, boolean setCuioDefaultResource) {
    if (isInitialized()) {
      throw new IllegalStateException("Can only initialize the pool once.");
    }
    ExecutorService initService = Executors.newSingleThreadExecutor(runnable -> {
      Thread t = new Thread(runnable, "pinned pool init");
      t.setDaemon(true);
      return t;
    });
    initFuture = initService.submit(() -> new PinnedMemoryPool(poolSize, gpuId, setCuioDefaultResource));
    initService.shutdown();
  }

  /**
   * Check if the pool has been initialized or not.
   */
  public static boolean isInitialized() {
    return getSingleton() != null;
  }

  /**
   * Shut down the pool of memory. If there are outstanding allocations this may fail.
   */
  public static synchronized void shutdown() {
    PinnedMemoryPool pool = getSingleton();
    if (pool != null) {
      pool.close();
    }
    initFuture = null;
    singleton_ = null;
  }

  /**
   * Factory method to create a pinned host memory buffer.
   *
   * @param bytes size in bytes to allocate
   * @return newly created buffer or null if insufficient pinned memory
   */
  public static HostMemoryBuffer tryAllocate(long bytes) {
    HostMemoryBuffer result = null;
    PinnedMemoryPool pool = getSingleton();
    if (pool != null) {
      result = pool.tryAllocateInternal(bytes);
    }
    return result;
  }

  /**
   * Factory method to create a host buffer but preferably pointing to pinned memory.
   * It is not guaranteed that the returned buffer will be pointer to pinned memory.
   *
   * @param bytes size in bytes to allocate
   * @return newly created buffer
   */
  public static HostMemoryBuffer allocate(long bytes, HostMemoryAllocator hostMemoryAllocator) {
    HostMemoryBuffer result = tryAllocate(bytes);
    if (result == null) {
      result = hostMemoryAllocator.allocate(bytes, false);
    }
    return result;
  }

  /**
   * Factory method to create a host buffer but preferably pointing to pinned memory.
   * It is not guaranteed that the returned buffer will be pointer to pinned memory.
   *
   * @param bytes size in bytes to allocate
   * @return newly created buffer
   */
  public static HostMemoryBuffer allocate(long bytes) {
    return allocate(bytes, DefaultHostMemoryAllocator.get());
  }

  /**
   * Get the number of bytes that the pinned memory pool was allocated with.
   */
  public static long getTotalPoolSizeBytes() {
    PinnedMemoryPool pool = getSingleton();
    if (pool != null) {
      return pool.getTotalPoolSizeInternal();
    }
    return 0;
  }

  private PinnedMemoryPool(long poolSize, int gpuId, boolean setCuioDefaultResource) {
    if (gpuId > -1) {
      // set the gpu device to use
      Cuda.setDevice(gpuId);
      Cuda.freeZero();
    }
    this.totalPoolSize = poolSize;
    this.rmmPool =
        new RmmPoolHostMemoryResource<>(new RmmPinnedHostMemoryResource(), poolSize, poolSize);

    if (setCuioDefaultResource) {
      Rmm.setCuioCurrentHostMemoryResource(this.rmmPool.getHandle());
    }
  }

  @Override
  public void close() {
    this.rmmPool.close();
  }

  public void free(long ptr, long size) {
    try(NvtxRange x = new NvtxRange("pinned_free", NvtxColor.YELLOW)) {
      Rmm.freeHost(ptr, size, this.rmmPool.getHandle(), Cuda.DEFAULT_STREAM.getStream());
    }
  }

  private synchronized HostMemoryBuffer tryAllocateInternal(long bytes) {
    long allocated = Rmm.allocHostInternal(bytes, this.rmmPool.getHandle(), Cuda.DEFAULT_STREAM.getStream());
    if (allocated == -1) {
      try(NvtxRange x = new NvtxRange("pinned_failed", NvtxColor.RED)) {
        return null;
      }
    } else {
      try(NvtxRange x = new NvtxRange("pinned_success", NvtxColor.GREEN)) {
        return new HostMemoryBuffer(allocated, bytes,
                new PinnedHostBufferCleanerNew(allocated, bytes));
      }
    }
  }

  private long getTotalPoolSizeInternal() {
    return this.totalPoolSize;
  }
}
