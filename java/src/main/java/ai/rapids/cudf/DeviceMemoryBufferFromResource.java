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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class represents data in some form on the GPU. Closing this object will effectively release
 * the memory held by the buffer.  Note that because of pooling in RMM or reference counting if a
 * buffer is sliced it may not actually result in the memory being released.
 */
public class DeviceMemoryBufferFromResource  extends BaseDeviceMemoryBuffer {
  private static final Logger log = LoggerFactory.getLogger(DeviceMemoryBufferFromResource.class);

  private static final class DeviceBufferCleaner extends MemoryBufferCleaner {
    private long address;
    private long lengthInBytes;
    private RmmDeviceMemoryResource resource;
    private Cuda.Stream stream;

    DeviceBufferCleaner(
        long address, 
        long lengthInBytes, 
        RmmDeviceMemoryResource resource, 
        Cuda.Stream stream) {
      this.address = address;
      this.lengthInBytes = lengthInBytes;
      this.resource = resource;
      this.stream = stream;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long origAddress = address;
      if (address != 0) {
        long s = stream == null ? 0 : stream.getStream();
        try {
          Rmm.freeFromResource(resource.getHandle(), address, lengthInBytes, s);
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          address = 0;
          lengthInBytes = 0;
          stream = null;
        }
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("A DEVICE BUFFER WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress) + ")");
        logRefCountDebug("Leaked device buffer");
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return address == 0;
    }
  }

  DeviceMemoryBufferFromResource(long address, long lengthInBytes, RmmDeviceMemoryResource resource, Cuda.Stream stream) {
    super(address, lengthInBytes, new DeviceBufferCleaner(address, lengthInBytes, resource, stream));
  }

  private DeviceMemoryBufferFromResource(
    long address, long lengthInBytes, DeviceMemoryBufferFromResource parent) {
    super(address, lengthInBytes, parent);
  }

  /**
   * Slice off a part of the device buffer. Note that this is a zero copy operation and all
   * slices must be closed along with the original buffer before the memory is released to RMM.
   * So use this with some caution.
   * @param offset where to start the slice at.
   * @param len how many bytes to slice
   * @return a device buffer that will need to be closed independently from this buffer.
   */
  @Override
  public synchronized final DeviceMemoryBufferFromResource slice(long offset, long len) {
    addressOutOfBoundsCheck(address + offset, len, "slice");
    incRefCount();
    return new DeviceMemoryBufferFromResource(
      getAddress() + offset, len, this);
  }

  /**
   * Convert a view that is a subset of this Buffer by slicing this.
   * @param view the view to use as a reference.
   * @return the sliced buffer.
   */
  synchronized final BaseDeviceMemoryBuffer sliceFrom(DeviceMemoryBufferView view) {
    if (view == null) {
      return null;
    }
    addressOutOfBoundsCheck(view.address, view.length, "sliceFrom");
    incRefCount();
    return new DeviceMemoryBufferFromResource(view.address, view.length, this);
  }
}
