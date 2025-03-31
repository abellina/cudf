/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class CudaTest {

  @Test
  public void testGetCudaRuntimeInfo() {
    // The driver version is not necessarily larger than runtime version. Drivers of previous
    // version are also able to support runtime of later version, only if they support same
    // kinds of computeModes.
    assert Cuda.getDriverVersion() >= 1000;
    assert Cuda.getRuntimeVersion() >= 1000;
    assertEquals(Cuda.getNativeComputeMode(), Cuda.getComputeMode().nativeId);
  }

  @Tag("noSanitizer")
  @Test
  public void testCudaException() {
    assertThrows(CudaException.class, () -> {
          try {
            Cuda.freePinned(-1L);
          } catch (CudaFatalException fatalEx) {
            throw new AssertionError("Expected CudaException but got fatal error", fatalEx);
          } catch (CudaException ex) {
            assertEquals(CudaException.CudaError.cudaErrorInvalidValue, ex.getCudaError());
            throw ex;
          }
        }
    );
    // non-fatal CUDA error will not fail subsequent CUDA calls
    try (ColumnVector cv = ColumnVector.fromBoxedInts(1, 2, 3, 4, 5)) {
    }
  }

  public DeviceMemoryBuffer makeDeviceMemoryBufferSrc(long size) {
    byte[] data = new byte[(int)size];
    for (int i = 0;  i < size; i++) {
      data[i] = (byte)(i % 256);
    }

    try(DeviceMemoryBuffer res = DeviceMemoryBuffer.allocate(size);
        ColumnVector dData = ColumnVector.fromBytes(data)) {
      BaseDeviceMemoryBuffer b = dData.getData();
      res.copyFromMemoryBuffer(0, b, 0, size, Cuda.DEFAULT_STREAM);
      res.incRefCount();
      return res;
    }
  }

  public HostMemoryBuffer makeHostMemoryBufferSrc(long size) {
    byte[] data = new byte[(int)size];
    for (int i = 0;  i < size; i++) {
      data[i] = (byte)(i % 256);
    }

    try(HostMemoryBuffer res = HostMemoryBuffer.allocate(size, false);
        HostColumnVector hData = HostColumnVector.fromBytes(data)) {
      HostMemoryBuffer b = hData.getData();
      res.copyFromMemoryBuffer(0, b, 0, size, Cuda.DEFAULT_STREAM);
      res.incRefCount();
      return res;
    }
  }

  @Test
  public void testBatchedMemcpyH2DSingle() {
    try(DeviceMemoryBuffer dst = DeviceMemoryBuffer.allocate(2048); 
        HostMemoryBuffer src = makeHostMemoryBufferSrc(2048)) {
      long[] dstAddrs = new long[1]; 
      long[] srcAddrs = new long[1]; 
      long[] sizes = new long[1]; 
      long offset = 0;
      long offset2 = 0;
      for (int i = 0 ; i < sizes.length; i++) {
        srcAddrs[i] = src.getAddress() + offset;
        dstAddrs[i] = dst.getAddress() + offset2;
        sizes[i] = 100;
        offset += 100;
        offset2 += 200;
      }
      Cuda.multiBufferCopyAsync(dstAddrs, srcAddrs, sizes, Cuda.DEFAULT_STREAM);
      Cuda.DEFAULT_STREAM.sync();
      for (int i = 0; i < 1; i++) {
        System.out.println((byte)(src.getByte(srcAddrs[i] - src.getAddress())));
      }
      try (HostMemoryBuffer test = HostMemoryBuffer.allocate(2048)) {
        test.copyFromMemoryBuffer(0, dst, 0, 2048, Cuda.DEFAULT_STREAM);
        Cuda.DEFAULT_STREAM.sync();
        for (int i = 0; i < 1; i++) {
          assert test.getByte(dstAddrs[i] - dst.getAddress()) == 
                 src.getByte(srcAddrs[i] - src.getAddress());
        }
      }
    }
  }

  @Test
  public void testBatchedMemcpyH2D() {
    try(DeviceMemoryBuffer dst = DeviceMemoryBuffer.allocate(2048); 
        HostMemoryBuffer src = makeHostMemoryBufferSrc(2048)) {
      long[] dstAddrs = new long[10]; 
      long[] srcAddrs = new long[10]; 
      long[] sizes = new long[10]; 
      long offset = 0;
      long offset2 = 0;
      for (int i = 0 ; i < sizes.length; i++) {
        srcAddrs[i] = src.getAddress() + offset;
        dstAddrs[i] = dst.getAddress() + offset2;
        sizes[i] = 100;
        offset += 100;
        offset2 += 200;
      }
      Cuda.multiBufferCopyAsync(dstAddrs, srcAddrs, sizes, Cuda.DEFAULT_STREAM);
      Cuda.DEFAULT_STREAM.sync();
      for (int i = 0; i < 10; i++) {
        System.out.println((byte)(src.getByte(srcAddrs[i] - src.getAddress())));
      }
      try (HostMemoryBuffer test = HostMemoryBuffer.allocate(2048)) {
        test.copyFromMemoryBuffer(0, dst, 0, 2048, Cuda.DEFAULT_STREAM);
        Cuda.DEFAULT_STREAM.sync();
        for (int i = 0; i < 10; i++) {
          assert test.getByte(dstAddrs[i] - dst.getAddress()) == 
                 src.getByte(srcAddrs[i] - src.getAddress());
        }
      }
    }
  }

  @Test
  public void testBatchedMemcpyCombined2D() {
    try(DeviceMemoryBuffer dst = DeviceMemoryBuffer.allocate(2048); 
        HostMemoryBuffer src = makeHostMemoryBufferSrc(2048);
        DeviceMemoryBuffer src2 = makeDeviceMemoryBufferSrc(2048)) {
      long[] dstAddrs = new long[10]; 
      long[] srcAddrs = new long[10]; 
      long[] sizes = new long[10]; 
      long offset = 0;
      long offset2 = 0;
      for (int i = 0 ; i < sizes.length; i++) {
        if (i % 2 == 0) {
          srcAddrs[i] = src.getAddress() + offset;
        } else {
          srcAddrs[i] = src2.getAddress() + offset;
        }
        dstAddrs[i] = dst.getAddress() + offset2;
        sizes[i] = 100;
        offset += 100;
        offset2 += 200;
      }
      Cuda.multiBufferCopyAsync(dstAddrs, srcAddrs, sizes, Cuda.DEFAULT_STREAM);
      Cuda.DEFAULT_STREAM.sync();

      try (HostMemoryBuffer test = HostMemoryBuffer.allocate(2048);
           HostMemoryBuffer hSrc2 = HostMemoryBuffer.allocate(2048)) {
        hSrc2.copyFromMemoryBuffer(0, src2, 0, 2048, Cuda.DEFAULT_STREAM);
        test.copyFromMemoryBuffer(0, dst, 0, 2048, Cuda.DEFAULT_STREAM);
        for (int i = 0; i < 10; i++) {
          byte srcByte = i % 2  == 0 ? src.getByte(srcAddrs[i] - src.getAddress()) : 
                                       hSrc2.getByte(srcAddrs[i] - src2.getAddress());
          assert test.getByte(dstAddrs[i] - dst.getAddress()) == srcByte;
        }
      }
    }
  }

}
