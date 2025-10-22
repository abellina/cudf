/*
 *
 *  Copyright (c) 2025, NVIDIA CORPORATION.
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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("noSanitizer")
public class JniMemoryAccessorTest {
  @Test
  public void testAllocate() {
    long address = JniMemoryAccessor.allocate(3);
    try {
      assertNotEquals(0, address);
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testPageSize() {
    int pageSize = JniMemoryAccessor.pageSize();
    assertTrue(pageSize > 0);
    // Common page sizes are 4096, 8192, 16384
    assertTrue(pageSize >= 4096);
  }

  @Test
  public void setByteAndGetByte() {
    long address = JniMemoryAccessor.allocate(2);
    try {
      JniMemoryAccessor.setByte(address, (byte) 34);
      JniMemoryAccessor.setByte(address + 1, (byte) 63);
      Byte b = JniMemoryAccessor.getByte(address);
      assertEquals((byte) 34, b);
      b = JniMemoryAccessor.getByte(address + 1);
      assertEquals((byte) 63, b);
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void setIntAndGetInt() {
    long address = JniMemoryAccessor.allocate(2 * 4);
    try {
      JniMemoryAccessor.setInt(address, 2);
      JniMemoryAccessor.setInt(address + 4, 4);
      int v = JniMemoryAccessor.getInt(address);
      assertEquals(2, v);
      v = JniMemoryAccessor.getInt(address + 4);
      assertEquals(4, v);
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void setAndGetInts() {
    int numInts = 289;
    long address = JniMemoryAccessor.allocate(numInts * 4);
    try {
      for (int i = 0; i < numInts; i++) {
        JniMemoryAccessor.setInt(address + i * 4, i);
      }
      int[] ints = new int[numInts];
      JniMemoryAccessor.getInts(ints, 0, address, numInts);
      for (int i = 0; i < numInts; i++) {
        assertEquals(i, ints[i]);
      }
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void setMemoryValue() {
    long address = JniMemoryAccessor.allocate(4);
    try {
      JniMemoryAccessor.setMemory(address, 4, (byte) 1);
      int v = JniMemoryAccessor.getInt(address);
      assertEquals(16843009, v);
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testGetLongs() {
    int numLongs = 257;
    long address = JniMemoryAccessor.allocate(numLongs * 8);
    try {
      for (int i = 0; i < numLongs; ++i) {
        JniMemoryAccessor.setLong(address + (i * 8), i);
      }
      long[] result = new long[numLongs];
      JniMemoryAccessor.getLongs(result, 0, address, numLongs);
      for (int i = 0; i < numLongs; ++i) {
        assertEquals(i, result[i]);
      }
      JniMemoryAccessor.getLongs(result, 1,
          address + ((numLongs - 1) * 8), 1);
      for (int i = 0; i < numLongs; ++i) {
        long expected = (i == 1) ? numLongs - 1 : i;
        assertEquals(expected, result[i]);
      }
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testShortOperations() {
    long address = JniMemoryAccessor.allocate(2 * 2);
    try {
      JniMemoryAccessor.setShort(address, (short) 100);
      JniMemoryAccessor.setShort(address + 2, (short) 200);
      short v = JniMemoryAccessor.getShort(address);
      assertEquals((short) 100, v);
      v = JniMemoryAccessor.getShort(address + 2);
      assertEquals((short) 200, v);
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testFloatOperations() {
    long address = JniMemoryAccessor.allocate(2 * 4);
    try {
      JniMemoryAccessor.setFloat(address, 3.14f);
      JniMemoryAccessor.setFloat(address + 4, 2.71f);
      float v = JniMemoryAccessor.getFloat(address);
      assertEquals(3.14f, v, 0.001f);
      v = JniMemoryAccessor.getFloat(address + 4);
      assertEquals(2.71f, v, 0.001f);
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testDoubleOperations() {
    long address = JniMemoryAccessor.allocate(2 * 8);
    try {
      JniMemoryAccessor.setDouble(address, 3.14159);
      JniMemoryAccessor.setDouble(address + 8, 2.71828);
      double v = JniMemoryAccessor.getDouble(address);
      assertEquals(3.14159, v, 0.00001);
      v = JniMemoryAccessor.getDouble(address + 8);
      assertEquals(2.71828, v, 0.00001);
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testBooleanOperations() {
    long address = JniMemoryAccessor.allocate(2);
    try {
      JniMemoryAccessor.setBoolean(address, true);
      JniMemoryAccessor.setBoolean(address + 1, false);
      boolean v = JniMemoryAccessor.getBoolean(address);
      assertEquals(true, v);
      v = JniMemoryAccessor.getBoolean(address + 1);
      assertEquals(false, v);
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testSetAndGetBytes() {
    long address = JniMemoryAccessor.allocate(10);
    try {
      byte[] data = new byte[]{1, 2, 3, 4, 5};
      JniMemoryAccessor.setBytes(address, data, 0, 5);
      
      byte[] result = new byte[5];
      JniMemoryAccessor.getBytes(result, 0, address, 5);
      
      for (int i = 0; i < 5; i++) {
        assertEquals(data[i], result[i]);
      }
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testSetAndGetIntsArray() {
    long address = JniMemoryAccessor.allocate(20);
    try {
      int[] data = new int[]{10, 20, 30, 40, 50};
      JniMemoryAccessor.setInts(address, data, 0, 5);
      
      int[] result = new int[5];
      JniMemoryAccessor.getInts(result, 0, address, 5);
      
      for (int i = 0; i < 5; i++) {
        assertEquals(data[i], result[i]);
      }
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testSetAndGetLongsArray() {
    long address = JniMemoryAccessor.allocate(40);
    try {
      long[] data = new long[]{100L, 200L, 300L, 400L, 500L};
      JniMemoryAccessor.setLongs(address, data, 0, 5);
      
      long[] result = new long[5];
      JniMemoryAccessor.getLongs(result, 0, address, 5);
      
      for (int i = 0; i < 5; i++) {
        assertEquals(data[i], result[i]);
      }
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testSetAndGetShortsArray() {
    long address = JniMemoryAccessor.allocate(10);
    try {
      short[] data = new short[]{10, 20, 30, 40, 50};
      JniMemoryAccessor.setShorts(address, data, 0, 5);
      
      // Verify by reading individual shorts
      for (int i = 0; i < 5; i++) {
        short v = JniMemoryAccessor.getShort(address + i * 2);
        assertEquals(data[i], v);
      }
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testSetAndGetFloatsArray() {
    long address = JniMemoryAccessor.allocate(20);
    try {
      float[] data = new float[]{1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
      JniMemoryAccessor.setFloats(address, data, 0, 5);
      
      // Verify by reading individual floats
      for (int i = 0; i < 5; i++) {
        float v = JniMemoryAccessor.getFloat(address + i * 4);
        assertEquals(data[i], v, 0.001f);
      }
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testSetAndGetDoublesArray() {
    long address = JniMemoryAccessor.allocate(40);
    try {
      double[] data = new double[]{1.1, 2.2, 3.3, 4.4, 5.5};
      JniMemoryAccessor.setDoubles(address, data, 0, 5);
      
      // Verify by reading individual doubles
      for (int i = 0; i < 5; i++) {
        double v = JniMemoryAccessor.getDouble(address + i * 8);
        assertEquals(data[i], v, 0.001);
      }
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testCopyMemoryNativeToNative() {
    long srcAddress = JniMemoryAccessor.allocate(20);
    long dstAddress = JniMemoryAccessor.allocate(20);
    try {
      // Set some data in source
      for (int i = 0; i < 5; i++) {
        JniMemoryAccessor.setInt(srcAddress + i * 4, i * 10);
      }
      
      // Copy from source to destination
      JniMemoryAccessor.copyMemory(null, srcAddress, null, dstAddress, 20);
      
      // Verify destination has the same data
      for (int i = 0; i < 5; i++) {
        int v = JniMemoryAccessor.getInt(dstAddress + i * 4);
        assertEquals(i * 10, v);
      }
    } finally {
      JniMemoryAccessor.free(srcAddress);
      JniMemoryAccessor.free(dstAddress);
    }
  }

  @Test
  public void testCopyMemoryArrayToNative() {
    long address = JniMemoryAccessor.allocate(20);
    try {
      byte[] data = new byte[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      
      // Copy from array to native memory
      JniMemoryAccessor.copyMemory(data, JniMemoryAccessor.BYTE_ARRAY_OFFSET, 
                                   null, address, 10);
      
      // Verify
      for (int i = 0; i < 10; i++) {
        byte v = JniMemoryAccessor.getByte(address + i);
        assertEquals(data[i], v);
      }
    } finally {
      JniMemoryAccessor.free(address);
    }
  }

  @Test
  public void testCopyMemoryNativeToArray() {
    long address = JniMemoryAccessor.allocate(20);
    try {
      // Set some data in native memory
      for (int i = 0; i < 10; i++) {
        JniMemoryAccessor.setByte(address + i, (byte) (i + 1));
      }
      
      byte[] result = new byte[10];
      
      // Copy from native memory to array
      JniMemoryAccessor.copyMemory(null, address, 
                                   result, JniMemoryAccessor.BYTE_ARRAY_OFFSET, 10);
      
      // Verify
      for (int i = 0; i < 10; i++) {
        assertEquals((byte) (i + 1), result[i]);
      }
    } finally {
      JniMemoryAccessor.free(address);
    }
  }
}

