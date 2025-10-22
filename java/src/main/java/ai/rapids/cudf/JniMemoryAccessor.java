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

import java.lang.reflect.Field;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * JNI-based Memory Accessor for accessing memory on host without using sun.misc.Unsafe.
 * This class provides all the functionality of UnsafeMemoryAccessor but implemented
 * through native JNI calls for better compatibility and safety.
 */
class JniMemoryAccessor {

  /**
   * Array base offsets for different primitive types.
   * These are computed in native code to match the JVM's internal layout.
   */
  public static final long BYTE_ARRAY_OFFSET;
  public static final long SHORT_ARRAY_OFFSET;
  public static final long INT_ARRAY_OFFSET;
  public static final long LONG_ARRAY_OFFSET;
  public static final long FLOAT_ARRAY_OFFSET;
  public static final long DOUBLE_ARRAY_OFFSET;

  private static final Logger log = LoggerFactory.getLogger(JniMemoryAccessor.class);

  static {
    sun.misc.Unsafe unsafe = null;
    try {
      Field unsafeField = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
      unsafeField.setAccessible(true);
      unsafe = (sun.misc.Unsafe) unsafeField.get(null);
      BYTE_ARRAY_OFFSET = unsafe.arrayBaseOffset(byte[].class);
      SHORT_ARRAY_OFFSET = unsafe.arrayBaseOffset(short[].class);
      INT_ARRAY_OFFSET = unsafe.arrayBaseOffset(int[].class);
      LONG_ARRAY_OFFSET = unsafe.arrayBaseOffset(long[].class);
      FLOAT_ARRAY_OFFSET = unsafe.arrayBaseOffset(float[].class);
      DOUBLE_ARRAY_OFFSET = unsafe.arrayBaseOffset(double[].class);
    } catch (Throwable t) {
      log.error("Failed to get unsafe object, got this error: ", t);
      throw new NullPointerException("Failed to get unsafe object, got this error: " + t.getMessage());
    }
  }

  /**
   * Get the system memory page size.
   * @return system memory page size in bytes
   */
  public static native int pageSize();

  /**
   * Allocate bytes on host
   * @param bytes - number of bytes to allocate
   * @return - allocated address
   */
  public static native long allocate(long bytes);

  /**
   * Free memory at that location
   * @param address - memory location
   */
  public static native void free(long address);

  /**
   * Sets the values at this address repeatedly
   * @param address - memory location
   * @param size    - number of bytes to set
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static native void setMemory(long address, long size, byte value);

  /**
   * Sets the Byte value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static native void setByte(long address, byte value);

  /**
   * Sets an array of bytes.
   * @param address - memory address
   * @param values  to be set
   * @param offset  index into values to start at.
   * @param len     the number of bytes to copy
   * @throws IndexOutOfBoundsException
   */
  public static void setBytes(long address, byte[] values, long offset, long len) {
    copyMemory(values, BYTE_ARRAY_OFFSET + offset, null, address, len);
  }

  /**
   * Returns the Byte value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static native byte getByte(long address);

  /**
   * Copy out an array of bytes.
   * @param dst       where to write the data
   * @param dstOffset index into values to start writing at.
   * @param address   src memory address
   * @param len       the number of bytes to copy
   * @throws IndexOutOfBoundsException
   */
  public static void getBytes(byte[] dst, long dstOffset, long address, long len) {
    copyMemory(null, address, dst, BYTE_ARRAY_OFFSET + dstOffset, len);
  }

  /**
   * Returns the Integer value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static native int getInt(long address);

  /**
   * Copy out an array of ints.
   * @param dst       where to write the data
   * @param dstIndex  index into values to start writing at.
   * @param address   src memory address
   * @param count     the number of ints to copy
   * @throws IndexOutOfBoundsException
   */
  public static void getInts(int[] dst, long dstIndex, long address, int count) {
    copyMemory(null, address, dst, INT_ARRAY_OFFSET + (dstIndex * 4), count * 4L);
  }

  /**
   * Sets the Integer value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static native void setInt(long address, int value);

  /**
   * Sets an array of ints.
   * @param address memory address
   * @param values  to be set
   * @param offset  index into values to start at.
   * @param len     the number of ints to copy
   * @throws IndexOutOfBoundsException
   */
  public static void setInts(long address, int[] values, long offset, long len) {
    copyMemory(values, INT_ARRAY_OFFSET + (offset * 4), null, address, len * 4);
  }

  /**
   * Sets the Long value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static native void setLong(long address, long value);

  /**
   * Sets an array of longs.
   * @param address memory address
   * @param values  to be set
   * @param offset  index into values to start at
   * @param len     the number of longs to copy
   * @throws IndexOutOfBoundsException
   */
  public static void setLongs(long address, long[] values, long offset, long len) {
    copyMemory(values, LONG_ARRAY_OFFSET + (offset * 8), null, address, len * 8);
  }

  /**
   * Returns the Long value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static native long getLong(long address);

  /**
   * Copy out an array of longs.
   * @param dst       where to write the data
   * @param dstIndex  index into values to start writing at.
   * @param address   src memory address
   * @param count     the number of longs to copy
   * @throws IndexOutOfBoundsException
   */
  public static void getLongs(long[] dst, long dstIndex, long address, int count) {
    copyMemory(null, address, dst, LONG_ARRAY_OFFSET + (dstIndex * 8), count * 8L);
  }

  /**
   * Returns the Short value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static native short getShort(long address);

  /**
   * Sets the Short value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static native void setShort(long address, short value);

  /**
   * Sets an array of shorts.
   * @param address memory address
   * @param values  to be set
   * @param offset  index into values to start at
   * @param len     the number of shorts to copy
   * @throws IndexOutOfBoundsException
   */
  public static void setShorts(long address, short[] values, long offset, long len) {
    copyMemory(values, SHORT_ARRAY_OFFSET + (offset * 2), null, address, len * 2);
  }

  /**
   * Sets the Double value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static native void setDouble(long address, double value);

  /**
   * Sets an array of doubles.
   * @param address memory address
   * @param values  to be set
   * @param offset  index into values to start at
   * @param len     the number of doubles to copy
   * @throws IndexOutOfBoundsException
   */
  public static void setDoubles(long address, double[] values, long offset, long len) {
    copyMemory(values, DOUBLE_ARRAY_OFFSET + (offset * 8), null, address, len * 8);
  }

  /**
   * Returns the Double value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static native double getDouble(long address);

  /**
   * Returns the Float value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static native float getFloat(long address);

  /**
   * Sets the Float value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static native void setFloat(long address, float value);

  /**
   * Sets an array of floats.
   * @param address memory address
   * @param values  to be set
   * @param offset  the index in values to start at
   * @param len     the number of floats to copy
   * @throws IndexOutOfBoundsException
   */
  public static void setFloats(long address, float[] values, long offset, long len) {
    copyMemory(values, FLOAT_ARRAY_OFFSET + (offset * 4), null, address, len * 4);
  }

  /**
   * Returns the Boolean value at this address
   * @param address - memory address
   * @return - value
   * @throws IndexOutOfBoundsException
   */
  public static boolean getBoolean(long address) {
    return getByte(address) != 0;
  }

  /**
   * Sets the Boolean value at that address
   * @param address - memory address
   * @param value   - value to be set
   * @throws IndexOutOfBoundsException
   */
  public static void setBoolean(long address, boolean value) {
    setByte(address, (byte) (value ? 1 : 0));
  }

  /**
   * Copy memory from one address to another, or between arrays and native memory.
   * Handles overlapping regions correctly by choosing forward or backward copy direction.
   * 
   * @param src source object (array) or null for native memory
   * @param srcOffset offset in source (array index offset or native address)
   * @param dst destination object (array) or null for native memory
   * @param dstOffset offset in destination (array index offset or native address)
   * @param length number of bytes to copy
   */
  public static void copyMemory(Object src, long srcOffset, Object dst, long dstOffset,
                                long length) {
    copyMemoryNative(src, srcOffset, dst, dstOffset, length);
  }

  /**
   * Native implementation of memory copy.
   * @param src source object (array) or null for native memory
   * @param srcOffset offset in source
   * @param dst destination object (array) or null for native memory
   * @param dstOffset offset in destination
   * @param length number of bytes to copy
   */
  private static native void copyMemoryNative(Object src, long srcOffset, 
                                              Object dst, long dstOffset, long length);
}

