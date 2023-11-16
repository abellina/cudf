package ai.rapids.cudf;

class THP {
    public native static long allocate(long bytes);
    public native static void free(long address, long bytes);
    private static final long UNSAFE_COPY_THRESHOLD = 1024L * 1024L;
    public static void copyMemory(byte[] src, long srcOffset, byte[] dst, long dstOffset,
                                    long length) {
        // Check if dstOffset is before or after srcOffset to determine if we should copy
        // forward or backwards. This is necessary in case src and dst overlap.
        if (dstOffset < srcOffset) {
        while (length > 0) {
            long size = Math.min(length, UNSAFE_COPY_THRESHOLD);
            copyMemoryNative(src, srcOffset, dst, dstOffset, size);
            length -= size;
            srcOffset += size;
            dstOffset += size;
        }
        } else {
        srcOffset += length;
        dstOffset += length;
        while (length > 0) {
            long size = Math.min(length, UNSAFE_COPY_THRESHOLD);
            srcOffset -= size;
            dstOffset -= size;
            copyMemoryNative(src, srcOffset, dst, dstOffset, size);
            length -= size;
        }

        }
    }
    public static void copyMemory(long srcOffset, long dstOffset,
                                    long length) {
        copyMemory(null, srcOffset, null, dstOffset, length);
    }

    public static void setBytes(long address, byte[] values, long offset, long len) {
        copyMemory(values, UnsafeMemoryAccessor.BYTE_ARRAY_OFFSET + offset,
            null, address, len);
    }

    public static void getBytes(byte[] dst, long dstOffset, long address, long len) {
        copyMemory(null, address,
            dst, UnsafeMemoryAccessor.BYTE_ARRAY_OFFSET + dstOffset, len);
    }

    private static native void copyMemoryNative(
        byte[] src, long srcOffset, byte[] dst, long dstOffset, long length); 
}
