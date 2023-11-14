package ai.rapids.cudf;

class THP {
    public native static long allocate(long bytes);
    public native static void free(long address, long bytes);
    private static final long UNSAFE_COPY_THRESHOLD = 1024L * 1024L;
    public static void copyMemory(Object src, long srcOffset, Object dst, long dstOffset,
                                    long length) {
        // Check if dstOffset is before or after srcOffset to determine if we should copy
        // forward or backwards. This is necessary in case src and dst overlap.
        if (dstOffset < srcOffset) {
        while (length > 0) {
            long size = Math.min(length, UNSAFE_COPY_THRESHOLD);
            copyMemoryNative(srcOffset, dstOffset, size);
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
            copyMemoryNative(srcOffset, dstOffset, size);
            length -= size;
        }

        }
    }

    private static native void copyMemoryNative(
        long srcOffset, long dstOffset, long length); 
}
