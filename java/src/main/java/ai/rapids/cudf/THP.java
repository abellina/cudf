package ai.rapids.cudf;

class THP {
    public native static long allocate(long bytes);
    public native static void free(long address, long bytes);
    private static final long UNSAFE_COPY_THRESHOLD = 1024L * 1024L;
    public static void copyMemory(byte[] src, long srcOffset, byte[] dst, long dstOffset,
                                    long length) {
        if (dstOffset < srcOffset) {
        while (length > 0) {
            long size = Math.min(length, UNSAFE_COPY_THRESHOLD);
            //System.out.println(
            //    "copying src: " + src + 
            //    " srcOffset: " + srcOffset + 
            //    " dst: " + dst + 
            //    " dstOffset: " + dstOffset +
            //    " size: " + size);
            copyMemoryNative(src, srcOffset, dst, dstOffset, size);
            //System.out.println(
            //    "done with copy src: " + src + 
            //    " srcOffset: " + srcOffset + 
            //    " dst: " + dst + 
            //    " dstOffset: " + dstOffset +
            //    " size: " + size);
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
            //System.out.println(
            //    "copying src: " + src + 
            //    " srcOffset: " + srcOffset + 
            //    " dst: " + dst + 
            //    " dstOffset: " + dstOffset +
            //    " size: " + size);
            copyMemoryNative(src, srcOffset, dst, dstOffset, size);
            //System.out.println(
            //    "done with copy src: " + src + 
            //    " srcOffset: " + srcOffset + 
            //    " dst: " + dst + 
            //    " dstOffset: " + dstOffset +
            //    " size: " + size);
            length -= size;
        }

        }
    }
    public static void copyMemory(long srcOffset, long dstOffset,
                                    long length) {
        copyMemory(null, srcOffset, null, dstOffset, length);
    }

    public static void setBytes(long address, byte[] values, long offset, long len) {
        copyMemory(values, offset, null, address, len);
    }

    public static void getBytes(byte[] dst, long dstOffset, long address, long len) {
        copyMemory(null, address, dst, dstOffset, len);
    }

    private static native void copyMemoryNative(
        byte[] src, long srcOffset, byte[] dst, long dstOffset, long length); 
}
