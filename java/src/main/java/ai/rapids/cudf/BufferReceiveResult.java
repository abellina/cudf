package ai.rapids.cudf;

public class BufferReceiveResult {
    public int blockId;
    public long packedBuffer;
    public long size;

    public BufferReceiveResult(int blockId, long packedBuffer, long size) {
        this.blockId = blockId;
        this.packedBuffer = packedBuffer;
        this.size = size;
    }
}