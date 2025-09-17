package ai.rapids.cudf;

/**
 * Java wrapper for cudf::join_match_context.
 */
public class JoinMatchContext implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  // Native pointer to a cudf::join_match_context (owned by this wrapper)
  private long nativeHandle;

  private JoinMatchContext(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  /**
   * Create a JoinMatchContext by computing inner-join match counts using a pre-built hash join.
   *
   * @param hashJoinNativeHandle native handle to an existing cudf::hash_join
   * @param build cudf Table to use for match counting
   */
  public static JoinMatchContext fromHashJoinInner(long hashJoinNativeHandle, Table build) {
    long ctx = createFromHashJoinInner(hashJoinNativeHandle, build.getNativeView());
    return new JoinMatchContext(ctx);
  }

  /**
   * Create a JoinMatchContext by computing left-join match counts using a pre-built hash join.
   */
  public static JoinMatchContext fromHashJoinLeft(long hashJoinNativeHandle, Table build) {
    long ctx = createFromHashJoinLeft(hashJoinNativeHandle, build.getNativeView());
    return new JoinMatchContext(ctx);
  }

  /**
   * Create a JoinMatchContext by computing full-join match counts using a pre-built hash join.
   */
  public static JoinMatchContext fromHashJoinFull(long hashJoinNativeHandle, Table build) {
    long ctx = createFromHashJoinFull(hashJoinNativeHandle, build.getNativeView());
    return new JoinMatchContext(ctx);
  }

  /**
   * Returns the match counts as a host array (long[]) with one element per row.
   */
  public long[] getMatchCounts() {
    return exportMatchCounts(nativeHandle);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      closeNative(nativeHandle);
      nativeHandle = 0;
    }
  }

  // Native methods
  private static native long createFromHashJoinInner(long hashJoinHandle, long leftTableView);
  private static native long createFromHashJoinLeft(long hashJoinHandle, long leftTableView);
  private static native long createFromHashJoinFull(long hashJoinHandle, long leftTableView);
  private static native long[] exportMatchCounts(long joinMatchContextHandle);
  private static native void closeNative(long joinMatchContextHandle);
}
