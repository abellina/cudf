/*
 *
 *  Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

/**
 * This class represents settings for writing Parquet files. It includes meta data information
 * that will be used by the Parquet writer to write the file
 */
public final class ParquetWriterOptions extends CompressionMetadataWriterOptions {
  private final StatisticsFrequency statsGranularity;
  private final long uncompressedRowGroupSizeBytes;
  private final long rowGroupSizeRows;
  private final int dictionaryPolicy;

  private ParquetWriterOptions(Builder builder) {
    super(builder);
    this.statsGranularity = builder.statsGranularity;
    this.uncompressedRowGroupSizeBytes = builder.uncompressedRowGroupSizeBytes;
    this.rowGroupSizeRows = builder.rowGroupSizeRows;
    this.dictionaryPolicy = builder.dictionaryPolicy;
  }

  public enum StatisticsFrequency {
    /** Do not generate statistics */
    NONE(0),

    /** Generate column statistics for each rowgroup */
    ROWGROUP(1),

    /** Generate column statistics for each page */
    PAGE(2);

    final int nativeId;

    StatisticsFrequency(int nativeId) {
      this.nativeId = nativeId;
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  public StatisticsFrequency getStatisticsFrequency() {
    return statsGranularity;
  }

  public long getUncompressedRowGroupSizeBytes() {
    return uncompressedRowGroupSizeBytes;
  }

  public long getRowGroupSizeRows() {
    return rowGroupSizeRows;
  }


  public int getDictionaryPolicy() {
    return dictionaryPolicy;
  }

  public static class Builder extends CompressionMetadataWriterOptions.Builder
        <Builder, ParquetWriterOptions> {
    private StatisticsFrequency statsGranularity = StatisticsFrequency.ROWGROUP;
    private long uncompressedRowGroupSizeBytes = 0L;
    private long rowGroupSizeRows = 0L;
    private int dictionaryPolicy = 2;

    public Builder() {
      super();
    }

    public Builder withStatisticsFrequency(StatisticsFrequency statsGranularity) {
      this.statsGranularity = statsGranularity;
      return this;
    }

    public Builder withUncompressedRowGroupSizeBytes(long uncompressedRowGroupSizeBytes) {
      this.uncompressedRowGroupSizeBytes = uncompressedRowGroupSizeBytes;
      return this;
    }

    public Builder withRowGroupSizeRows(long rowGroupSizeRows) {
      this.rowGroupSizeRows = rowGroupSizeRows;
      return this;
    }

    public Builder withDictionaryPolicy(int dictionaryPolicy) {
      this.dictionaryPolicy = dictionaryPolicy;
      return this;
    }

    public ParquetWriterOptions build() {
      return new ParquetWriterOptions(this);
    }
  }
}
