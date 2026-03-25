/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import ai.rapids.cudf.ast.CompiledExpression;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Options for reading a parquet file
 */
public class ParquetOptions extends ColumnFilterOptions {

  public static ParquetOptions DEFAULT = new ParquetOptions(new Builder());

  private final DType unit;
  private final boolean[] readBinaryAsString;
  private final CompiledExpression filter;

  private ParquetOptions(Builder builder) {
    super(builder);
    unit = builder.unit;
    readBinaryAsString = new boolean[builder.binaryAsStringColumns.size()];
    for (int i = 0 ; i < builder.binaryAsStringColumns.size() ; i++) {
      readBinaryAsString[i] = builder.binaryAsStringColumns.get(i);
    }
    filter = builder.filter;
  }

  DType timeUnit() {
    return unit;
  }

  boolean[] getReadBinaryAsString() {
    return readBinaryAsString;
  }

  /**
   * Returns the filter expression to apply during parquet reading, or null if no filter.
   * The filter uses AST expressions to enable row group filtering based on statistics,
   * dictionary filtering, and optionally bloom filter filtering.
   */
  CompiledExpression getFilter() {
    return filter;
  }

  /**
   * Returns true if a filter expression has been set.
   */
  boolean hasFilter() {
    return filter != null;
  }

  public static ParquetOptions.Builder builder() {
    return new Builder();
  }

  public static class Builder extends ColumnFilterOptions.Builder<Builder> {
    private DType unit = DType.EMPTY;
    final List<Boolean> binaryAsStringColumns = new ArrayList<>();
    private CompiledExpression filter = null;

    /**
     * Specify the time unit to use when returning timestamps.
     * @param unit default unit of time specified by the user
     * @return builder for chaining
     */
    public Builder withTimeUnit(DType unit) {
      assert unit.isTimestampType();
      this.unit = unit;
      return this;
    }

    /**
     * Set a filter expression to apply during parquet reading.
     * The filter uses AST expressions to enable row group filtering based on statistics,
     * dictionary filtering, and optionally bloom filter filtering.
     * 
     * Note: The caller is responsible for closing the CompiledExpression after the
     * parquet read operation is complete.
     * 
     * @param filter the compiled AST expression to use as a filter
     * @return builder for chaining
     */
    public Builder withFilter(CompiledExpression filter) {
      this.filter = filter;
      return this;
    }

    /**
     * Include one or more specific columns.  Any column not included will not be read.
     * @param names the name of the column, or more than one if you want.
     */
    @Override
    public Builder includeColumn(String... names) {
      super.includeColumn(names);
      for (int i = 0 ; i < names.length ; i++) {
        binaryAsStringColumns.add(true);
      }
      return this;
    }

    /**
     * Include this column.
     * @param name the name of the column
     * @param isBinary whether this column is to be read in as binary
     */
    public Builder includeColumn(String name, boolean isBinary) {
      includeColumnNames.add(name);
      binaryAsStringColumns.add(!isBinary);
      return this;
    }

    /**
     * Include one or more specific columns.  Any column not included will not be read.
     * @param names the name of the column, or more than one if you want.
     */
    @Override
    public Builder includeColumn(Collection<String> names) {
      super.includeColumn(names);
      for (int i = 0 ; i < names.size() ; i++) {
        binaryAsStringColumns.add(true);
      }
      return this;
    }

    public ParquetOptions build() {
      return new ParquetOptions(this);
    }
  }
}
