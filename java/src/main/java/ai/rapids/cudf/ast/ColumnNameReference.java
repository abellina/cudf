/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * A reference to a column in an input table by name.
 * This is useful when the column index is not known at AST construction time,
 * but the column name is available. The column name will be resolved to an index
 * when the expression is evaluated using table metadata.
 */
public final class ColumnNameReference extends AstExpression {
  private final String columnName;

  /**
   * Construct a column reference using the column name.
   * @param columnName the name of the column to reference
   */
  public ColumnNameReference(String columnName) {
    if (columnName == null || columnName.isEmpty()) {
      throw new IllegalArgumentException("Column name cannot be null or empty");
    }
    this.columnName = columnName;
  }

  /**
   * Get the column name.
   * @return the column name
   */
  public String getColumnName() {
    return columnName;
  }

  @Override
  int getSerializedSize() {
    byte[] nameBytes = columnName.getBytes(StandardCharsets.UTF_8);
    // node type + string length (int) + string bytes
    return ExpressionType.COLUMN_NAME_REFERENCE.getSerializedSize() +
        Integer.BYTES +
        nameBytes.length;
  }

  @Override
  void serialize(ByteBuffer bb) {
    byte[] nameBytes = columnName.getBytes(StandardCharsets.UTF_8);
    ExpressionType.COLUMN_NAME_REFERENCE.serialize(bb);
    bb.putInt(nameBytes.length);
    bb.put(nameBytes);
  }

  @Override
  public String toString() {
    return "COLUMN(\"" + columnName + "\")";
  }
}

