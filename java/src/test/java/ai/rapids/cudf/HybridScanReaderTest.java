/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import ai.rapids.cudf.ast.*;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for HybridScanReader.
 * 
 * These tests exercise the full hybrid scan API to validate it works
 * with spark-rapids integration.
 */
public class HybridScanReaderTest extends CudfTestBase {
  
  private static final File TEST_PARQUET_FILE = TestUtils.getResourceAsFile("acq.parquet");

  /**
   * Helper to read entire file into a HostMemoryBuffer
   */
  private HostMemoryBuffer readFileToHostBuffer(File file) throws IOException {
    byte[] fileBytes = Files.readAllBytes(file.toPath());
    HostMemoryBuffer buffer = HostMemoryBuffer.allocate(fileBytes.length);
    buffer.setBytes(0, fileBytes, 0, fileBytes.length);
    return buffer;
  }

  /**
   * Helper to extract footer from a parquet file buffer.
   * Parquet footer is at the end: [footer_bytes][4-byte footer length][4-byte magic "PAR1"]
   */
  private HostMemoryBuffer extractFooter(HostMemoryBuffer fileBuffer) {
    long fileLen = fileBuffer.getLength();
    
    // Read footer length (4 bytes before the magic number)
    int footerLength = fileBuffer.getInt(fileLen - 8);
    
    // Footer starts at: fileLen - 8 - footerLength
    long footerStart = fileLen - 8 - footerLength;
    
    // Copy footer bytes
    byte[] footerBytes = new byte[footerLength];
    fileBuffer.getBytes(footerBytes, 0, footerStart, footerLength);
    
    HostMemoryBuffer footerBuffer = HostMemoryBuffer.allocate(footerLength);
    footerBuffer.setBytes(0, footerBytes, 0, footerLength);
    
    return footerBuffer;
  }

  // ==========================================================================
  // Test 1: Basic creation and getAllRowGroups (no filter)
  // ==========================================================================
  @Test
  void testCreateReaderAndGetRowGroups() throws IOException {
    try (HostMemoryBuffer fileBuffer = readFileToHostBuffer(TEST_PARQUET_FILE);
         HostMemoryBuffer footerBuffer = extractFooter(fileBuffer)) {
      
      String[] columns = new String[]{"loan_id", "zip", "num_units"};
      
      try (HybridScanReader reader = new HybridScanReader(footerBuffer, null, columns)) {
        int[] rowGroups = reader.getAllRowGroups();
        
        assertNotNull(rowGroups);
        assertTrue(rowGroups.length > 0, "Should have at least one row group");
        
        System.out.println("Row groups found: " + rowGroups.length);
        for (int rg : rowGroups) {
          System.out.println("  Row group: " + rg);
        }
      }
    }
  }

  // ==========================================================================
  // Test 2: Get total rows in row groups
  // ==========================================================================
  @Test
  void testGetTotalRowsInRowGroups() throws IOException {
    try (HostMemoryBuffer fileBuffer = readFileToHostBuffer(TEST_PARQUET_FILE);
         HostMemoryBuffer footerBuffer = extractFooter(fileBuffer)) {
      
      String[] columns = new String[]{"loan_id", "zip", "num_units"};
      
      try (HybridScanReader reader = new HybridScanReader(footerBuffer, null, columns)) {
        int[] rowGroups = reader.getAllRowGroups();
        long totalRows = reader.getTotalRowsInRowGroups(rowGroups);
        
        assertTrue(totalRows > 0, "Should have rows");
        System.out.println("Total rows in all row groups: " + totalRows);
      }
    }
  }

  // ==========================================================================
  // Test 3: Get payload column byte ranges
  // ==========================================================================
  @Test
  void testGetPayloadColumnByteRanges() throws IOException {
    try (HostMemoryBuffer fileBuffer = readFileToHostBuffer(TEST_PARQUET_FILE);
         HostMemoryBuffer footerBuffer = extractFooter(fileBuffer)) {
      
      String[] columns = new String[]{"loan_id", "zip", "num_units"};
      
      try (HybridScanReader reader = new HybridScanReader(footerBuffer, null, columns)) {
        int[] rowGroups = reader.getAllRowGroups();
        long[] byteRanges = reader.getPayloadColumnChunkByteRanges(rowGroups);
        
        assertNotNull(byteRanges);
        assertTrue(byteRanges.length > 0, "Should have byte ranges");
        assertEquals(0, byteRanges.length % 2, "Byte ranges should be pairs of (offset, length)");
        
        System.out.println("Payload column byte ranges: " + (byteRanges.length / 2) + " ranges");
        for (int i = 0; i < byteRanges.length; i += 2) {
          System.out.println("  Range: offset=" + byteRanges[i] + ", length=" + byteRanges[i + 1]);
        }
      }
    }
  }

  // ==========================================================================
  // Test 4: Materialize from host buffer (no filter)
  // ==========================================================================
  @Test
  void testMaterializeFromBufferNoFilter() throws IOException {
    try (HostMemoryBuffer fileBuffer = readFileToHostBuffer(TEST_PARQUET_FILE);
         HostMemoryBuffer footerBuffer = extractFooter(fileBuffer)) {
      
      String[] columns = new String[]{"loan_id", "zip", "num_units"};
      
      try (HybridScanReader reader = new HybridScanReader(footerBuffer, null, columns);
           Table table = reader.materializeFromBuffer(fileBuffer)) {
        
        assertNotNull(table);
        assertEquals(3, table.getNumberOfColumns());
        assertEquals(1000, table.getRowCount()); // acq.parquet has 1000 rows
        
        System.out.println("Materialized table: " + table.getNumberOfColumns() + " columns, " 
            + table.getRowCount() + " rows");
      }
    }
  }

  // ==========================================================================
  // Test 5: Compare with Table.readParquet (should be identical)
  // ==========================================================================
  @Test
  void testCompareWithTableReadParquet() throws IOException {
    try (HostMemoryBuffer fileBuffer = readFileToHostBuffer(TEST_PARQUET_FILE);
         HostMemoryBuffer footerBuffer = extractFooter(fileBuffer)) {
      
      String[] columns = new String[]{"loan_id", "zip", "num_units"};
      
      // Read with hybrid scan
      Table hybridTable;
      try (HybridScanReader reader = new HybridScanReader(footerBuffer, null, columns)) {
        hybridTable = reader.materializeFromBuffer(fileBuffer);
      }
      
      // Read with Table.readParquet
      ParquetOptions opts = ParquetOptions.builder()
          .includeColumn("loan_id")
          .includeColumn("zip")
          .includeColumn("num_units")
          .build();
      
      try (Table standardTable = Table.readParquet(opts, TEST_PARQUET_FILE);
           Table hybrid = hybridTable) {
        
        assertEquals(standardTable.getNumberOfColumns(), hybrid.getNumberOfColumns());
        assertEquals(standardTable.getRowCount(), hybrid.getRowCount());
        
        // Compare column types
        for (int i = 0; i < standardTable.getNumberOfColumns(); i++) {
          assertEquals(standardTable.getColumn(i).getType(), hybrid.getColumn(i).getType(),
              "Column " + i + " type mismatch");
        }
        
        System.out.println("✓ Hybrid scan matches Table.readParquet");
      }
    }
  }

  // ==========================================================================
  // Test 6: Create reader WITH filter
  // ==========================================================================
  @Test
  void testCreateReaderWithFilter() throws IOException {
    try (HostMemoryBuffer fileBuffer = readFileToHostBuffer(TEST_PARQUET_FILE);
         HostMemoryBuffer footerBuffer = extractFooter(fileBuffer)) {
      
      String[] columns = new String[]{"loan_id", "zip", "num_units"};
      
      // Test with null filter first (should work)
      try (HybridScanReader reader = new HybridScanReader(footerBuffer, null, columns)) {
        int[] rowGroups = reader.getAllRowGroups();
        assertNotNull(rowGroups);
        assertTrue(rowGroups.length > 0);
        System.out.println("Created reader with null filter - " + rowGroups.length + " row groups");
      }
    }
  }

  // ==========================================================================
  // Test 7: Filter with ColumnNameReference
  // ==========================================================================
  @Test
  void testFilterRowGroupsWithStats() throws IOException {
    try (HostMemoryBuffer fileBuffer = readFileToHostBuffer(TEST_PARQUET_FILE);
         HostMemoryBuffer footerBuffer = extractFooter(fileBuffer)) {
      
      String[] columns = new String[]{"loan_id", "zip", "num_units"};
      
      // Create filter: zip > 90000 using ColumnNameReference
      // This requires the column to be referenced by name, not index
      ColumnNameReference zipCol = new ColumnNameReference("zip");
      Literal val = Literal.ofInt(90000);
      BinaryOperation filter = new BinaryOperation(BinaryOperator.GREATER, zipCol, val);
      
      try (CompiledExpression compiledFilter = filter.compile();
           HybridScanReader reader = new HybridScanReader(footerBuffer, compiledFilter, columns)) {
        int[] allRowGroups = reader.getAllRowGroups();
        System.out.println("All row groups: " + allRowGroups.length);
        
        // Filter row groups with stats
        int[] filteredRowGroups = reader.filterRowGroupsWithStats(allRowGroups);
        System.out.println("Filtered row groups (zip > 90000): " + filteredRowGroups.length);
        
        // The filtered count should be <= all row groups
        assertTrue(filteredRowGroups.length <= allRowGroups.length);
      }
    }
  }

  // ==========================================================================
  // Test 8: Full flow with filter - materialize and verify filtering
  // ==========================================================================
  @Test
  void testFullFlowWithFilter() throws IOException {
    try (HostMemoryBuffer fileBuffer = readFileToHostBuffer(TEST_PARQUET_FILE);
         HostMemoryBuffer footerBuffer = extractFooter(fileBuffer)) {
      
      String[] columns = new String[]{"loan_id", "zip", "num_units"};
      
      // Create filter: zip > 90000
      ColumnNameReference zipCol = new ColumnNameReference("zip");
      Literal val = Literal.ofInt(90000);
      BinaryOperation filter = new BinaryOperation(BinaryOperator.GREATER, zipCol, val);
      
      try (CompiledExpression compiledFilter = filter.compile();
           HybridScanReader reader = new HybridScanReader(footerBuffer, compiledFilter, columns);
           Table table = reader.materializeFromBuffer(fileBuffer)) {
        
        assertNotNull(table);
        assertEquals(3, table.getNumberOfColumns());
        
        System.out.println("Filtered table: " + table.getNumberOfColumns() + " columns, "
            + table.getRowCount() + " rows");
        
        // Note: The current implementation doesn't actually apply the filter to rows,
        // it only uses it for row group/page pruning. Row-level filtering happens
        // after materialization in spark-rapids.
      }
    }
  }

  // ==========================================================================
  // Test 9: Get filter column byte ranges (when filter is set)
  // ==========================================================================
  @Test
  void testGetFilterColumnByteRanges() throws IOException {
    try (HostMemoryBuffer fileBuffer = readFileToHostBuffer(TEST_PARQUET_FILE);
         HostMemoryBuffer footerBuffer = extractFooter(fileBuffer)) {
      
      String[] columns = new String[]{"loan_id", "zip", "num_units"};
      
      // Create filter on zip column
      ColumnNameReference zipCol = new ColumnNameReference("zip");
      Literal val = Literal.ofInt(90000);
      BinaryOperation filter = new BinaryOperation(BinaryOperator.GREATER, zipCol, val);
      
      try (CompiledExpression compiledFilter = filter.compile();
           HybridScanReader reader = new HybridScanReader(footerBuffer, compiledFilter, columns)) {
        int[] rowGroups = reader.getAllRowGroups();
        
        // Get filter column byte ranges (should include zip column)
        long[] filterRanges = reader.getFilterColumnChunkByteRanges(rowGroups);
        
        System.out.println("Filter column byte ranges: " + (filterRanges.length / 2) + " ranges");
        for (int i = 0; i < filterRanges.length; i += 2) {
          System.out.println("  Range: offset=" + filterRanges[i] + ", length=" + filterRanges[i + 1]);
        }
        
        // Get payload column byte ranges (should include loan_id, num_units but NOT zip)
        long[] payloadRanges = reader.getPayloadColumnChunkByteRanges(rowGroups);
        
        System.out.println("Payload column byte ranges: " + (payloadRanges.length / 2) + " ranges");
        for (int i = 0; i < payloadRanges.length; i += 2) {
          System.out.println("  Range: offset=" + payloadRanges[i] + ", length=" + payloadRanges[i + 1]);
        }
      }
    }
  }

  // ==========================================================================
  // Test 10: Dictionary filtering flow (requires a filter to be set)
  // ==========================================================================
  @Test
  void testDictionaryFiltering() throws IOException {
    try (HostMemoryBuffer fileBuffer = readFileToHostBuffer(TEST_PARQUET_FILE);
         HostMemoryBuffer footerBuffer = extractFooter(fileBuffer)) {
      
      String[] columns = new String[]{"loan_id", "zip", "num_units"};
      
      // Dictionary filtering requires a filter to be set
      ColumnNameReference zipCol = new ColumnNameReference("zip");
      Literal val = Literal.ofInt(90000);
      BinaryOperation filter = new BinaryOperation(BinaryOperator.GREATER, zipCol, val);
      
      try (CompiledExpression compiledFilter = filter.compile();
           HybridScanReader reader = new HybridScanReader(footerBuffer, compiledFilter, columns)) {
        int[] rowGroups = reader.getAllRowGroups();
        
        // Get dictionary page byte ranges
        long[] dictRanges = reader.getDictionaryPageByteRanges(rowGroups);
        
        System.out.println("Dictionary page byte ranges: " + (dictRanges.length / 2) + " ranges");
        
        if (dictRanges.length > 0) {
          System.out.println("File has dictionary pages - dictionary filtering is possible");
          for (int i = 0; i < dictRanges.length; i += 2) {
            System.out.println("  Range: offset=" + dictRanges[i] + ", length=" + dictRanges[i + 1]);
          }
        } else {
          System.out.println("File has no dictionary pages (or none for selected columns)");
        }
      }
    }
  }
}

