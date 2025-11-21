/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for NVTX Counter APIs.
 * 
 * Note: These tests verify that the API calls don't crash or throw exceptions.
 * The actual counter values can be observed using profiling tools like Nsight Systems.
 * To enable NVTX functionality, run with: -Dai.rapids.cudf.nvtx.enabled=true
 */
public class NvtxCounterTest {

  @Test
  public void testCounterCreation() {
    // Test basic counter creation
    try (NvtxCounter counter = NvtxCounter.create("Test Counter")) {
      assertNotNull(counter, "Counter should not be null");
      assertFalse(counter.isClosed(), "Counter should not be closed initially");
    }
  }

  @Test
  public void testCounterCreationWithDescription() {
    // Test counter creation with description
    try (NvtxCounter counter = NvtxCounter.create("Test Counter", "A test counter for unit testing")) {
      assertNotNull(counter, "Counter should not be null");
      assertFalse(counter.isClosed(), "Counter should not be closed initially");
    }
  }

  @Test
  public void testCounterCreationWithFullAttributes() {
    // Test counter creation with full attributes
    NvtxCounterAttr attr = new NvtxCounterAttr.Builder("Static ID Counter")
        .description("Counter with static ID")
        .counterId(NvtxCounterConstants.COUNTER_ID_STATIC_START + 1000)
        .build();
    
    try (NvtxCounter counter = NvtxCounter.create(attr)) {
      assertNotNull(counter, "Counter should not be null");
      assertFalse(counter.isClosed(), "Counter should not be closed initially");
    }
  }

  @Test
  public void testSampleInt64() {
    // Test sampling int64 values
    try (NvtxCounter counter = NvtxCounter.create("Int64 Counter", "Counter for 64-bit integer values")) {
      counter.sampleInt64(100L);
      counter.sampleInt64(200L);
      counter.sampleInt64(150L);
      
      // If we get here without exceptions, the test passes
      assertTrue(true, "Should be able to sample int64 values");
    }
  }

  @Test
  public void testSampleFloat64() {
    // Test sampling float64 values
    try (NvtxCounter counter = NvtxCounter.create("Float64 Counter", "Counter for 64-bit floating-point values")) {
      counter.sampleFloat64(98.6);
      counter.sampleFloat64(99.1);
      counter.sampleFloat64(97.8);
      
      // If we get here without exceptions, the test passes
      assertTrue(true, "Should be able to sample float64 values");
    }
  }

  @Test
  public void testSampleNoValue() {
    // Test sampling with no-value indicators
    try (NvtxCounter counter = NvtxCounter.create("No Value Counter", "Counter that can indicate missing values")) {
      counter.sampleInt64(100L);
      counter.sampleNoValue(NvtxCounterConstants.COUNTER_SAMPLE_UNCHANGED);
      counter.sampleNoValue(NvtxCounterConstants.COUNTER_SAMPLE_UNAVAILABLE);
      counter.sampleNoValue(NvtxCounterConstants.COUNTER_SAMPLE_ZERO);
      
      // If we get here without exceptions, the test passes
      assertTrue(true, "Should be able to sample no-value indicators");
    }
  }

  @Test
  public void testBatchSubmission() {
    // Test batch submission
    try (NvtxCounter counter = NvtxCounter.create("Batch Counter", "Counter for batch submissions")) {
      // Create a batch of 10 int64 values
      int numSamples = 10;
      ByteBuffer buffer = ByteBuffer.allocate(numSamples * 8);
      buffer.order(ByteOrder.nativeOrder());
      
      for (int i = 0; i < numSamples; i++) {
        buffer.putLong((long)(100 + i * 10));
      }
      
      byte[] counterData = buffer.array();
      
      // Create timestamps (one per sample)
      long[] timestamps = new long[numSamples];
      long baseTime = System.nanoTime();
      for (int i = 0; i < numSamples; i++) {
        timestamps[i] = baseTime + (i * 1000000); // 1ms intervals
      }
      
      // Submit the batch
      NvtxCounterBatch batch = new NvtxCounterBatch.Builder()
          .counters(counterData)
          .timestamps(timestamps)
          .build();
      
      counter.submitBatch(batch);
      
      // If we get here without exceptions, the test passes
      assertTrue(true, "Should be able to submit counter batch");
    }
  }

  @Test
  public void testBatchSubmissionWithoutTimestamps() {
    // Test batch submission without explicit timestamps
    try (NvtxCounter counter = NvtxCounter.create("Batch Counter No Timestamps")) {
      // Create a batch of 5 double values
      int numSamples = 5;
      ByteBuffer buffer = ByteBuffer.allocate(numSamples * 8);
      buffer.order(ByteOrder.nativeOrder());
      
      for (int i = 0; i < numSamples; i++) {
        buffer.putDouble(98.6 + (i * 0.5));
      }
      
      byte[] counterData = buffer.array();
      
      // Submit the batch without explicit timestamps
      NvtxCounterBatch batch = new NvtxCounterBatch.Builder()
          .counters(counterData)
          .build();
      
      counter.submitBatch(batch);
      
      // If we get here without exceptions, the test passes
      assertTrue(true, "Should be able to submit counter batch without timestamps");
    }
  }

  @Test
  public void testMultipleCounters() {
    // Test using multiple counters
    try (NvtxCounter memoryCounter = NvtxCounter.create("Memory Usage", "Current memory usage in bytes");
         NvtxCounter threadCounter = NvtxCounter.create("Active Threads", "Number of active threads");
         NvtxCounter queueCounter = NvtxCounter.create("Queue Size", "Number of items in processing queue")) {
      
      // Sample all counters
      memoryCounter.sampleInt64(1024L * 1024L * 512L); // 512 MB
      threadCounter.sampleInt64(Thread.activeCount());
      queueCounter.sampleInt64(42L);
      
      // Sample again with different values
      memoryCounter.sampleInt64(1024L * 1024L * 768L); // 768 MB
      threadCounter.sampleInt64(Thread.activeCount());
      queueCounter.sampleInt64(38L);
      
      // If we get here without exceptions, the test passes
      assertTrue(true, "Should be able to use multiple counters");
    }
  }

  @Test
  public void testCounterAutoClose() {
    // Test that counter is properly closed
    NvtxCounter counter = NvtxCounter.create("Auto Close Counter");
    assertFalse(counter.isClosed(), "Counter should not be closed initially");
    
    counter.close();
    assertTrue(counter.isClosed(), "Counter should be closed after close()");
  }

  @Test
  public void testDoubleClose() {
    // Test that double close doesn't throw
    NvtxCounter counter = NvtxCounter.create("Double Close Counter");
    counter.close();
    counter.close(); // Should not throw
    assertTrue(counter.isClosed(), "Counter should still be closed");
  }

  @Test
  public void testSampleAfterClose() {
    // Test that sampling after close throws exception
    NvtxCounter counter = NvtxCounter.create("Sample After Close Counter");
    counter.close();
    
    assertThrows(IllegalStateException.class, () -> {
      counter.sampleInt64(100L);
    }, "Should throw exception when sampling closed counter");
  }

  @Test
  public void testSampleFloat64AfterClose() {
    // Test that sampling after close throws exception
    NvtxCounter counter = NvtxCounter.create("Sample After Close Counter");
    counter.close();
    
    assertThrows(IllegalStateException.class, () -> {
      counter.sampleFloat64(98.6);
    }, "Should throw exception when sampling closed counter");
  }

  @Test
  public void testSampleNoValueAfterClose() {
    // Test that sampling after close throws exception
    NvtxCounter counter = NvtxCounter.create("Sample After Close Counter");
    counter.close();
    
    assertThrows(IllegalStateException.class, () -> {
      counter.sampleNoValue(NvtxCounterConstants.COUNTER_SAMPLE_UNCHANGED);
    }, "Should throw exception when sampling closed counter");
  }

  @Test
  public void testBatchSubmitAfterClose() {
    // Test that batch submission after close throws exception
    NvtxCounter counter = NvtxCounter.create("Batch After Close Counter");
    counter.close();
    
    byte[] data = new byte[]{1, 2, 3, 4, 5, 6, 7, 8};
    NvtxCounterBatch batch = new NvtxCounterBatch.Builder()
        .counters(data)
        .build();
    
    assertThrows(IllegalStateException.class, () -> {
      counter.submitBatch(batch);
    }, "Should throw exception when submitting batch to closed counter");
  }

  @Test
  public void testCounterAttrBuilderValidation() {
    // Test that builder validates required fields
    assertThrows(IllegalArgumentException.class, () -> {
      new NvtxCounterAttr.Builder(null);
    }, "Should throw exception for null name");
    
    assertThrows(IllegalArgumentException.class, () -> {
      new NvtxCounterAttr.Builder("");
    }, "Should throw exception for empty name");
    
    // Test that builder validates counter ID range
    assertThrows(IllegalArgumentException.class, () -> {
      new NvtxCounterAttr.Builder("Test")
          .counterId(100L) // Too small, not in valid range
          .build();
    }, "Should throw exception for invalid counter ID");
  }

  @Test
  public void testCounterBatchBuilderValidation() {
    // Test that batch builder validates required fields
    assertThrows(IllegalStateException.class, () -> {
      new NvtxCounterBatch.Builder().build();
    }, "Should throw exception when counters array is not set");
    
    assertThrows(IllegalArgumentException.class, () -> {
      new NvtxCounterBatch.Builder()
          .counters(null)
          .build();
    }, "Should throw exception for null counters array");
    
    assertThrows(IllegalArgumentException.class, () -> {
      new NvtxCounterBatch.Builder()
          .counters(new byte[0])
          .build();
    }, "Should throw exception for empty counters array");
  }

  @Test
  public void testCounterWithScope() {
    // Test counter with a custom scope ID
    long scopeId = 12345L;
    
    NvtxCounterAttr attr = new NvtxCounterAttr.Builder("Scoped Counter")
        .description("Counter with custom scope")
        .scopeId(scopeId)
        .build();
    
    try (NvtxCounter counter = NvtxCounter.create(attr)) {
      assertEquals(scopeId, attr.getScopeId(), "Scope ID should match");
      
      // Sample the counter
      counter.sampleInt64(999L);
      
      // If we get here without exceptions, the test passes
      assertTrue(true, "Should be able to use counter with custom scope");
    }
  }

  @Test
  public void testCounterAttrGetters() {
    // Test that all attribute getters work correctly
    String name = "Test Counter Name";
    String description = "Test Counter Description";
    long schemaId = NvtxCounterConstants.SCHEMA_ID_INT64;
    long scopeId = 789L;
    long counterId = NvtxCounterConstants.COUNTER_ID_STATIC_START + 500;
    
    NvtxCounterAttr attr = new NvtxCounterAttr.Builder(name)
        .description(description)
        .schemaId(schemaId)
        .scopeId(scopeId)
        .counterId(counterId)
        .build();
    
    assertEquals(name, attr.getName(), "Name should match");
    assertEquals(description, attr.getDescription(), "Description should match");
    assertEquals(schemaId, attr.getSchemaId(), "Schema ID should match");
    assertEquals(scopeId, attr.getScopeId(), "Scope ID should match");
    assertEquals(counterId, attr.getCounterId(), "Counter ID should match");
  }

  @Test
  public void testCounterBatchGetters() {
    // Test that all batch getters work correctly
    byte[] counterData = new byte[]{1, 2, 3, 4, 5, 6, 7, 8};
    long flags = NvtxCounterConstants.COUNTER_BATCH_FLAG_BEGINTIME_INTERVAL_PAIR;
    long[] timestamps = new long[]{1000L, 2000L};
    
    NvtxCounterBatch batch = new NvtxCounterBatch.Builder()
        .counters(counterData)
        .flags(flags)
        .timestamps(timestamps)
        .build();
    
    assertArrayEquals(counterData, batch.getCounters(), "Counter data should match");
    assertEquals(flags, batch.getFlags(), "Flags should match");
    assertArrayEquals(timestamps, batch.getTimestamps(), "Timestamps should match");
  }

  @Test
  public void testCounterWithinRange() {
    // Test using counter within an NVTX range
    try (NvtxRange range = new NvtxRange("Processing Phase", NvtxColor.BLUE);
         NvtxCounter counter = NvtxCounter.create("Operations", "Number of operations performed")) {
      
      for (int i = 0; i < 5; i++) {
        counter.sampleInt64(i * 10);
      }
      
      // If we get here without exceptions, the test passes
      assertTrue(true, "Should be able to use counter within NVTX range");
    }
  }
}
