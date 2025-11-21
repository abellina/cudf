/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * Attributes for registering an NVTX counter or counter group.
 * 
 * This class encapsulates the metadata needed to register a counter with the NVTX profiler.
 * Use the Builder pattern to construct instances with the desired configuration.
 * 
 * Example:
 * <pre>
 * {@code
 * NvtxCounterAttr attr = new NvtxCounterAttr.Builder("CPU Temperature")
 *     .description("Current CPU temperature in Celsius")
 *     .schemaId(NvtxCounterConstants.SCHEMA_ID_INT64)
 *     .scopeId(myScopeId)
 *     .build();
 * }
 * </pre>
 */
public class NvtxCounterAttr {
  private final String name;
  private final String description;
  private final long schemaId;
  private final long scopeId;
  private final long counterId;

  private NvtxCounterAttr(Builder builder) {
    this.name = builder.name;
    this.description = builder.description;
    this.schemaId = builder.schemaId;
    this.scopeId = builder.scopeId;
    this.counterId = builder.counterId;
  }

  /**
   * @return Name of the counter
   */
  public String getName() {
    return name;
  }

  /**
   * @return Optional detailed description of the counter
   */
  public String getDescription() {
    return description;
  }

  /**
   * @return Schema ID referring to the data layout of the counter group
   */
  public long getSchemaId() {
    return schemaId;
  }

  /**
   * @return Identifier of the counter's scope
   */
  public long getScopeId() {
    return scopeId;
  }

  /**
   * @return Static counter ID (or COUNTER_ID_NONE for dynamic allocation)
   */
  public long getCounterId() {
    return counterId;
  }

  /**
   * Builder for creating NvtxCounterAttr instances.
   */
  public static class Builder {
    private final String name;
    private String description = null;
    private long schemaId = 0; // Will use default schema
    private long scopeId = NvtxCounterConstants.SCOPE_NONE;
    private long counterId = NvtxCounterConstants.COUNTER_ID_NONE;

    /**
     * Create a new builder with the specified counter name.
     * 
     * @param name Name of the counter (required)
     */
    public Builder(String name) {
      if (name == null || name.isEmpty()) {
        throw new IllegalArgumentException("Counter name cannot be null or empty");
      }
      this.name = name;
    }

    /**
     * Set an optional detailed description for the counter.
     * 
     * @param description Description of the counter
     * @return This builder instance
     */
    public Builder description(String description) {
      this.description = description;
      return this;
    }

    /**
     * Set the schema ID for the counter's data layout.
     * 
     * @param schemaId Schema ID (use predefined constants or registered schema)
     * @return This builder instance
     */
    public Builder schemaId(long schemaId) {
      this.schemaId = schemaId;
      return this;
    }

    /**
     * Set the scope ID for the counter.
     * 
     * @param scopeId Scope identifier
     * @return This builder instance
     */
    public Builder scopeId(long scopeId) {
      this.scopeId = scopeId;
      return this;
    }

    /**
     * Set a static counter ID (must be >= COUNTER_ID_STATIC_START and unique within domain).
     * 
     * @param counterId Static counter ID
     * @return This builder instance
     */
    public Builder counterId(long counterId) {
      if (counterId != NvtxCounterConstants.COUNTER_ID_NONE &&
          counterId < NvtxCounterConstants.COUNTER_ID_STATIC_START) {
        throw new IllegalArgumentException(
            "Counter ID must be >= COUNTER_ID_STATIC_START or COUNTER_ID_NONE");
      }
      this.counterId = counterId;
      return this;
    }

    /**
     * Build the NvtxCounterAttr instance.
     * 
     * @return A new NvtxCounterAttr instance
     */
    public NvtxCounterAttr build() {
      return new NvtxCounterAttr(this);
    }
  }
}

