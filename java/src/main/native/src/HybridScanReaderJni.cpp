/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "jni_compiled_expr.hpp"
#include "jni_utils.hpp"

#include <unordered_map>

#include <cudf/utilities/span.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
namespace {

// Wrapper to hold hybrid scan reader and associated resources
struct hybrid_scan_reader_wrapper {
  cudf::io::parquet_reader_options options;
  std::unique_ptr<cudf::io::parquet::experimental::hybrid_scan_reader> reader;
  
  hybrid_scan_reader_wrapper(
    cudf::host_span<uint8_t const> footer_bytes,
    cudf::io::parquet_reader_options opts)
    : options(std::move(opts)),
      reader(std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
        footer_bytes, options))
  {
  }
};

// Result of copy planning - holds device memory and spans into it
struct planned_copy_result {
  std::vector<rmm::device_buffer> device_buffers;  // Owns the device memory
  std::vector<cudf::device_span<uint8_t const>> spans;  // Views into device_buffers
};

// Check if byte ranges are contiguous and can be coalesced into a single copy
// Returns true if ranges are contiguous (no gaps between them)
bool are_ranges_contiguous(std::vector<byte_range_info> const& ranges) {
  if (ranges.size() <= 1) return true;
  
  for (size_t i = 1; i < ranges.size(); ++i) {
    // Check if previous range end == current range start
    if (ranges[i-1].offset() + ranges[i-1].size() != ranges[i].offset()) {
      return false;
    }
  }
  return true;
}

// Plan and execute H2D copy for byte ranges from a single host buffer
// If ranges are contiguous, does a single large copy; otherwise individual copies
planned_copy_result plan_and_copy_ranges(
    uint8_t const* host_ptr,
    std::vector<byte_range_info> const& ranges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
  planned_copy_result result;
  if (ranges.empty()) return result;
  
  if (are_ranges_contiguous(ranges)) {
    // Coalesce into single copy
    auto const first_offset = ranges.front().offset();
    auto const last_range = ranges.back();
    auto const total_size = (last_range.offset() + last_range.size()) - first_offset;
    
    // Single device buffer for all ranges
    rmm::device_buffer coalesced_buf(total_size, stream, mr);
    
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      coalesced_buf.data(),
      host_ptr + first_offset,
      total_size,
      cudaMemcpyHostToDevice,
      stream.value()));
    
    // Create spans pointing into the coalesced buffer
    result.spans.reserve(ranges.size());
    auto const base_ptr = static_cast<uint8_t const*>(coalesced_buf.data());
    for (auto const& range : ranges) {
      auto const offset_in_buffer = range.offset() - first_offset;
      result.spans.emplace_back(base_ptr + offset_in_buffer, range.size());
    }
    
    result.device_buffers.emplace_back(std::move(coalesced_buf));
  } else {
    // Individual copies (fallback)
    result.device_buffers.reserve(ranges.size());
    
    for (auto const& range : ranges) {
      rmm::device_buffer dev_buf(range.size(), stream, mr);
      
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        dev_buf.data(),
        host_ptr + range.offset(),
        range.size(),
        cudaMemcpyHostToDevice,
        stream.value()));
      
      result.device_buffers.emplace_back(std::move(dev_buf));
    }
    
    // Create spans after all buffers added (to avoid reallocation issues)
    result.spans.reserve(result.device_buffers.size());
    for (auto const& buf : result.device_buffers) {
      result.spans.emplace_back(
        static_cast<uint8_t const*>(buf.data()), buf.size());
    }
  }
  
  return result;
}

// Plan and execute H2D copy for individual host buffers (address/length pairs)
// No coalescing possible since buffers are separate
planned_copy_result plan_and_copy_buffers(
    std::vector<std::pair<void const*, size_t>> const& host_buffers,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
  planned_copy_result result;
  if (host_buffers.empty()) return result;
  
  result.device_buffers.reserve(host_buffers.size());
  
  for (auto const& [host_ptr, size] : host_buffers) {
    rmm::device_buffer dev_buf(size, stream, mr);
    
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      dev_buf.data(),
      host_ptr,
      size,
      cudaMemcpyHostToDevice,
      stream.value()));
    
    result.device_buffers.emplace_back(std::move(dev_buf));
  }
  
  // Create spans after all buffers added
  result.spans.reserve(result.device_buffers.size());
  for (auto const& buf : result.device_buffers) {
    result.spans.emplace_back(
      static_cast<uint8_t const*>(buf.data()), buf.size());
  }
  
  return result;
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_HybridScanReader_createFromFooter(JNIEnv* env,
                                                       jclass,
                                                       jlong footer_address,
                                                       jlong footer_length,
                                                       jlong filter_handle,
                                                       jobjectArray j_columns)
{
  JNI_NULL_CHECK(env, footer_address, "footer address is null", 0);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    // Create source info with empty source (we only use footer bytes directly)
    cudf::io::source_info source{};
    auto builder = cudf::io::parquet_reader_options::builder(source);
    
    // Add columns if specified
    cudf::jni::native_jstringArray column_names(env, j_columns);
    if (column_names.size() > 0) {
      builder = builder.column_names(column_names.as_cpp_vector());
    }
    
    // Build options
    auto opts = builder.build();
    
    // Set filter if provided
    if (filter_handle != 0) {
      auto const filter_expr =
        reinterpret_cast<cudf::jni::ast::compiled_expr const*>(filter_handle);
      opts.set_filter(filter_expr->get_top_expression());
    }
    
    // Create footer span
    auto const footer_ptr = reinterpret_cast<uint8_t const*>(footer_address);
    cudf::host_span<uint8_t const> footer_bytes{footer_ptr, static_cast<size_t>(footer_length)};
    
    // Create wrapper
    auto wrapper = std::make_unique<hybrid_scan_reader_wrapper>(footer_bytes, std::move(opts));
    
    return reinterpret_cast<jlong>(wrapper.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jintArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_getAllRowGroups(JNIEnv* env,
                                                      jclass,
                                                      jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    auto wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto row_groups = wrapper->reader->all_row_groups(wrapper->options);
    
    // Convert to Java array
    auto result = env->NewIntArray(row_groups.size());
    JNI_NULL_CHECK(env, result, "failed to allocate int array", nullptr);
    
    if (!row_groups.empty()) {
      env->SetIntArrayRegion(result, 0, row_groups.size(), row_groups.data());
    }
    
    return result;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jintArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_filterRowGroupsWithStats(JNIEnv* env,
                                                               jclass,
                                                               jlong handle,
                                                               jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    auto wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    auto const stream = cudf::get_default_stream();
    
    // Get input row groups from JNI array
    cudf::jni::native_jintArray row_groups(env, j_row_groups);
    auto const num_row_groups = static_cast<size_t>(row_groups.size());
    
    // Copy to pinned memory for faster access
    auto pinned_row_groups = cudf::detail::make_pinned_vector_async<cudf::size_type>(
      num_row_groups, stream);
    std::memcpy(pinned_row_groups.data(), row_groups.data(), 
                num_row_groups * sizeof(cudf::size_type));
    
    cudf::host_span<cudf::size_type const> row_group_span{
      pinned_row_groups.data(), num_row_groups};
    
    // Filter with stats
    auto filtered = wrapper->reader->filter_row_groups_with_stats(
      row_group_span,
      wrapper->options,
      stream);
    
    // Convert to Java array
    auto result = env->NewIntArray(filtered.size());
    JNI_NULL_CHECK(env, result, "failed to allocate int array", nullptr);
    
    if (!filtered.empty()) {
      env->SetIntArrayRegion(result, 0, filtered.size(), filtered.data());
    }
    
    row_groups.cancel();
    return result;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_getDictionaryPageByteRanges(JNIEnv* env,
                                                                  jclass,
                                                                  jlong handle,
                                                                  jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    auto wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    
    // Get input row groups
    cudf::jni::native_jintArray row_groups(env, j_row_groups);
    cudf::host_span<cudf::size_type const> row_group_span{
      reinterpret_cast<cudf::size_type const*>(row_groups.data()),
      static_cast<size_t>(row_groups.size())};
    
    // Get secondary filter byte ranges (bloom filters, dictionary pages)
    auto [bloom_ranges, dict_ranges] = wrapper->reader->secondary_filters_byte_ranges(
      row_group_span, wrapper->options);
    
    // Convert dictionary ranges to Java array (pairs of offset, length)
    auto result = env->NewLongArray(dict_ranges.size() * 2);
    JNI_NULL_CHECK(env, result, "failed to allocate long array", nullptr);
    
    std::vector<jlong> range_data;
    range_data.reserve(dict_ranges.size() * 2);
    for (auto const& range : dict_ranges) {
      range_data.push_back(static_cast<jlong>(range.offset()));
      range_data.push_back(static_cast<jlong>(range.size()));
    }
    
    if (!range_data.empty()) {
      env->SetLongArrayRegion(result, 0, range_data.size(), range_data.data());
    }
    
    row_groups.cancel();
    return result;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jintArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_filterRowGroupsWithDictionaries(JNIEnv* env,
                                                                      jclass,
                                                                      jlong handle,
                                                                      jlongArray j_buffer_addrs,
                                                                      jlongArray j_buffer_lens,
                                                                      jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_buffer_addrs, "buffer addresses is null", nullptr);
  JNI_NULL_CHECK(env, j_buffer_lens, "buffer lengths is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    auto wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    
    // Get input row groups
    cudf::jni::native_jintArray row_groups(env, j_row_groups);
    cudf::host_span<cudf::size_type const> row_group_span{
      reinterpret_cast<cudf::size_type const*>(row_groups.data()),
      static_cast<size_t>(row_groups.size())};
    
    // Get buffer addresses and lengths
    cudf::jni::native_jlongArray buffer_addrs(env, j_buffer_addrs);
    cudf::jni::native_jlongArray buffer_lens(env, j_buffer_lens);
    
    // Create device buffer views (note: the data is already on device)
    std::vector<cudf::device_span<uint8_t const>> dict_buffers;
    dict_buffers.reserve(buffer_addrs.size());
    for (int i = 0; i < buffer_addrs.size(); ++i) {
      // Create a device buffer that wraps the existing memory
      // Note: This is a view, not a copy
      dict_buffers.emplace_back(
        cudf::device_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer_addrs[i]), static_cast<size_t>(buffer_lens[i])));
    }
    
    // Filter with dictionaries
    auto filtered = wrapper->reader->filter_row_groups_with_dictionary_pages(
      cudf::host_span<cudf::device_span<uint8_t const>>(dict_buffers),
      row_group_span,
      wrapper->options,
      cudf::get_default_stream());
    
    // Convert to Java array
    auto result = env->NewIntArray(filtered.size());
    JNI_NULL_CHECK(env, result, "failed to allocate int array", nullptr);
    
    if (!filtered.empty()) {
      env->SetIntArrayRegion(result, 0, filtered.size(), filtered.data());
    }
    
    row_groups.cancel();
    buffer_addrs.cancel();
    buffer_lens.cancel();
    return result;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_HybridScanReader_getTotalRowsInRowGroups(JNIEnv* env,
                                                              jclass,
                                                              jlong handle,
                                                              jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", 0);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", 0);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    auto wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    
    // Get input row groups
    cudf::jni::native_jintArray row_groups(env, j_row_groups);
    cudf::host_span<cudf::size_type const> row_group_span{
      reinterpret_cast<cudf::size_type const*>(row_groups.data()),
      static_cast<size_t>(row_groups.size())};
    
    auto total_rows = wrapper->reader->total_rows_in_row_groups(row_group_span);
    
    row_groups.cancel();
    return static_cast<jlong>(total_rows);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_getFilterColumnChunkByteRanges(JNIEnv* env,
                                                                     jclass,
                                                                     jlong handle,
                                                                     jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    auto wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    
    // Get input row groups
    cudf::jni::native_jintArray row_groups(env, j_row_groups);
    cudf::host_span<cudf::size_type const> row_group_span{
      reinterpret_cast<cudf::size_type const*>(row_groups.data()),
      static_cast<size_t>(row_groups.size())};
    
    // Get filter column chunk byte ranges
    auto ranges = wrapper->reader->filter_column_chunks_byte_ranges(
      row_group_span, wrapper->options);
    
    // Convert to Java array (pairs of offset, length)
    auto result = env->NewLongArray(ranges.size() * 2);
    JNI_NULL_CHECK(env, result, "failed to allocate long array", nullptr);
    
    std::vector<jlong> range_data;
    range_data.reserve(ranges.size() * 2);
    for (auto const& range : ranges) {
      range_data.push_back(static_cast<jlong>(range.offset()));
      range_data.push_back(static_cast<jlong>(range.size()));
    }
    
    if (!range_data.empty()) {
      env->SetLongArrayRegion(result, 0, range_data.size(), range_data.data());
    }
    
    row_groups.cancel();
    return result;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_getPayloadColumnChunkByteRanges(JNIEnv* env,
                                                                      jclass,
                                                                      jlong handle,
                                                                      jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    auto wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    
    // Get input row groups
    cudf::jni::native_jintArray row_groups(env, j_row_groups);
    cudf::host_span<cudf::size_type const> row_group_span{
      reinterpret_cast<cudf::size_type const*>(row_groups.data()),
      static_cast<size_t>(row_groups.size())};
    
    // Get payload column chunk byte ranges
    auto ranges = wrapper->reader->payload_column_chunks_byte_ranges(
      row_group_span, wrapper->options);
    
    // Convert to Java array (pairs of offset, length)
    auto result = env->NewLongArray(ranges.size() * 2);
    JNI_NULL_CHECK(env, result, "failed to allocate long array", nullptr);
    
    std::vector<jlong> range_data;
    range_data.reserve(ranges.size() * 2);
    for (auto const& range : ranges) {
      range_data.push_back(static_cast<jlong>(range.offset()));
      range_data.push_back(static_cast<jlong>(range.size()));
    }
    
    if (!range_data.empty()) {
      env->SetLongArrayRegion(result, 0, range_data.size(), range_data.data());
    }
    
    row_groups.cancel();
    return result;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_getAllColumnChunkByteRanges(JNIEnv* env,
                                                                  jclass,
                                                                  jlong handle,
                                                                  jintArray j_row_groups)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    auto wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    
    // Get input row groups
    cudf::jni::native_jintArray row_groups(env, j_row_groups);
    cudf::host_span<cudf::size_type const> row_group_span{
      reinterpret_cast<cudf::size_type const*>(row_groups.data()),
      static_cast<size_t>(row_groups.size())};
    
    // Get ALL column chunk byte ranges (for single-stage materialization)
    auto ranges = wrapper->reader->all_column_chunks_byte_ranges(
      row_group_span, wrapper->options);
    
    // Convert to Java array (pairs of offset, length)
    auto result = env->NewLongArray(ranges.size() * 2);
    JNI_NULL_CHECK(env, result, "failed to allocate long array", nullptr);
    
    std::vector<jlong> range_data;
    range_data.reserve(ranges.size() * 2);
    for (auto const& range : ranges) {
      range_data.push_back(static_cast<jlong>(range.offset()));
      range_data.push_back(static_cast<jlong>(range.size()));
    }
    
    if (!range_data.empty()) {
      env->SetLongArrayRegion(result, 0, range_data.size(), range_data.data());
    }
    
    row_groups.cancel();
    return result;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_materializePayloadColumns(JNIEnv* env,
                                                                jclass,
                                                                jlong handle,
                                                                jintArray j_row_groups,
                                                                jlongArray j_buffer_addrs,
                                                                jlongArray j_buffer_lens)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_NULL_CHECK(env, j_buffer_addrs, "buffer addresses is null", nullptr);
  JNI_NULL_CHECK(env, j_buffer_lens, "buffer lengths is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    auto wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    
    // Get input row groups
    cudf::jni::native_jintArray row_groups(env, j_row_groups);
    cudf::host_span<cudf::size_type const> row_group_span{
      reinterpret_cast<cudf::size_type const*>(row_groups.data()),
      static_cast<size_t>(row_groups.size())};
    
    // Get buffer addresses and lengths
    cudf::jni::native_jlongArray buffer_addrs(env, j_buffer_addrs);
    cudf::jni::native_jlongArray buffer_lens(env, j_buffer_lens);
    
    // Create device buffers from the provided addresses
    std::vector<cudf::device_span<uint8_t const>> column_chunk_buffers;
    column_chunk_buffers.reserve(buffer_addrs.size());
    for (int i = 0; i < buffer_addrs.size(); ++i) {
      column_chunk_buffers.emplace_back(
        cudf::device_span<uint8_t const>(
          reinterpret_cast<uint8_t const*>(buffer_addrs[i]),
          static_cast<size_t>(buffer_lens[i])));
    }

    // Create all-true row mask (page pruning not applicable without filter)
    auto total_row_count = wrapper->reader->total_rows_in_row_groups(row_group_span);
    auto true_scalar = cudf::make_fixed_width_scalar(true, cudf::get_default_stream());
    auto all_rows_mask = cudf::make_column_from_scalar(
      *true_scalar,
      total_row_count,
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());

    // Materialize payload columns (no filter, so no page pruning benefit)
    auto result = wrapper->reader->materialize_payload_columns(
      row_group_span,
      column_chunk_buffers,
      *all_rows_mask,
      cudf::io::parquet::experimental::use_data_page_mask::NO,
      wrapper->options,
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());

    row_groups.cancel();
    buffer_addrs.cancel();
    buffer_lens.cancel();
    
    // Convert table to column handles
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_materializeFromHostBuffer(JNIEnv* env,
                                                                jclass,
                                                                jlong handle,
                                                                jlong buffer_address,
                                                                jlong buffer_length)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, buffer_address, "buffer address is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    auto wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    
    // Get all row groups
    auto all_row_groups = wrapper->reader->all_row_groups(wrapper->options);
    cudf::host_span<cudf::size_type const> row_group_span{all_row_groups};

    // Create the host buffer pointer
    auto const host_ptr = reinterpret_cast<uint8_t const*>(buffer_address);
    auto const stream = cudf::get_default_stream();
    auto const mr = cudf::get_current_device_resource_ref();
    
    // Plan and copy filter column data (with coalescing if contiguous)
    planned_copy_result filter_copy_result;
    if (wrapper->options.get_filter().has_value()) {
      auto filter_ranges = wrapper->reader->filter_column_chunks_byte_ranges(
        row_group_span, wrapper->options);
      filter_copy_result = plan_and_copy_ranges(host_ptr, filter_ranges, stream, mr);
    }

    // Get byte ranges for payload columns
    auto byte_ranges = wrapper->reader->payload_column_chunks_byte_ranges(
      row_group_span, wrapper->options);

    // Plan and copy payload column data (with coalescing if contiguous)
    auto payload_copy_result = plan_and_copy_ranges(host_ptr, byte_ranges, stream, mr);

    // Create all-true row mask - will be updated by materialize_filter_columns
    // Note: Pre-filtering via build_row_mask_with_page_index_stats is disabled due to
    // https://github.com/rapidsai/cudf/issues/20833 (fails if page index isn't found)
    // Instead, we rely on page pruning during materialize_payload_columns
    auto total_row_count = wrapper->reader->total_rows_in_row_groups(row_group_span);
    auto true_scalar = cudf::make_fixed_width_scalar(true, cudf::get_default_stream());
    auto all_rows_mask = 
      cudf::make_column_from_scalar(
        *true_scalar,
        total_row_count,
        cudf::get_default_stream(),
        cudf::get_current_device_resource_ref());

    auto row_mask_view = all_rows_mask->mutable_view();

    cudf::io::table_with_metadata filter_result;
    if (wrapper->options.get_filter().has_value()) {
      filter_result = wrapper->reader->materialize_filter_columns(
        row_group_span,
        filter_copy_result.spans,
        row_mask_view,
        cudf::io::parquet::experimental::use_data_page_mask::NO,
        wrapper->options,
        stream,
        cudf::get_current_device_resource_ref());
    }

    // Materialize payload columns with page pruning enabled
    // The row_mask was updated by materialize_filter_columns above,
    // so cuDF can skip pages where all rows are filtered out
    auto result = wrapper->reader->materialize_payload_columns(
      row_group_span,
      payload_copy_result.spans,
      all_rows_mask->view(),
      cudf::io::parquet::experimental::use_data_page_mask::YES,
      wrapper->options,
      stream,
      cudf::get_current_device_resource_ref());

    // Merge filter and payload columns in the originally requested order
    auto requested_columns = wrapper->options.get_column_names();
    
    // If no filter was set, just return the payload columns directly
    if (!wrapper->options.get_filter().has_value()) {
      return cudf::jni::convert_table_for_return(env, result.tbl);
    }
    
    if (!requested_columns.has_value() || requested_columns->empty()) {
      // No specific columns requested - just return payload columns
      // (filter columns were only for filtering, not output)
      return cudf::jni::convert_table_for_return(env, result.tbl);
    }
    
    // Build name → index maps for filter and payload columns
    std::unordered_map<std::string, size_t> filter_col_idx;
    if (filter_result.tbl) {
      for (size_t i = 0; i < filter_result.metadata.schema_info.size(); ++i) {
        filter_col_idx[filter_result.metadata.schema_info[i].name] = i;
      }
    }
    
    std::unordered_map<std::string, size_t> payload_col_idx;
    for (size_t i = 0; i < result.metadata.schema_info.size(); ++i) {
      payload_col_idx[result.metadata.schema_info[i].name] = i;
    }
    
    // Release columns from both tables
    std::vector<std::unique_ptr<cudf::column>> filter_cols;
    if (filter_result.tbl) {
      filter_cols = filter_result.tbl->release();
    }
    auto payload_cols = result.tbl->release();
    
    // Merge in original requested order
    std::vector<std::unique_ptr<cudf::column>> merged_columns;
    merged_columns.reserve(requested_columns->size());
    
    for (const auto& col_name : *requested_columns) {
      if (auto it = filter_col_idx.find(col_name); it != filter_col_idx.end()) {
        merged_columns.push_back(std::move(filter_cols[it->second]));
      } else if (auto it = payload_col_idx.find(col_name); it != payload_col_idx.end()) {
        merged_columns.push_back(std::move(payload_cols[it->second]));
      }
    }
    
    auto merged_table = std::make_unique<cudf::table>(std::move(merged_columns));
    
    return cudf::jni::convert_table_for_return(env, std::move(merged_table));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_materializeFromHostBuffers(JNIEnv* env,
                                                                 jclass,
                                                                 jlong handle,
                                                                 jintArray j_row_groups,
                                                                 jlongArray j_filter_buffer_addrs,
                                                                 jlongArray j_filter_buffer_lens,
                                                                 jlongArray j_payload_buffer_addrs,
                                                                 jlongArray j_payload_buffer_lens)
{
  CUDF_FUNC_RANGE();
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  // Filter buffers can be null if no filter
  JNI_NULL_CHECK(env, j_payload_buffer_addrs, "payload buffer addresses is null", nullptr);
  JNI_NULL_CHECK(env, j_payload_buffer_lens, "payload buffer lengths is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    auto wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    
    // Get input row groups
    cudf::jni::native_jintArray row_groups(env, j_row_groups);
    cudf::host_span<cudf::size_type const> row_group_span{
      reinterpret_cast<cudf::size_type const*>(row_groups.data()),
      static_cast<size_t>(row_groups.size())};
    
    auto const stream = cudf::get_default_stream();
    auto const mr = cudf::get_current_device_resource_ref();
    
    // Create all-true row mask - will be updated by materialize_filter_columns
    // Page pruning is enabled for payload columns via use_data_page_mask::YES
    auto total_row_count = wrapper->reader->total_rows_in_row_groups(row_group_span);
    auto true_scalar = cudf::make_fixed_width_scalar(true, stream);
    auto all_rows_mask = cudf::make_column_from_scalar(
      *true_scalar,
      total_row_count,
      stream,
      mr);
    
    auto row_mask_view = all_rows_mask->mutable_view();
    
    // Handle filter columns if present
    cudf::io::table_with_metadata filter_result;
    planned_copy_result filter_copy_result;
    
    if (j_filter_buffer_addrs != nullptr && j_filter_buffer_lens != nullptr) {
      cudf::jni::native_jlongArray filter_addrs(env, j_filter_buffer_addrs);
      cudf::jni::native_jlongArray filter_lens(env, j_filter_buffer_lens);
      
      if (filter_addrs.size() > 0) {
        // Build list of host buffer pointers
        std::vector<std::pair<void const*, size_t>> filter_host_buffers;
        filter_host_buffers.reserve(filter_addrs.size());
        for (int i = 0; i < filter_addrs.size(); ++i) {
          filter_host_buffers.emplace_back(
            reinterpret_cast<void const*>(filter_addrs[i]),
            static_cast<size_t>(filter_lens[i]));
        }
        
        // Copy to device (no coalescing possible - separate buffers)
        filter_copy_result = plan_and_copy_buffers(filter_host_buffers, stream, mr);
        
        // Materialize filter columns
        filter_result = wrapper->reader->materialize_filter_columns(
          row_group_span,
          filter_copy_result.spans,
          row_mask_view,
          cudf::io::parquet::experimental::use_data_page_mask::NO,
          wrapper->options,
          stream,
          cudf::get_current_device_resource_ref());
        
        filter_addrs.cancel();
        filter_lens.cancel();
      }
    }
    
    // Get payload buffer addresses and lengths (these are HOST addresses)
    cudf::jni::native_jlongArray payload_addrs(env, j_payload_buffer_addrs);
    cudf::jni::native_jlongArray payload_lens(env, j_payload_buffer_lens);
    
    // Handle payload columns only if we have payload buffers
    cudf::io::table_with_metadata payload_result;
    planned_copy_result payload_copy_result;
    
    if (payload_addrs.size() > 0) {
      // Build list of host buffer pointers
      std::vector<std::pair<void const*, size_t>> payload_host_buffers;
      payload_host_buffers.reserve(payload_addrs.size());
      for (int i = 0; i < payload_addrs.size(); ++i) {
        payload_host_buffers.emplace_back(
          reinterpret_cast<void const*>(payload_addrs[i]),
          static_cast<size_t>(payload_lens[i]));
      }
      
      // Copy to device (no coalescing possible - separate buffers)
      payload_copy_result = plan_and_copy_buffers(payload_host_buffers, stream, mr);
      
      // Materialize payload columns with page pruning enabled
      // The row_mask was updated by materialize_filter_columns above,
      // so cuDF can skip pages where all rows are filtered out
      payload_result = wrapper->reader->materialize_payload_columns(
        row_group_span,
        payload_copy_result.spans,
        all_rows_mask->view(),
        cudf::io::parquet::experimental::use_data_page_mask::YES,
        wrapper->options,
        stream,
        cudf::get_current_device_resource_ref());
    }

    row_groups.cancel();
    payload_addrs.cancel();
    payload_lens.cancel();
    
    // Get the requested columns - we MUST return columns in this order
    auto requested_columns = wrapper->options.get_column_names();
    
    // If no filter columns, just return payload columns (or empty if no payload either)
    if (!filter_result.tbl) {
      if (!payload_result.tbl) {
        // No filter and no payload columns - return empty table
        // This shouldn't normally happen, but handle it gracefully
        auto empty_table = std::make_unique<cudf::table>();
        return cudf::jni::convert_table_for_return(env, std::move(empty_table));
      }
      // Payload only - still need to reorder to match requested columns
      if (!requested_columns.has_value() || requested_columns->empty()) {
        return cudf::jni::convert_table_for_return(env, payload_result.tbl);
      }
      // Build name → index map and reorder
      std::unordered_map<std::string, size_t> payload_col_idx;
      for (size_t i = 0; i < payload_result.metadata.schema_info.size(); ++i) {
        payload_col_idx[payload_result.metadata.schema_info[i].name] = i;
      }
      auto payload_cols = payload_result.tbl->release();
      std::vector<std::unique_ptr<cudf::column>> reordered;
      reordered.reserve(requested_columns->size());
      for (const auto& col_name : *requested_columns) {
        if (auto it = payload_col_idx.find(col_name); it != payload_col_idx.end()) {
          reordered.push_back(std::move(payload_cols[it->second]));
        }
      }
      auto reordered_table = std::make_unique<cudf::table>(std::move(reordered));
      return cudf::jni::convert_table_for_return(env, std::move(reordered_table));
    }
    
    // If no payload columns, return filter columns in the correct order
    if (!payload_result.tbl) {
      if (!requested_columns.has_value() || requested_columns->empty()) {
        return cudf::jni::convert_table_for_return(env, filter_result.tbl);
      }
      // Build name → index map and reorder filter columns
      std::unordered_map<std::string, size_t> filter_col_idx;
      for (size_t i = 0; i < filter_result.metadata.schema_info.size(); ++i) {
        filter_col_idx[filter_result.metadata.schema_info[i].name] = i;
      }
      auto filter_cols = filter_result.tbl->release();
      std::vector<std::unique_ptr<cudf::column>> reordered;
      reordered.reserve(requested_columns->size());
      for (const auto& col_name : *requested_columns) {
        if (auto it = filter_col_idx.find(col_name); it != filter_col_idx.end()) {
          reordered.push_back(std::move(filter_cols[it->second]));
        }
      }
      auto reordered_table = std::make_unique<cudf::table>(std::move(reordered));
      return cudf::jni::convert_table_for_return(env, std::move(reordered_table));
    }
    
    // Both filter and payload columns - merge in the originally requested order
    if (!requested_columns.has_value() || requested_columns->empty()) {
      // No specific columns requested - just return payload columns
      return cudf::jni::convert_table_for_return(env, payload_result.tbl);
    }
    
    // Build name → index maps for filter and payload columns
    std::unordered_map<std::string, size_t> filter_col_idx;
    for (size_t i = 0; i < filter_result.metadata.schema_info.size(); ++i) {
      filter_col_idx[filter_result.metadata.schema_info[i].name] = i;
    }
    
    std::unordered_map<std::string, size_t> payload_col_idx;
    for (size_t i = 0; i < payload_result.metadata.schema_info.size(); ++i) {
      payload_col_idx[payload_result.metadata.schema_info[i].name] = i;
    }
    
    // Release columns from both tables
    auto filter_cols = filter_result.tbl->release();
    auto payload_cols = payload_result.tbl->release();
    
    // Merge in original requested order
    std::vector<std::unique_ptr<cudf::column>> merged_columns;
    merged_columns.reserve(requested_columns->size());
    
    for (const auto& col_name : *requested_columns) {
      if (auto it = filter_col_idx.find(col_name); it != filter_col_idx.end()) {
        merged_columns.push_back(std::move(filter_cols[it->second]));
      } else if (auto it = payload_col_idx.find(col_name); it != payload_col_idx.end()) {
        merged_columns.push_back(std::move(payload_cols[it->second]));
      }
    }
    
    auto merged_table = std::make_unique<cudf::table>(std::move(merged_columns));
    
    return cudf::jni::convert_table_for_return(env, std::move(merged_table));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_HybridScanReader_materializeAllColumnsFromHostBuffers(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle,
                                                                          jintArray j_row_groups,
                                                                          jlongArray j_buffer_addrs,
                                                                          jlongArray j_buffer_lens)
{
  CUDF_FUNC_RANGE();
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);
  JNI_NULL_CHECK(env, j_row_groups, "row groups is null", nullptr);
  JNI_NULL_CHECK(env, j_buffer_addrs, "buffer addresses is null", nullptr);
  JNI_NULL_CHECK(env, j_buffer_lens, "buffer lengths is null", nullptr);
  
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    
    auto wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    
    // Get input row groups
    cudf::jni::native_jintArray row_groups(env, j_row_groups);
    cudf::host_span<cudf::size_type const> row_group_span{
      reinterpret_cast<cudf::size_type const*>(row_groups.data()),
      static_cast<size_t>(row_groups.size())};
    
    auto const stream = cudf::get_default_stream();
    auto const mr = cudf::get_current_device_resource_ref();
    
    // Get buffer addresses and lengths (these are HOST addresses)
    cudf::jni::native_jlongArray buffer_addrs(env, j_buffer_addrs);
    cudf::jni::native_jlongArray buffer_lens(env, j_buffer_lens);
    
    // Build list of host buffer pointers
    std::vector<std::pair<void const*, size_t>> host_buffers;
    host_buffers.reserve(buffer_addrs.size());
    for (int i = 0; i < buffer_addrs.size(); ++i) {
      host_buffers.emplace_back(
        reinterpret_cast<void const*>(buffer_addrs[i]),
        static_cast<size_t>(buffer_lens[i]));
    }
    
    // Copy to device
    auto copy_result = plan_and_copy_buffers(host_buffers, stream, mr);
    
    // Materialize all columns in a single step
    // This applies the AST filter internally after reading all data
    auto result = wrapper->reader->materialize_all_columns(
      row_group_span,
      copy_result.spans,
      wrapper->options,
      stream,
      cudf::get_current_device_resource_ref());
    
    row_groups.cancel();
    buffer_addrs.cancel();
    buffer_lens.cancel();
    
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_HybridScanReader_destroy(JNIEnv* env,
                                              jclass,
                                              jlong handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto wrapper = reinterpret_cast<hybrid_scan_reader_wrapper*>(handle);
    delete wrapper;
  }
  JNI_CATCH(env, );
}

}  // extern "C"

