/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <inttypes.h>
#include "parquet_gpu.hpp"
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

namespace cudf::io::parquet::detail {

template<int num_threads>
constexpr int rle_stream_required_run_buffer_size()
{
  int num_rle_stream_decode_warps = (num_threads / cudf::detail::warp_size) - 1;
  return (num_rle_stream_decode_warps * 2);
}


// TODO: consider if these should be template parameters to rle_stream
constexpr int num_rle_stream_decode_threads = 512;
// the -1 here is for the look-ahead warp that fills in the list of runs to be decoded
// in an overlapped manner. so if we had 16 total warps:
// - warp 0 would be filling in batches of runs to be processed
// - warps 1-15 would be decoding the previous batch of runs generated
constexpr int num_rle_stream_decode_warps =
  (num_rle_stream_decode_threads / cudf::detail::warp_size) - 1;

constexpr int run_buffer_size = (num_rle_stream_decode_warps * 2);

/**
 * @brief Read a 32-bit varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The 32-bit value read
 */
inline __device__ uint32_t get_vlq32(uint8_t const*& cur, uint8_t const* end)
{
  uint32_t v = *cur++;
  if (v >= 0x80 && cur < end) {
    v = (v & 0x7f) | ((*cur++) << 7);
    if (v >= (0x80 << 7) && cur < end) {
      v = (v & ((0x7f << 7) | 0x7f)) | ((*cur++) << 14);
      if (v >= (0x80 << 14) && cur < end) {
        v = (v & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | ((*cur++) << 21);
        if (v >= (0x80 << 21) && cur < end) {
          v = (v & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) | ((*cur++) << 28);
        }
      }
    }
  }
  return v;
}

// an individual batch. processed by a warp.
// batches should be in shared memory.
template <typename level_t, int max_output_values>
struct rle_batch {
  uint8_t const* run_start;  // start of the run we are part of
  int run_offset;            // value offset of this batch from the start of the run
  level_t* output;
  int level_run;
  int size;

  __device__ inline void decode(uint8_t const* const end, int level_bits, int lane, int warp_id, int roll)
  {
    int output_pos = 0;
    int remain     = size;
    #ifdef ABDEBUG
    printf("decode lane %i warp_id %i roll %i remain %i\n", lane, warp_id, roll, remain);
    #endif

    // for bitpacked/literal runs, total size is always a multiple of 8. so we need to take care if
    // we are not starting/ending exactly on a run boundary
    uint8_t const* cur;
    if (level_run & 1) {
      int const effective_offset = cudf::util::round_down_safe(run_offset, 8);
      int const lead_values      = (run_offset - effective_offset);
      output_pos -= lead_values;
      remain += lead_values;
      cur = run_start + ((effective_offset >> 3) * level_bits);
    }

    #ifdef ABDEBUG
    printf("done1 level_run %i\n", level_run);
    #endif

    // if this is a repeated run, compute the repeated value
    int level_val;
    if (!(level_run & 1)) {
      level_val = run_start[0];
      #ifdef ABDEBUG
      printf("level bits?? %i level_val %i\n", level_bits, level_val);
      #endif
      if (level_bits > 8) { level_val |= run_start[1] << 8; }
    }
    #ifdef ABDEBUG
    printf("done2\n");
    #endif

    // process
    while (remain > 0) {
      int const batch_len = min(32, remain);

      // if this is a literal run. each thread computes its own level_val
      if (level_run & 1) {
        int const batch_len8 = (batch_len + 7) >> 3;
        if (lane < batch_len) {
          int bitpos                = lane * level_bits;
          uint8_t const* cur_thread = cur + (bitpos >> 3);
          bitpos &= 7;
          level_val = 0;
          if (cur_thread < end) { level_val = cur_thread[0]; }
          cur_thread++;
          if (level_bits > 8 - bitpos && cur_thread < end) {
            level_val |= cur_thread[0] << 8;
            cur_thread++;
            if (level_bits > 16 - bitpos && cur_thread < end) { level_val |= cur_thread[0] << 16; }
          }
          level_val = (level_val >> bitpos) & ((1 << level_bits) - 1);
        }

        cur += batch_len8 * level_bits;
      }

      // store level_val
      if (lane < batch_len && (lane + output_pos) >= 0) { 
        int ix = rolling_index<max_output_values>(lane + output_pos + roll);
        #ifdef ABDEBUG
        printf("run_buffer_size: %i rolling index %i, lane %i, output_pos %i roll %i level_val %i\n", 
          (int)run_buffer_size, (int)ix, (int)lane, (int)output_pos, (int)roll, (int)level_val);
        #endif
        output[rolling_index<max_output_values>(lane + output_pos + roll)] = level_val;
      }
      remain -= batch_len;
      output_pos += batch_len;
    }
    #ifdef ABDEBUG
    printf("leaving decode lane %i\n", lane);
    #endif
  }
};

// a single rle run. may be broken up into multiple rle_batches
template <typename level_t, int max_output_values>
struct rle_run {
  int size;  // total size of the run
  int output_pos;
  uint8_t const* start;
  int level_run;  // level_run header value
  int remaining;

  __device__ __inline__ rle_batch<level_t, max_output_values> next_batch(level_t* const output, int max_size)
  {
    int const batch_len  = min(max_size, remaining);
    int const run_offset = size - remaining;
    remaining -= batch_len;
    return rle_batch<level_t, max_output_values>{
      start, run_offset, output, level_run, batch_len};
  }
};

// a stream of rle_runs
template <typename level_t, int decode_threads, int max_output_values>
struct rle_stream {
  // TODO: consider if these should be template parameters to rle_stream
  static constexpr int num_rle_stream_decode_threads = decode_threads;
  // the -1 here is for the look-ahead warp that fills in the list of runs to be decoded
  // in an overlapped manner. so if we had 16 total warps:
  // - warp 0 would be filling in batches of runs to be processed
  // - warps 1-15 would be decoding the previous batch of runs generated
  static constexpr int num_rle_stream_decode_warps =
    (num_rle_stream_decode_threads / cudf::detail::warp_size) - 1;

  static constexpr int run_buffer_size = rle_stream_required_run_buffer_size<decode_threads>();

  int level_bits;
  uint8_t const* start;
  uint8_t const* cur;
  uint8_t const* end;

  int total_values;
  int cur_values;

  level_t* output;

  rle_run<level_t, max_output_values>* runs;
  int run_index;
  int run_count;
  int output_pos;
  bool spill;

  int next_batch_run_start;
  int next_batch_run_count;

  __device__ rle_stream(rle_run<level_t, max_output_values>* _runs) : runs(_runs) {}

  __device__ void init(int _level_bits,
                       uint8_t const* _start,
                       uint8_t const* _end,
                       level_t* _output,
                       int _total_values)
  {
    level_bits = _level_bits;
    start      = _start;
    cur        = _start;
    end        = _end;
    output            = _output;

    run_index            = 0;
    run_count            = 0;
    output_pos           = 0;
    spill                = false;
    next_batch_run_start = 0;
    next_batch_run_count = 0;

    total_values = _total_values;
    cur_values   = 0;
  }

  __device__ inline thrust::pair<int, int> get_run_batch()
  {
    return {next_batch_run_start, next_batch_run_count};
  }

  // fill in up to num_rle_stream_decode_warps runs or until we reach the max_count limit.
  // this function is the critical hotspot.  please be very careful altering it.
  __device__ void fill_run_batch(int max_count)
  {
    // if we spilled over, we've already got a run at the beginning
    next_batch_run_start = spill ? run_index - 1 : run_index;
    spill                = false;

    // generate runs until we either run out of warps to decode them with, or
    // we cross the output limit.
    while (run_count < num_rle_stream_decode_warps && output_pos < max_count && cur < end) {
      auto& run = runs[rolling_index<run_buffer_size>(run_index)];

      // Encoding::RLE

      // bytes for the varint header
      uint8_t const* _cur = cur;
      int const level_run = get_vlq32(_cur, end);
      int run_bytes       = _cur - cur;
      #ifdef ABDEBUG
      printf("run_index %i level_run %i end-cur %i run_bytes %i\n", run_index, level_run, (int)(end-cur), run_bytes);
      #endif

      // literal run
      if (level_run & 1) {
        #ifdef ABDEBUG
        printf("literal run\n");
        #endif
        int const run_size  = (level_run >> 1) * 8;
        run.size            = run_size;
        int const run_size8 = (run_size + 7) >> 3;
        run_bytes += run_size8 * level_bits;
      }
      // repeated value run
      else {
        #ifdef ABDEBUG
        printf("repeated run\n");
        #endif
        run.size = (level_run >> 1);
        run_bytes++;
        // can this ever be > 16?  it effectively encodes nesting depth so that would require
        // a nesting depth > 64k.
        if (level_bits > 8) { run_bytes++; }
      }
      run.output_pos = output_pos;
      run.start      = _cur;
      run.level_run  = level_run;
      run.remaining  = run.size;
      cur += run_bytes;
      #ifdef ABDEBUG
      printf("run_bytes finished at %i\n", run_bytes);
      #endif

      output_pos += run.size;
      run_count++;
      run_index++;
    }

    // the above loop computes a batch of runs to be processed. mark down
    // the number of runs because the code after this point resets run_count
    // for the next batch. each batch is returned via get_next_batch().

    #ifdef ABDEBUG
    printf("run_count is %i", run_count);
    #endif
    next_batch_run_count = run_count;

    // -------------------------------------
    // prepare for the next run:

    // if we've reached the value output limit on the last run
    if (output_pos >= max_count) {
      // first, see if we've spilled over
      auto const& src       = runs[rolling_index<run_buffer_size>(run_index - 1)];
      int const spill_count = output_pos - max_count;

      // a spill has occurred in the current run. spill the extra values over into the beginning of
      // the next run.
      if (spill_count > 0) {
        auto& spill_run      = runs[rolling_index<run_buffer_size>(run_index)];
        spill_run            = src;
        spill_run.output_pos = 0;
        spill_run.remaining  = spill_count;

        run_count = 1;
        run_index++;
        output_pos = spill_run.remaining;
        spill      = true;
      }
      // no actual spill needed. just reset the output pos
      else {
        output_pos = 0;
        run_count  = 0;
      }
    }
    // didn't cross the limit, so reset the run count
    else {
      run_count = 0;
    }
  }

  __device__ int decode_next(int t, int count, int roll)
  {
    int const output_count = 
      count < 0 ? 
        min(max_output_values, (total_values - cur_values)) : 
        count;

    #ifdef ABDEBUG
    printf("decode_next i %i count %i roll %i\n", t, count, roll);
    #endif

    // special case. if level_bits == 0, just return all zeros. this should tremendously speed up
    // a very common case: columns with no nulls, especially if they are non-nested
    if (level_bits == 0) {
      int written = 0;
      while (written < output_count) {
        int const batch_size = min(num_rle_stream_decode_threads, output_count - written);
        if (t < batch_size) { output[rolling_index<max_output_values>(written + t + roll)] = 0; }
        written += batch_size;
      }
      cur_values += output_count;
      return output_count;
    }

    // otherwise, full decode.
    int const warp_id        = t / cudf::detail::warp_size;
    int const warp_decode_id = warp_id - 1;
    int const warp_lane      = t % cudf::detail::warp_size;

    __shared__ int run_start;
    __shared__ int num_runs;
    __shared__ int values_processed;
    if (!t) {
      // carryover from the last call.
      thrust::tie(run_start, num_runs) = get_run_batch();
      values_processed                 = 0;
    }
    __syncthreads();

    do {
      // warp 0 reads ahead and generates batches of runs to be decoded by remaining warps.
      if (!warp_id) {
        // fill the next set of runs. fill_runs will generally be the bottleneck for any
        // kernel that uses an rle_stream.
        if (warp_lane == 0) { fill_run_batch(output_count); }
      }
      // remaining warps decode the runs
      else if (warp_decode_id < num_runs) {
        // each warp handles 1 run, regardless of size.
        // TODO: having each warp handle exactly 32 values would be ideal. as an example, the
        // repetition levels for one of the list benchmarks decodes in ~3ms total, while the
        // definition levels take ~11ms - the difference is entirely due to long runs in the
        // definition levels.
        auto& run  = runs[rolling_index<run_buffer_size>(run_start + warp_decode_id)];
        #ifdef ABDEBUG
        printf("trying run_start %i  run.start %" PRIu64 "\n", (int)run_start, (uint64_t)run.start);
        #endif
        auto batch = run.next_batch(output + run.output_pos,
                                    min(run.remaining, (output_count - run.output_pos)));
        batch.decode(end, level_bits, warp_lane, warp_decode_id, roll);
        #ifdef ABDEBUG
        printf("done with decode warp_lane %i\n", warp_lane);
        #endif
      // last warp updates total values processed
        if (warp_lane == 0 && warp_decode_id == num_runs - 1) {
          values_processed = run.output_pos + batch.size;
          #ifdef ABDEBUG
          printf("values_processed=%i\n",values_processed);
          #endif
        }
      }
      __syncthreads();

      // if we haven't run out of space, retrieve the next batch. otherwise leave it for the next
      // call.
      if (!t && values_processed < output_count) {
        thrust::tie(run_start, num_runs) = get_run_batch();
      }
      __syncthreads();
      #ifdef ABDEBUG
      printf("values_processed %i, output_count %i\n", values_processed, output_count);
      #endif
    } while (num_runs > 0 && values_processed < output_count);

    cur_values += values_processed;

    #ifdef ABDEBUG
    printf("returning values_processed %i\n", values_processed);
    #endif

    // valid for every thread
    return values_processed;
  }

  __device__ inline int decode_next(int t) {
    return decode_next(t, -1, 0);
  }

};

}  // namespace cudf::io::parquet::gpu