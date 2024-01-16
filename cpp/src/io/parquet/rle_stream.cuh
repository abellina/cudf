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

#include "parquet_gpu.hpp"
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <inttypes.h>

namespace cudf::io::parquet::detail {

template <int num_threads>
constexpr int rle_stream_required_run_buffer_size()
{
  constexpr int num_rle_stream_decode_warps = (num_threads / cudf::detail::warp_size) - 1;
  return (num_rle_stream_decode_warps * 2);
}

inline __device__ int rolling_index_d(int index, int rolling_size)
{
  return index % rolling_size;
}

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
template <typename level_t>
struct rle_batch {
  uint8_t const* run_start;  // start of the run we are part of
  int run_offset;            // value offset of this batch from the start of the run
  level_t* output;
  int _output_pos;
  int level_run;
  int size;

  __device__ inline int decode(
    int run_index,
    int t,
    uint8_t const* const end, int level_bits, int lane, int warp_id, 
    int roll, int max_output_values, int do_print, int values_processed)
  {
    int output_pos = 0;
    int remain     = size;
    int processed = 0;

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

    // if this is a repeated run, compute the repeated value
    int level_val;
    if (!(level_run & 1)) {
      level_val = run_start[0];
      //if (lane==0) {
      //printf("warp %i rep. run_start[0] = %i\n", warp_id, run_start[0]);
      //}
      if (level_bits > 8) { 
        level_val |= run_start[1] << 8; 
        if (level_bits > 16) {
          level_val |= run_start[2] << 16; 
          if (level_bits > 24) {
            level_val |= run_start[3] << 24; 
          }
        }
      }
    }

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
          if (cur_thread < end) { 
            level_val = cur_thread[0];
            //if (lane == 0) { printf("warp %i lit1. level_val = %i\n", warp_id, level_val); }
          }
          cur_thread++;
          if (level_bits > 8 - bitpos && cur_thread < end) {
            level_val |= cur_thread[0] << 8;
            //if (lane == 0) { printf("warp %i lit2. level_val = %i\n", warp_id ,level_val); }
            cur_thread++;
            if (level_bits > 16 - bitpos && cur_thread < end) { 
              level_val |= cur_thread[0] << 16; 
            //if (lane == 0) { printf("warp %i lit3. level_val = %i\n", warp_id, level_val); }
            }
          }
          level_val = (level_val >> bitpos) & ((1 << level_bits) - 1);
          //if (lane == 0) { printf("warp %i lit4. level_val = %i\n", warp_id, level_val); }
        }

        cur += batch_len8 * level_bits;
      }

      // store level_val
      if (lane < batch_len && (lane + output_pos) >= 0) { 
        [[maybe_unused]] auto idx = lane + _output_pos + output_pos /*+ roll */ + run_offset;
        
        // TODO: abellina bring back if you want to print output
       //if (do_print == 2) {
       //  printf("run_index: %i run_start: %" PRIu64 " literal? %i level_bits: %i idx: %i output[idx]=%i remain: %i batch_len: %i RLE\n", 
       //  run_index,
       //  (uint64_t)run_start,
       //  level_run & 1,
       //  level_bits,
       //  idx, 
       //  level_val,
       //  remain,
       //  batch_len);
       //}
       //if (do_print == 2) {
       //  printf("lane=%i _output_pos=%i output_pos=%i roll=%i run_offset=%i new[%i]=%i\n", 
       //  lane,
       //  _output_pos,
       //  output_pos,
       //  roll,
       //  run_offset,
       //  idx, 
       //  level_val);
       //}
        output[rolling_index_d(idx, max_output_values)] = level_val;
      }
      remain -= batch_len;
      output_pos += batch_len;
      processed += batch_len;
    }
    return processed;
  }
};

// a single rle run. may be broken up into multiple rle_batches
template <typename level_t>
struct rle_run {
  int size;  // total size of the run
  int output_pos;
  uint8_t const* start;
  int level_run;  // level_run header value
  int remaining;

  __device__ __inline__ rle_batch<level_t> next_batch(
    level_t* const output, int output_count, int values_processed, int cur_values)
  {
    int batch_len = max(0, min(remaining, cur_values + output_count - (output_pos + size - remaining))); 
    // TODO: abellina this is slihtly better but we end up processing too much
    // this is because run.output_pos is the output buffer offset for this run
    // so it's pretty useful here. Need to handle this differently because now
    // a run can survive calls to decode_next.
    //int max_size = max(0, min(remaining, output_count - values_processed));
    int const run_offset = size - remaining;
    remaining -= batch_len;
    return rle_batch<level_t>{start, run_offset, output, output_pos, level_run, batch_len};
  }
};

// a stream of rle_runs
template <typename level_t, int decode_threads>
struct rle_stream {
  static constexpr int num_rle_stream_decode_threads = decode_threads;
  // the -1 here is for the look-ahead warp that fills in the list of runs to be decoded
  // in an overlapped manner. so if we had 16 total warps:
  // - warp 0 would be filling in batches of runs to be processed
  // - warps 1-15 would be decoding the previous batch of runs generated
  // == 3 if decode_threads is 128, default.
  static constexpr int num_rle_stream_decode_warps =
    (num_rle_stream_decode_threads / cudf::detail::warp_size) - 1;

  static constexpr int run_buffer_size = rle_stream_required_run_buffer_size<decode_threads>();

  int level_bits;
  uint8_t const* start;
  uint8_t const* cur;
  uint8_t const* end;

  int max_output_values;
  int total_values;
  int cur_values;

  level_t* output;

  rle_run<level_t>* runs;
  int run_index;
  int run_count;
  int output_pos;
  bool spill;

  int next_batch_run_start;
  int next_batch_run_count;
  int dict;
  int t;

  int max_runs_to_fill;
  int fill_index;
  int fill_warp_db_half;
  int decode_warps_db_half;

  __device__ rle_stream(rle_run<level_t>* _runs) : runs(_runs) {
    for (int i = 0; i <6; ++i) {
      runs[i].remaining = 0;
    }
  }

  __device__ void init(int _level_bits,
                       uint8_t const* _start,
                       uint8_t const* _end,
                       int _max_output_values,
                       level_t* _output,
                       int _total_values,
                       int _dict,
                       int _t)
  {
    level_bits = _level_bits;
    start      = _start;
    cur        = _start;
    end        = _end;

    max_output_values = _max_output_values; // 256
    output            = _output;

    run_index            = 0;
    run_count            = 0;
    output_pos           = 0;
    spill                = false;
    next_batch_run_start = 0;
    next_batch_run_count = 0;

    total_values = _total_values;
    cur_values   = 0;
    dict = _dict;
    t = _t;
    max_runs_to_fill = num_rle_stream_decode_warps;
    fill_index = 0;
    fill_warp_db_half = 0;
    decode_warps_db_half = 0;
  }

  __device__ void init(int _level_bits,
                       uint8_t const* _start,
                       uint8_t const* _end,
                       int _max_output_values,
                       level_t* _output,
                       int _total_values) {
    init(_level_bits, _start, _end, _max_output_values, _output, _total_values, -1, -1);
  }

  __device__ inline thrust::pair<int, int> get_run_batch()
  {
    return {next_batch_run_start, next_batch_run_count};
  }

  __device__ inline void fill_run_batch(int do_print)
  {
    // generate runs until we either run out of warps to decode them with
    [[maybe_unused]] uint8_t const* my_start = cur;
    //if (do_print == 1) {
    //  printf("at fill_run_batch with run_index: %i run_count: %i\n", run_index, run_count);
    //}

    while (fill_index < max_runs_to_fill && cur < end) {
      auto& run = runs[fill_index];

      // Encoding::RLE

      // bytes for the varint header
      uint8_t const* _cur = cur;
      int const level_run = get_vlq32(_cur, end);
      // run_bytes includes the header size
      int run_bytes       = _cur - cur;

      // literal run
      if (level_run & 1) {
        // multiples of 8
        run.size            = (level_run >> 1) * 8; 
        run_bytes += ((run.size * level_bits) + 7) >> 3;
      }
      // repeated value run
      else {
        run.size = (level_run >> 1);
        run_bytes += ((level_bits) + 7) >> 3;
      }
      run.output_pos = output_pos;
      run.start      = _cur;
      run.level_run  = level_run;
      run.remaining  = run.size;
      cur += run_bytes;
      
      output_pos += run.size;
      // TODO: abellina fill_run_batch print
      //if (do_print == 2) {
      //  printf("t: %i run_index: %i fill_index: %i is_literal: %i level_run: %i run.remaining: %i RLE\n", 
      //    t, run_index, fill_index, level_run & 1, level_run, run.remaining);
      //}
      run_index++;
      run_count++;
      fill_index++;
    }

    next_batch_run_count = run_count;
  }

  __device__ inline int decode_next(int t, int do_print, int count, int roll)
  {
    int const output_count = min(count < 0 ? max_output_values : count, total_values - cur_values);

    // special case. if level_bits == 0, just return all zeros. this should tremendously speed up
    // a very common case: columns with no nulls, especially if they are non-nested
    if (level_bits == 0) {
      int written = 0;
      while (written < output_count) {
        int const batch_size = min(num_rle_stream_decode_threads, output_count - written);
        if (t < batch_size) { 
          output[rolling_index_d(written + t + roll, max_output_values)] = 0; 
        }
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
        if (warp_lane == 0 && fill_index >= 0) { 
          fill_run_batch(do_print); 
        }
      }
      // remaining warps decode the runs
      else if (warp_decode_id < num_runs && next_batch_run_start >= 0) {
        // each warp handles 1 run, regardless of size.
        // TODO: having each warp handle exactly 32 values would be ideal. as an example, the
        // repetition levels for one of the list benchmarks decodes in ~3ms total, while the
        // definition levels take ~11ms - the difference is entirely due to long runs in the
        // definition levels.
        // TODO: run_start needs to stay at the lowest non-consumed run in this set of warps
        int run_index = run_start + warp_decode_id;
        auto& run  = runs[rolling_index<run_buffer_size>(run_index)];
        if (run.remaining > 0 && (cur_values + output_count - run.output_pos) > 0) {
          int remain_prio = run.remaining;
          auto batch = run.next_batch(output, output_count, values_processed, cur_values);
          // TODO: if remaining is > 0, we need to make sure we account for that next iteration..
          // does run.output_pos need to be changed?
          //if (!warp_lane && do_print == 2) {
          //  printf(
          //    "run_start: %i warp_id: %i num_runs: %i decoding batch at run_index: %i "
          //    "run.output_pos: %i "
          //    "this_remaining: %i cur_values: %i buf_run_output_pos: %i output_count: %i "
          //    "run.remaining: %i "
          //    "run.size: %i batch.offset: %i "
          //    "max_runs_to_fill: %i fill_index: %i \n",
          //    run_start,
          //    warp_id,
          //    num_runs,
          //    run_index,
          //    run.output_pos,
          //    remain_prio,
          //    cur_values,
          //    cur_values - run.output_pos,
          //    output_count,
          //    run.remaining,
          //    run.size,
          //    batch.run_offset,
          //    max_runs_to_fill,
          //    fill_index);
          //}
          [[maybe_unused]] int batch_processed = batch.decode(run_index,
                      t,
                      end,
                      level_bits,
                      warp_lane,
                      warp_decode_id,
                      roll,
                      max_output_values,
                      do_print,
                      values_processed);
          //if (!warp_lane && do_print == 2) { 
          //  printf("decode run_index: %i processed %i\n", run_index, batch_processed);
          //}
          if (warp_lane == 0) {
            atomicAdd(&values_processed, remain_prio - run.remaining);
          }
          //// last warp updates total values processed
          //if (warp_lane == 0 && warp_id == (max_runs_to_fill - fill_index)) {
          //  values_processed = run.output_pos + batch.size;
          //}
        }
      }
      __syncthreads();

      //for (int i = 0; i < num_rle_stream_decode_warps * 2; ++i) {
      //  if (warp_id == 0 && warp_lane == 0 && do_print == 2) {
      //    printf("runs[%i] remaining: %i output_pos: %i output_pos_end: %i\n", 
      //    i, 
      //    runs[i].remaining, 
      //    runs[i].remaining == 0 ? -1 : rolling_index<256>(runs[i].output_pos),
      //    runs[i].remaining == 0 ? -1 : rolling_index<256>(runs[i].output_pos + runs[i].remaining));
      //  }
      //}

      if (runs[0].remaining == 0 && 
          runs[1].remaining == 0 && 
          runs[2].remaining == 0) {
        fill_warp_db_half = 0;
        max_runs_to_fill = 3;
      } else if (runs[3].remaining == 0 && 
                 runs[4].remaining == 0 && 
                 runs[5].remaining == 0) {
        fill_warp_db_half = 1;
        max_runs_to_fill = 6;
      } else {
        fill_warp_db_half = -1;
        max_runs_to_fill = -1;
      }

      if (decode_warps_db_half == 0) {
        if (
          runs[0].remaining == 0 &&
          runs[1].remaining == 0 && 
          runs[2].remaining == 0) {
            decode_warps_db_half = 1;
        } // else stay in first half
      } else {
        if (
          runs[3].remaining == 0 &&
          runs[4].remaining == 0 && 
          runs[5].remaining == 0) {
            decode_warps_db_half = 0;
        } // else stay in second half
      }

      fill_index = -1;
      if (fill_warp_db_half == 0) {
        fill_index = 0;
      } else if (fill_warp_db_half == 1) {
        fill_index = 3;
      }

      next_batch_run_start = -1;
      if (decode_warps_db_half == 0) {
        next_batch_run_start = 0;
      } else if (decode_warps_db_half == 1) {
        next_batch_run_start = 3;
      }

      //if (warp_id == 0 && warp_lane == 0 && do_print == 1) {
      //  printf("after values_processed: %i fill/decode max_runs_to_fill should become: %i fill_index: %i next_batch_run_start: %i\n", 
      //    values_processed, max_runs_to_fill, fill_index, next_batch_run_start);
      //}

      // if we haven't run out of space, retrieve the next batch. otherwise leave it for the next
      // call.
      if (!t && values_processed < output_count) {
        thrust::tie(run_start, num_runs) = get_run_batch();
      }
      __syncthreads();

    } while (num_runs > 0 && values_processed < output_count);

    cur_values += values_processed;

    // valid for every thread
    return values_processed;
  }

  __device__ inline int decode_next(int t) {
    return decode_next(t, 0, -1, 0);
  }

  __device__ inline int decode_next(int t, int do_print) {
    return decode_next(t, do_print, -1, 0);
  }
};

}  // namespace cudf::io::parquet::detail
