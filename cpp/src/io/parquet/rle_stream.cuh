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
  int _output_pos;
  int level_run;
  int size;

  __device__ inline void decode(
    uint8_t const* const end, 
    int level_bits, 
    int lane) 
  {
    int output_pos = 0;
    int remain     = size;

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
          }
          cur_thread++;
          if (level_bits > 8 - bitpos && cur_thread < end) {
            level_val |= cur_thread[0] << 8;
            cur_thread++;
            if (level_bits > 16 - bitpos && cur_thread < end) { 
              level_val |= cur_thread[0] << 16; 
            }
          }
          level_val = (level_val >> bitpos) & ((1 << level_bits) - 1);
        }

        cur += batch_len8 * level_bits;
      }

      // store level_val
      if (lane < batch_len && (lane + output_pos) >= 0) { 
        auto idx = lane + _output_pos + output_pos + run_offset;
        output[rolling_index<max_output_values>(idx)] = level_val;
      }
      remain -= batch_len;
      output_pos += batch_len;
    }
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

  template<int max_output_values>
  __device__ __inline__ rle_batch<level_t, max_output_values> next_batch(
    level_t* const output, int output_count, int cur_values)
  {
    int batch_len = max(0, min(remaining, cur_values + output_count - (output_pos + size - remaining))); 
    int const run_offset = size - remaining;
    return rle_batch<level_t, max_output_values>{
      start, run_offset, output, output_pos, level_run, batch_len};
  }
};

// a stream of rle_runs
template <typename level_t, int decode_threads, int max_output_values>
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

  int total_values;
  int cur_values;

  level_t* output;

  rle_run<level_t>* runs;
  int output_pos;

  int fill_index;
  int decode_index;

  __device__ rle_stream(rle_run<level_t>* _runs) : runs(_runs) {
    for (int i = 0; i < num_rle_stream_decode_warps * 2; ++i) {
      runs[i].remaining = 0;
    }
  }

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

    output_pos           = 0;

    total_values = _total_values;
    cur_values   = 0;
    fill_index = 0;
    decode_index = -1;
  }

  __device__ inline void fill_run_batch()
  {
    
    while (((decode_index == -1 && fill_index < num_rle_stream_decode_warps) || 
            fill_index < decode_index) && 
            cur < end) {
      auto& run = runs[rolling_index<run_buffer_size>(fill_index)];

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
      fill_index++;
    }

    if (decode_index == -1) {
      // first time, set it to the beginning of the buffer (rolled)
      decode_index = run_buffer_size;
    }
  }

  __device__ inline int decode_next(int t, int count, int roll)
  {
    int const output_count = min(count < 0 ? max_output_values : count, total_values - cur_values);

    // special case. if level_bits == 0, just return all zeros. this should tremendously speed up
    // a very common case: columns with no nulls, especially if they are non-nested
    // TODO: this may not work with the logic of decode_next
    // we'd like to remove `roll`.
    if (level_bits == 0) {
      int written = 0;
      while (written < output_count) {
        int const batch_size = min(num_rle_stream_decode_threads, output_count - written);
        if (t < batch_size) { 
          output[rolling_index<max_output_values>(written + t + roll)] = 0; 
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

    __shared__ int values_processed;
    __shared__ int decode_index_shared;
    __shared__ int fill_index_shared;
    if (!t) {
      // carryover from the last call.
      values_processed = 0;
      decode_index_shared = decode_index;
      fill_index_shared = fill_index;
    }
    __syncthreads();

    do {
      // warp 0 reads ahead and generates batches of runs to be decoded by remaining warps.
      if (!warp_id) {
        // fill the next set of runs. fill_runs will generally be the bottleneck for any
        // kernel that uses an rle_stream.
        if (warp_lane == 0 && fill_index >= 0) { 
          fill_run_batch(); 
        }
      }
      // remaining warps decode the runs
      else if (decode_index_shared >= 0 && decode_index_shared >= fill_index_shared) {
        int run_index = decode_index_shared + warp_decode_id;
        auto& run  = runs[rolling_index<run_buffer_size>(run_index)];
        //if (warp_lane == 0) {
        //  printf("warp: %i run: %i remaining: %i cur_values %i output_count %i run.output_pos %i\n", 
        //  warp_id, run_index, run.remaining, cur_values, output_count, run.output_pos);
        //}

        if (run.remaining > 0 && (cur_values + output_count - run.output_pos) > 0) {
          //if (warp_lane == 0) {
          //printf("running run_index %i run.remaining %i\n",
          //       rolling_index<run_buffer_size>(run_index),
          //       run.remaining);
          //}
          auto remain_prio = run.remaining;
          auto batch = run.next_batch<max_output_values>(output, output_count, cur_values);
          batch.decode(end, level_bits, warp_lane);
          if (warp_lane == 0) {
            run.remaining -= batch.size;
            // TODO: can this be done in terms of _output_pos and batch.size and rolling_index of the
            // output buffer?, like the index we use to write into output[]
            //printf("am I the last run? %i output_count: %i remaining: %i, processed: %i run.output_pos: %i, batch._output_po: %i, cur_values: %i batch.size: %i\n", 
            //  rolling_index<run_buffer_size>(run_index),
            //  output_count,
            //  run.remaining,
            //  new_values_processed,
            //  run.output_pos,
            //  batch._output_pos,
            //  cur_values,
            //  batch.size);
            
            auto last_pos = (run.output_pos + (run.size - remain_prio) + batch.size) - cur_values;
            if (run.remaining > 0 || last_pos == output_count) {
              values_processed = last_pos;
              //printf("I am the last run!! %i processed: %i run.output_pos: %i, batch._output_po: %i, cur_values: %i batch.size: %i\n", 
              //  rolling_index<run_buffer_size>(run_index),
              //  values_processed,
              //  run.output_pos,
              //  batch._output_pos,
              //  cur_values,
              //  batch.size);
              if (run.remaining == 0) {
                //printf("also advancing decode index to %i", run_index + 1);
                decode_index_shared = run_index + 1;
              }
            } else if (run.remaining == 0 && warp_id == num_rle_stream_decode_warps) {
              //printf("advancing decode_index to %i", decode_index + num_rle_stream_decode_warps);
              decode_index_shared += num_rle_stream_decode_warps;
            }
          }
        }
      }
      __syncthreads();

     //if(warp_id == 0 && warp_lane == 0) {
     // printf("----\n");
     //}
     //for (int i = 0; i < num_rle_stream_decode_warps * 2; ++i) {
     //  if (warp_id == 0 && warp_lane == 0) {
     //    printf("runs[%i] roll is: %i remaining: %i output_pos: %i output_pos_end: %i\n", 
     //      i, 
     //      roll,
     //      runs[i].remaining, 
     //      runs[i].remaining == 0 ? -1 : rolling_index<256>(runs[i].output_pos),
     //      runs[i].remaining == 0 ? -1 : rolling_index<256>(runs[i].output_pos + runs[i].remaining));
     //  }
     //}

     // advance decode indices
     if (!t) {
       if (decode_index_shared == -1) {
        decode_index_shared = decode_index;
       }
       fill_index_shared   = fill_index;
     }

     __syncthreads();

     decode_index = decode_index_shared;

    } while (values_processed < output_count);

    cur_values += values_processed;

    // valid for every thread
    return values_processed;
  }

  __device__ inline int decode_next(int t) {
    return decode_next(t, -1, 0);
  }
};

}  // namespace cudf::io::parquet::detail
