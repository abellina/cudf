/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
__device__ inline void decode(
    level_t* const output, 
    uint8_t const* const run_start, 
    int const run_offset,
    int const run_output_pos,
    int const level_run,
    int const size,
    uint8_t const* const end, 
    int level_bits, 
    int lane)
  {
    int batch_output_pos = 0;
    int remain     = size;

    // for bitpacked/literal runs, total size is always a multiple of 8. so we need to take care if
    // we are not starting/ending exactly on a run boundary
    uint8_t const* cur;
    if (level_run & 1) {
      int const effective_offset = cudf::util::round_down_safe(run_offset, 8);
      int const lead_values      = (run_offset - effective_offset);
      batch_output_pos -= lead_values;
      remain += lead_values;
      cur = run_start + ((effective_offset >> 3) * level_bits);
    }

    // if this is a repeated run, compute the repeated value
    int level_val;
    if (!(level_run & 1)) {
      level_val = run_start[0];
      // TODO (abellina): describe why you need this change
      if constexpr (sizeof(level_t) > 1) {
        if (level_bits > 8) {
          level_val |= run_start[1] << 8;
          if constexpr (sizeof(level_t) > 2) {
            if (level_bits > 16) {
              level_val |= run_start[2] << 16;
              if constexpr (sizeof(level_t) > 3) {
                if (level_bits > 24) {
                  level_val |= run_start[3] << 24;
                }
              }
            }
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
          if (cur_thread < end) { level_val = cur_thread[0]; }
          cur_thread++;
          if constexpr (sizeof(level_t) > 1) {
            if (level_bits > 8 - bitpos && cur_thread < end) {
              level_val |= cur_thread[0] << 8;
              cur_thread++;
              if constexpr (sizeof(level_t) > 2) {
                if (level_bits > 16 - bitpos && cur_thread < end) { 
                  level_val |= cur_thread[0] << 16;
                  cur_thread++;
                  if constexpr (sizeof(level_t) > 3) {
                    if (level_bits > 24 - bitpos && cur_thread < end) { 
                      level_val |= cur_thread[0] << 24;
                    }
                  }
                }
              }
            }
          }
          level_val = (level_val >> bitpos) & ((1 << level_bits) - 1);
        }

        cur += batch_len8 * level_bits;
      }

      // store level_val
      if (lane < batch_len && (lane + batch_output_pos) >= 0) { 
        auto idx = lane + run_output_pos + run_offset + batch_output_pos; // TODO abellina: why run_output_pos AND run_offset too
        output[rolling_index<max_output_values>(idx)] = level_val;
      }
      remain -= batch_len;
      batch_output_pos += batch_len;
    }
  }

// a single rle run. may be broken up into multiple rle_batches
template <typename level_t>
struct rle_run {
  int output_pos;   // absolute position of this run w.r.t output
  uint8_t const* start;
  int level_run;    // level_run header value
  int batch_remaining;
  int run_offset;

 //template<int max_output_values>
 //__device__ __inline__ rle_batch<level_t, max_output_values> next_batch(int max_count)
 //{
 //  int const run_offset = size - remaining;          
 //  int batch_len = 
 //  //max(0, 
 //    min(remaining, 
 //      // total
 //      max_count - 
 //      // position + processed by prior batches
 //      (output_pos + run_offset));//); 
 //  return rle_batch<level_t, max_output_values>{
 //    start, 
 //    run_offset, 
 //    output_pos, 
 //    level_run, 
 //    batch_len};
 //}
};

// a stream of rle_runs
template <typename level_t, int decode_threads, int max_output_values>
struct rle_stream {
  static constexpr int num_rle_stream_decode_threads = decode_threads;
  // the -1 here is for the look-ahead warp that fills in the list of runs to be decoded
  // in an overlapped manner. so if we had 16 total warps:
  // - warp 0 would be filling in batches of runs to be processed
  // - warps 1-15 would be decoding the previous batch of runs generated
  static constexpr int num_rle_stream_decode_warps =
    (num_rle_stream_decode_threads / cudf::detail::warp_size) - 1;

  static constexpr int run_buffer_size = rle_stream_required_run_buffer_size<decode_threads>();

  int level_bits;
  uint8_t const* cur;
  uint8_t const* end;

  int total_values;
  int cur_values;

  level_t* output;

  rle_run<level_t>* runs;

  int output_pos;

  int fill_index;
  int decode_index;

  int run_remaining;
  int run_size;

  __device__ rle_stream(rle_run<level_t>* _runs) : runs(_runs) {}

  __device__ void init(int _level_bits,
                       uint8_t const* _start,
                       uint8_t const* _end,
                       level_t* _output,
                       int _total_values)
  {
    level_bits = _level_bits;
    cur        = _start;
    end        = _end;

    output            = _output;

    output_pos           = 0;

    total_values = _total_values;
    cur_values   = 0;
    fill_index = 0;
    decode_index = -1;

    run_remaining = 0;
  }

  __device__ inline void fill_run_batch()
  {
    while (((decode_index == -1 && fill_index < num_rle_stream_decode_warps) || 
            fill_index < decode_index) && 
            (cur < end || run_remaining != 0)) {
      auto& run = runs[rolling_index<run_buffer_size>(fill_index)];

      // Encoding::RLE

      // bytes for the varint header
      int run_output_pos = 0;
      uint8_t const* run_cur = nullptr;
      int run_offset = 0;
      int run_level_run = 0;

      if (run_remaining == 0) {
        uint8_t const* _cur = cur;
        int const level_run = get_vlq32(_cur, end);
        run_cur = _cur;
        // run_bytes includes the header size
        int run_bytes       = _cur - cur;

        // literal run
        if (level_run & 1) {
          // multiples of 8
          run_size = (level_run >> 1) * 8; 
          run_bytes += ((run_size * level_bits) + 7) >> 3;
        }
        // repeated value run
        else {
          run_size = (level_run >> 1);
          run_bytes += ((level_bits) + 7) >> 3;
        }

        cur += run_bytes;
        run_output_pos = output_pos;
        output_pos += run_size;
        run_level_run = level_run;
        run_remaining = run_size;
        #ifdef ABDEBUG
        printf("run starting at %i with size %i\n", 
          rolling_index<run_buffer_size>(fill_index), 
          run_size);
        #endif
      } else {
        auto& prior_run = runs[rolling_index<run_buffer_size>(fill_index - 1)];
        run_output_pos = prior_run.output_pos;
        run_cur        = prior_run.start;
        run_level_run  = prior_run.level_run;
        run_offset     = run_size - run_remaining;
      }
      #ifdef ABDEBUG
      printf("fill_index: %i run size: %i remaining %i\n", 
      fill_index, run_size, run_remaining);
      
      #endif

      int this_batch = min(32, run_remaining);

      // don't change per batch
      run.output_pos = run_output_pos;
      run.start      = run_cur;
      run.level_run  = run_level_run;
      run.run_offset = run_offset;
      run.batch_remaining = this_batch;
      
      //run_output_pos += this_batch;
      run_remaining -= this_batch;
      fill_index++;
    }

  }

  __device__ inline int decode_next(int t, int count)
  {
    int const output_count = min(count, total_values - cur_values);

    // otherwise, full decode.
    int const warp_id        = t / cudf::detail::warp_size;
    int const warp_decode_id = warp_id - 1;
    int const warp_lane      = t % cudf::detail::warp_size;

    __shared__ int values_processed_shared;
    __shared__ int decode_index_shared;
    __shared__ int fill_index_shared;
    if (!t) {
      #ifdef ABDEBUG
      printf("-----start---------\n");
      #endif
      values_processed_shared = 0;
      decode_index_shared = decode_index;
      fill_index_shared = fill_index;
    }

    __syncthreads();

    fill_index = fill_index_shared;

    do {
      __syncthreads();
      // warp 0 reads ahead and generates batches of runs to be decoded by remaining warps.
      if (!warp_id) {
        // fill the next set of runs. fill_runs will generally be the bottleneck for any
        // kernel that uses an rle_stream.
        if (!warp_lane) { 
          fill_run_batch(); 
          if (decode_index == -1) {
            // first time, set it to the beginning of the buffer (rolled)
            decode_index = run_buffer_size;
            decode_index_shared = decode_index;
            for (int i = fill_index; i < run_buffer_size; ++i) {
              runs[i].batch_remaining = 0; // initialize rest
            }
          }
          fill_index_shared = fill_index;
        }
      }
      // remaining warps decode the runs
      // decode_index = -1 is the initial condition, as we want the first iteration to skip decode,
      // since we are filling.
      // fill_index is "behind" decode_index, that way we are always decoding upto fill_index,
      // and we are filling up to decode_index.
      else if (decode_index >= fill_index) {
        int const run_index = decode_index + warp_decode_id;
        auto& run  = runs[rolling_index<run_buffer_size>(run_index)];
        int const last_run_pos = run.output_pos + run.run_offset - cur_values;
        int const batch_limit = output_count - last_run_pos;
        if (batch_limit > 0) {
          int batch_remaining = run.batch_remaining;
          int const batch_len = min(batch_remaining, batch_limit);
          //auto batch = run.next_batch<max_output_values>(max_count);
          decode<level_t, max_output_values>(
            output,
            run.start,
            run.run_offset,
            run.output_pos,
            run.level_run,
            batch_len,
            end,
            level_bits,
            warp_lane);
          if (!warp_lane) {
            auto last_pos = last_run_pos + batch_len; 
            batch_remaining -= batch_len;
            #ifdef ABDEBUG
            printf("run[%i] batch_remaining: %i last_pos: %i output_count: %i\n",
              rolling_index<run_buffer_size>(run_index),
              batch_remaining,
              last_pos,
              output_count);
            #endif
            // this is the last batch we will process this iteration if:
            // - either this run still has remaining
            // - or it is consumed fully and its last index corresponds to output_count
            if (batch_remaining > 0 || last_pos == output_count) {
              values_processed_shared = last_pos;
            } 

            if (batch_remaining == 0 && (last_pos == output_count || warp_id == num_rle_stream_decode_warps)) {
              decode_index_shared = run_index + 1;
            }
            run.run_offset += batch_len;
            run.batch_remaining = batch_remaining;
          }
        }
      }
      __syncthreads();
      
      decode_index = decode_index_shared;
      fill_index = fill_index_shared;

      #ifdef ABDEBUG
        if (!t) {
        printf("have more?: %i processed: %i output_count: %i fill_index: %i/%i decode_index: %i/%i\n", 
          cur < end,
          values_processed_shared, output_count, 
          rolling_index<run_buffer_size>(fill_index), fill_index,
          rolling_index<run_buffer_size>(decode_index), decode_index);
        for (int i = 0; i < run_buffer_size; ++i) {
          printf("page %i runs[%i] remaining: %i\n",
            blockIdx.x,
            i,
            runs[i].batch_remaining);
        }
        printf("----end---------\n");
        }
      #endif
      
    } while (values_processed_shared < output_count);

    cur_values += values_processed_shared;

    // valid for every thread
    return values_processed_shared;
  }

  __device__ inline int decode_next(int t) {
    int const output_count = min(max_output_values, total_values - cur_values);
    // special case. if level_bits == 0, just return all zeros. this should tremendously speed up
    // a very common case: columns with no nulls, especially if they are non-nested
    // TODO: this may not work with the logic of decode_next
    // we'd like to remove `roll`.
    if (level_bits == 0) {
      int written = 0;
      while (written < output_count) {
        int const batch_size = min(num_rle_stream_decode_threads, output_count - written);
        if (t < batch_size) { 
          output[rolling_index<max_output_values>(written + t)] = 0; 
        }
        written += batch_size;
      }
      cur_values += output_count;
      return output_count;
    }

    return decode_next(t, max_output_values);
  }
};

}  // namespace cudf::io::parquet::detail
