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
  int run_offset;            // offset of this batch from the start of the run
  level_t* output;           // global output buffer
  int run_output_pos;        // the index at which this run starts, TODO (abellina): same as run_start - output?
  int level_run;             // level_run holds varint header for this batch
  int size;                  // count of values we will write to output

  __device__ inline int last_pos() {
    return run_output_pos + run_offset + size;
  }

  __device__ inline void decode(uint8_t const* const end, int level_bits, int lane)
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
      if (lane < batch_len && (lane + batch_output_pos) >= 0) { 
        auto idx = lane + run_output_pos + run_offset + batch_output_pos; // TODO abellina: why run_output_pos AND run_offset too
        output[rolling_index<max_output_values>(idx)] = level_val;
      }
      remain -= batch_len;
      batch_output_pos += batch_len;
    }
  }
};

// a single rle run. may be broken up into multiple rle_batches
template <typename level_t>
struct rle_run {
  int size;         // total size of the run
  int output_pos;   // absolute position of this run w.r.t output
  uint8_t const* start;
  int level_run;    // level_run header value
  int remaining;    // number of output items remaining to be decoded
  
  #ifdef ABDEBUG2
  bool did_process;
  int last_batch;
  int cur_values;
  int run_offset_;
  #endif

  template<int max_output_values>
  __device__ __inline__ rle_batch<level_t, max_output_values> next_batch(
    level_t* const output, int max_count)
  {
    int const run_offset = size - remaining;          
    int batch_len = 
    max(0, 
      min(remaining, 
        // total
        max_count - 
        // position + processed by prior batches
        (output_pos + run_offset))); 
    return rle_batch<level_t, max_output_values>{
      start, 
      run_offset, 
      output, 
      output_pos, 
      level_run, 
      batch_len};
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

  #ifdef ABDEBUG2
  __shared__ rle_run<level_t> prior_runs[256][6];
  int prior_fill_indices[256];
  int prior_decode_indices[256];
  int prior_values_processed[256];
  int num_iter;
  #endif

  int fill_index;
  int decode_index;

  __device__ rle_stream(rle_run<level_t>* _runs) : runs(_runs) {
    for (int i = 0; i < num_rle_stream_decode_warps * 2; ++i) {
      runs[i].remaining = 0;
      #ifdef ABDEBUG2
      runs[i].did_process = false;
      #endif
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
    #ifdef ABDEBUG2
    num_iter = 0;
    #endif
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
      #ifdef ABDEBUG2
      run.did_process  = false;
      run.cur_values = 0;
      #endif
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
    int const output_count = min(count, total_values - cur_values);

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

    __shared__ int values_processed_shared;
    __shared__ int decode_index_shared;
    __shared__ int fill_index_shared;
    if (!t) {
      values_processed_shared = 0;
      decode_index_shared = decode_index;
      fill_index_shared = fill_index;
    }

    __syncthreads();

    fill_index = fill_index_shared;
    int local_values_processed = 0;

    //int stuck_at_beginning = 0;
    //int stuck_not_processing = 0;

    do {
      //if (local_values_processed == prior_local_values_processed) {
      //  stuck_not_processing++;
      //}
      // warp 0 reads ahead and generates batches of runs to be decoded by remaining warps.
      if (!warp_id) {
        // fill the next set of runs. fill_runs will generally be the bottleneck for any
        // kernel that uses an rle_stream.
        if (!warp_lane) { 
          fill_run_batch(); 
          //if (decode_index_shared == -1) {
          //  stuck_at_beginning++;
          //}
        }
      }
      // remaining warps decode the runs
      // decode_index = -1 is the initial condition, as we want the first iteration to skip decode,
      // since we are filling.
      // fill_index is "behind" decode_index, that way we are always decoding upto fill_index,
      // and we are filling up to decode_index.
      else if (decode_index >= 0 && decode_index >= fill_index) {
        int const run_index = decode_index + warp_decode_id;
        auto& run  = runs[rolling_index<run_buffer_size>(run_index)];
        int remaining = run.remaining;
        int const max_count = cur_values + output_count;
        if (remaining > 0 && 
          // the maximum amount we would write includes this run
          // this is calculated in absolute position
          (max_count > run.output_pos)) {
          auto batch = run.next_batch<max_output_values>(output, max_count);
          batch.decode(end, level_bits, warp_lane);
          if (!warp_lane) {
            #ifdef ABDEBUG2
            run.run_offset_ = run.size - run.remaining;          
            run.cur_values = cur_values;
            run.last_batch = batch.size;
            if (batch.size > 0) {
              run.did_process = true;
            }
            #endif
            auto last_pos = batch.last_pos() - cur_values; 
            remaining -= batch.size;
            // this is the last batch we will process this iteration if:
            // - either this run still has remaining
            // - or it is consumed fully and its last index corresponds to output_count
            if (remaining > 0 || last_pos == output_count) {
              values_processed_shared = last_pos;
              // only if we consumed fully do we want to move on, in the decode side
              if (remaining == 0) {
                decode_index_shared = run_index + 1;
              }
            } else if (remaining == 0 && warp_id == num_rle_stream_decode_warps) {
              // we skip over all num_rle_stream_decode_warp indices since all of them
              // will have been consumed.
              decode_index_shared += num_rle_stream_decode_warps;
            }
            run.remaining = remaining;
          }
        }
      }
     //if (!t) {
     //  fill_index_shared   = fill_index;
     //}
      __syncthreads();

      #ifdef ABDEBUG2 
      bool first_time = false;
      #endif
      // TODO: abellina move this inside fill_run_batch?
      if (!t) {
        if (decode_index_shared == -1) { 
          #ifdef ABDEBUG2
          first_time = true;
          #endif
          decode_index_shared = decode_index;
        }
        fill_index_shared = fill_index;
      }
      __syncthreads();
      local_values_processed  = values_processed_shared;
      decode_index = decode_index_shared;
      fill_index = fill_index_shared;

#ifdef ABDEBUG2
      if (!t) {
        bool any_processed = false;
        for (int i = 0; i < num_rle_stream_decode_warps * 2; ++i) {
          int num_iter_roll = rolling_index<256>(num_iter);
          any_processed = any_processed || runs[i].did_process;
          runs[i].did_process = false;
          prior_runs[num_iter_roll][i].remaining = runs[i].remaining;
          prior_runs[num_iter_roll][i].cur_values = runs[i].cur_values;
          prior_runs[num_iter_roll][i].output_pos = runs[i].output_pos;
          prior_fill_indices[num_iter_roll] = fill_index;
          prior_decode_indices[num_iter_roll] = decode_index;
          prior_values_processed[num_iter_roll] = local_values_processed;
        }
        
        if (!first_time && !any_processed && local_values_processed < output_count) {
          
          for (int it = 0; it < num_iter; ++it) {
            auto rit = rolling_index<256>(it);
            for (int i = 0; i < num_rle_stream_decode_warps * 2; ++i) {
              if (!t) {
                printf("tg: %i prior[%i] runs[%i] remaining: %i cur_values: %i output_pos: %i run_offset: %i fill_index: %i decode_index: %i values_processed: %i\n",
                      blockIdx.x,
                      rit,
                      i,
                      prior_runs[rit][i].remaining,
                      prior_runs[rit][i].cur_values,
                      prior_runs[rit][i].output_pos,
                      prior_runs[rit][i].run_offset_,
                      prior_fill_indices[rit],
                      prior_decode_indices[rit],
                      prior_values_processed[rit]);
              }
            }
          }
          // current
          for (int i = 0; i < num_rle_stream_decode_warps * 2; ++i) {
            if (!t) {
              printf("rit: %i tg: %i current runs[%i] roll is: %i remaining: %i last_batch: %i fill_count: %i decode_index: %i output_count: %i cur_values: %i output_pos: %i run_offset: %i\n",
                    rolling_index<256>(num_iter),
                    blockIdx.x,
                    i,
                    roll,
                    runs[i].remaining,
                    runs[i].last_batch,
                    fill_index,
                    decode_index,
                    output_count,
                    runs[i].cur_values,
                    runs[i].output_pos,
                    runs[i].run_offset_);
            }
            runs[i].last_batch = -456789;
            runs[i].cur_values     = 0; 
          }
        }
        num_iter++;
      }
#endif


#ifdef ABDEBUG
     if(!t) {
      printf("block: %i warp: %i decode_index: %i fill_index: %i output_count: %i total_values: %i cur_values: %i processed: %i\n", 
      blockIdx.x,
      warp_id, decode_index, fill_index, output_count, total_values, cur_values, local_values_processed);
     }

     for (int i = 0; i < num_rle_stream_decode_warps * 2; ++i) {
       if (!t) {
         printf("runs[%i] roll is: %i remaining: %i\n", 
           i, 
           roll,
           runs[i].remaining);
       }
     }

     if(!t) {
     printf("----\n");
     }
     
#endif
     __syncthreads();


    } while (local_values_processed < output_count);

    cur_values += local_values_processed;

    // valid for every thread
    return local_values_processed;
  }

  __device__ inline int decode_next(int t) {
    return decode_next(t, max_output_values, 0);
  }
};

}  // namespace cudf::io::parquet::detail
