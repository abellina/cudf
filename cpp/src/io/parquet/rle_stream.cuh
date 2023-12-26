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
  int offset;
  int level_run;
  int size;

  template<typename other_run_t>
  __device__ inline void decode(
    uint8_t const* const end, int level_bits, int lane, 
    int warp_id, int roll, int print_it,
    other_run_t other_run)
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
      if (level_bits > 8) { level_val |= run_start[1] << 8; }
      #ifdef ABDEBUG
      if (print_it == 0 && lane == 0) {
        printf("is repeated run, level_val is %i\n", level_val);
      }
      #endif
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
        #ifdef ABDEBUG
        if (print_it == 0 && lane == 0) {
          printf("is literal run, level_val is %i\n", level_val);
        }
        #endif
      }
      #ifdef ABDEBUG3
      if (other_run != nullptr && lane== 0) {
        for (int i = 0; i < 6; i++){ 
          printf("before_store other run rix %i level_run %i\n", 
            i, 
            (int)(other_run[i].level_run));
        }
      }
      #endif

      // store level_val
      if (lane < batch_len && (lane + output_pos) >= 0) { 
        int ix = rolling_index<max_output_values>(offset + lane + output_pos + roll);
        #ifdef ABDEBUG3
        printf("dict? %i addr %" PRIu64 " offset %i lane %i output_pos %i roll %i output[%i]=%i remain %i original %i\n", 
        print_it, 
        output + ix,
        offset,
        lane,
        output_pos,
        roll,
        ix, 
        level_val, 
        remain, 
        output[ix]);
        output[rolling_index<max_output_values>(offset + lane + output_pos + roll)] = level_val;
        #endif
      }
      #ifdef ABDEBUG3
      if (other_run != nullptr && lane == 0) {
        for (int i = 0; i < 6; i++){ 
          printf("after_store other run rix %i level_run %i\n", 
            i, 
            (int)(other_run[i].level_run));
        }
      }
      #endif
      remain -= batch_len;
      output_pos += batch_len;
    }
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

  __device__ __inline__ rle_batch<level_t, max_output_values> next_batch(
    level_t* const output, int offset, int max_size, int print_it)
  {
    int const batch_len  = min(max_size, remaining);
    int const run_offset = size - remaining;
    remaining -= batch_len;
    //  // TODO: batch_len is weird. 58, 116, 174, 232, 98, 192, 348 and then 80... ??
    #ifdef ABDEBUG2
    printf("dict? %i next_batch run_offset %i remaining %i batch_len %i max_size %i output %" PRIu64 " offset %i\n", 
      print_it, run_offset, remaining, batch_len, max_size, output, offset);
    #endif
    return rle_batch<level_t, max_output_values>{
      start, run_offset, output, offset, level_run, batch_len};
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
  __device__ void fill_run_batch(int max_count, int print_it)
  {
    #ifdef ABDEBUG
    for (int i = 0; i < 6; i++){ 
      printf("dict? %i rix %i level_run %i\n", print_it, i, runs[i].level_run);
    }
    #endif
    // if we spilled over, we've already got a run at the beginning
    next_batch_run_start = spill ? run_index - 1 : run_index;
    spill                = false;

    // generate runs until we either run out of warps to decode them with, or
    // we cross the output limit.
    int ri = rolling_index<run_buffer_size>(run_index);
    auto& run = runs[rolling_index<run_buffer_size>(run_index)];
    //printf("before while run_index %i rolling_index %i level_run %i dict? %i run_count %i num_rle_stream_decode_warps %i "
    //       "output_pos %i max_count %i cur < end %i\n", 
    //       run_index,
    //  ri,
    //  run.level_run,
    //  print_it,
    //  run_count,
    //  num_rle_stream_decode_warps,
    //  output_pos,
    //  max_count,
    //  cur < end ? 1 : 0);
    [[maybe_unused]] bool did_loop = false;
    while (run_count < num_rle_stream_decode_warps && output_pos < max_count && cur < end) {
      did_loop = true;
      int ri = rolling_index<run_buffer_size>(run_index);
      auto& run = runs[rolling_index<run_buffer_size>(run_index)];

      // Encoding::RLE

      // bytes for the varint header
      uint8_t const* _cur = cur;
      int const level_run = get_vlq32(_cur, end);
      int run_bytes       = _cur - cur;

      // literal run
      if (level_run & 1) {
        int const run_size  = (level_run >> 1) * 8;
        run.size            = run_size; //valid count // print this
        #ifdef ABDEBUG
        printf("literal dict? %i ri: %i level_bits %i run: idx %i, output_pos %i, max_count: %i size %i start %" PRIu64 " cur: %" PRIu64 " end: %" PRIu64 "\n", 
          print_it,
          ri,
          level_bits,
          run_index,
          output_pos,
          max_count,
          run.size,
          (uint64_t)start, 
          (uint64_t)cur, 
          (uint64_t)end);
        #endif

        int const run_size8 = (run_size + 7) >> 3;
        run_bytes += run_size8 * level_bits;
      }
      // repeated value run
      else {
        run.size = (level_run >> 1);
        run_bytes++;
        #ifdef ABDEBUG
          printf("repeated dict? %i ri %i level_bits %i run: idx %i, outpot_pos: %i max_count: %i size %i start %" PRIu64 " cur: %" PRIu64 " end: %" PRIu64 "\n", 
          print_it,
          ri,
          level_bits,
          run_index,
          output_pos,
          max_count,
          run.size,
          (uint64_t)start, 
          (uint64_t)cur, 
          (uint64_t)end);
        #endif
        // can this ever be > 16?  it effectively encodes nesting depth so that would require
        // a nesting depth > 64k.
        if (level_bits > 8) { run_bytes++; }
      }
      run.output_pos = output_pos;
      run.start      = _cur;
      run.level_run  = level_run;
      #ifdef ABDEBUG   
      printf("run_index: %i ri: %i run.level_run is %i output_pos %i run.size %i\n", 
        run_index, ri, run.level_run, output_pos, run.size);
      #endif
      run.remaining  = run.size;
      cur += run_bytes;

      output_pos += run.size;
      run_count++;
      run_index++;
    }

    // the above loop computes a batch of runs to be processed. mark down
    // the number of runs because the code after this point resets run_count
    // for the next batch. each batch is returned via get_next_batch().

    next_batch_run_count = run_count;

    // -------------------------------------
    // prepare for the next run:

    // if we've reached the value output limit on the last run
    if (output_pos >= max_count) {
      // first, see if we've spilled over
      // TODO: AB run_buffer_size.. does it match def_runs size??
      int ri_to_spill  = rolling_index<run_buffer_size>(run_index - 1);
      int ri_spilling_to = rolling_index<run_buffer_size>(run_index);
      auto const& src       = runs[rolling_index<run_buffer_size>(run_index - 1)];
      auto const& spillx = runs[rolling_index<run_buffer_size>(run_index)];
      int const spill_count = output_pos - max_count;
      #ifdef ABDEBUG    
      printf("will spill did_loop %i dict? %i to_spill ri %i spilling to ri %i, spill_count? %i src.level_run %i spill.level_run %i\n", 
        did_loop, print_it, ri_to_spill, ri_spilling_to, spill_count, src.level_run, spillx.level_run);
      #endif

      // a spill has occurred in the current run. spill the extra values over into the beginning of
      // the next run.
      if (spill_count > 0) {
        int ri = rolling_index<run_buffer_size>(run_index);
        auto& spill_run      = runs[rolling_index<run_buffer_size>(run_index)];
        spill_run            = src;
        spill_run.output_pos = 0;
        spill_run.remaining  = spill_count;

        run_count = 1;
        run_index++;
        output_pos = spill_run.remaining;
        spill      = true;
        #ifdef ABDEBUG
        printf("did_loop %i spill ri: %i spilling dict? %i run idx: %i output_pos: %i max_count: %i src.level_run %i spill_run.level_run: %i \n", 
          did_loop ? 1 : 0, ri, print_it, run_index-1, output_pos, max_count, src.level_run, spill_run.level_run);
        #endif
      }
      // no actual spill needed. just reset the output pos
      else {
        #ifdef ABDEBUG
        //printf("NOT spilling dict? %i run idx: %i output_pos: %i max_count: %i \n", 
        //print_it, run_index-1, output_pos, max_count);
        #endif
        output_pos = 0;
        run_count  = 0;
      }
    }
    // didn't cross the limit, so reset the run count
    else {
      run_count = 0;
    }
  }

  template<typename other_run_t>
  __device__ int decode_next(int t, int count, int roll, int print_it, other_run_t other_run)
  {
    int const output_count = 
      count < 0 ? 
        min(max_output_values, (total_values - cur_values)) : 
        count; // should be 256

    if (t == 0) {
    printf("at decode next for dictionary %i max_output_values %i total_values %i cur_values %i count %i roll %i\n",
      print_it, max_output_values, total_values, cur_values, count, roll);
    }

    // special case. if level_bits == 0, just return all zeros. this should tremendously speed up
    // a very common case: columns with no nulls, especially if they are non-nested
    if (level_bits == 0) {
      printf("level_bits is zero???\n");
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
        if (warp_lane == 0) { fill_run_batch(output_count, print_it); }
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
        printf("run dict? %i level_run %i warp_decode_id: %i remaining %i output_count %i run.output_pos %i max_size %i\n", 
          print_it,
          run.level_run,
          warp_decode_id,
            run.remaining, 
            output_count, 
            run.output_pos, 
            min(run.remaining, (output_count - run.output_pos)));
        #endif
        // TODO: we must be running off the end of output here.. so output_pos
        // for the dictionary stream is bogus
        auto batch = run.next_batch(output, run.output_pos,
                                    min(run.remaining, (output_count - run.output_pos)), print_it);

        #ifdef ABDEBUG
        if (warp_lane == 0) {
          for (int i = 0; i < 6; i++){ 
            printf("before_decode  dict? %i rix %i level_run %i\n", 
              print_it, i, runs[i].level_run);
          }
          if (other_run != nullptr) {
            for (int i = 0; i < 6; i++){ 
              printf("before_decode other run rix %i level_run %i\n", 
                i, 
                (int)(other_run[i].level_run));
            }
          }
          
        }
          #endif
        batch.decode(
          end, level_bits, warp_lane, warp_decode_id, roll, print_it, other_run);

        #ifdef ABDEBUG
        if (warp_lane == 0) {
          for (int i = 0; i < 6; i++){ 
            printf("after_decode dict? %i rix %i level_run %i\n", print_it, i, runs[i].level_run);
          }
          if (other_run != nullptr) {
            for (int i = 0; i < 6; i++){ 
              printf("after_decode other run rix %i level_run %i\n", 
                i, 
                (int)(other_run[i].level_run));
            }
          }
        }
        #endif
        // last warp updates total values processed
        // TODO: why?? abellina . Why is the first warp doing X an the last warp doing Y
        if (warp_lane == 0 && warp_decode_id == num_runs - 1) {
          values_processed = run.output_pos + batch.size;
        }
      }
      __syncthreads();

      // if we haven't run out of space, retrieve the next batch. otherwise leave it for the next
      // call.
      if (!t && values_processed < output_count) {
        thrust::tie(run_start, num_runs) = get_run_batch();
      }
      __syncthreads();
      // TODO: abellina what bounds values_processed? does num_rows decrement?
    } while (num_runs > 0 && values_processed < output_count);

    cur_values += values_processed;

    // valid for every thread
    return values_processed;
  }

  __device__ inline int decode_next(int t) {
    return decode_next<rle_run<uint32_t, 1>*>(t, -1, 0, 0, nullptr);
  }

};

}  // namespace cudf::io::parquet::gpu