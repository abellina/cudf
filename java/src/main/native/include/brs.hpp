#pragma once

#include <vector>
#include <rmm/cuda_stream_view.hpp>
#include "jni_utils.hpp"

template <typename block_t> 
struct block_range {
  block_t block;
  int64_t range_start;
  int64_t range_end;

  uint64_t range_size() const { return range_end - range_start; }

  bool is_complete() const { return range_end == static_cast<int64_t>(block.size); }
};

template <typename block_t>
struct blocks_for_window {
    blocks_for_window(
        int last_block_, 
        std::vector<block_range<block_t>> const& block_ranges_,
        bool has_more_blocks_) {
        last_block = last_block_;
        block_ranges = block_ranges_;
        has_more_blocks = has_more_blocks_;
    }
    int last_block;
    std::vector<block_range<block_t>> block_ranges;
    bool has_more_blocks;
};

template <typename block_t>
class windowed_block_iterator {
private:
    struct block_with_offset { 
        block_t block;
        uint64_t start_offset;
        uint64_t end_offset;
    };

    struct block_window {
        block_window(uint64_t _start, uint64_t _size)
            : start(_start),
              size(_size),
              end(_start + _size)
        {}

        block_window move() const {
            return block_window { start + size, size };
        }
        uint64_t start;
        uint64_t size;
        uint64_t end;
    };

public:
    windowed_block_iterator(
        const std::vector<block_t>& blocks, 
        std::size_t window_size)
        : m_blocks(blocks),
          m_window_size(window_size),
          m_window(0, window_size),
          m_done(false),
          m_last_seen_block(0)
    {
        auto it = m_blocks.begin();
        uint64_t last_offset = 0;
        while (it != m_blocks.end()) {
            uint64_t start_offset = last_offset;
            uint64_t end_offset = start_offset + it->size;
            last_offset = end_offset;
            m_blocks_with_offsets.push_back(block_with_offset{
                *it, start_offset, end_offset});
            it++;
        }
    }

    blocks_for_window<block_t> get_blocks_for_window(
        const block_window& window, 
        int starting_block) {
        std::vector<block_range<block_t>> block_ranges_in_window;
        bool do_continue = true;
        std::size_t this_block = starting_block;
        int last_block = -1;
        while (do_continue && this_block < m_blocks_with_offsets.size()) {
            auto& b = m_blocks_with_offsets[this_block];
            if (window.start < b.end_offset && window.end > b.start_offset) {
                int range_start = window.start - b.start_offset;
                if (range_start < 0) {
                    range_start = 0;
                }
                int range_end = window.end - b.start_offset;
                if (window.end >= b.end_offset) {
                    range_end = b.end_offset - b.start_offset;
                }
                block_ranges_in_window.push_back(
                    block_range<block_t>{
                        b.block, range_start, range_end
                    });
                last_block = this_block;
            } else {
                do_continue = b.end_offset <= window.start;
            }
            this_block++;
        }
        auto last = block_ranges_in_window[block_ranges_in_window.size()-1];
        return blocks_for_window<block_t> (
            last_block,
            block_ranges_in_window, 
            !do_continue || !last.is_complete());
    }

    std::vector<block_range<block_t>> next() {
        auto blocks_w = get_blocks_for_window(m_window, m_last_seen_block);
        m_last_seen_block = blocks_w.last_block >= 0 ? blocks_w.last_block : 0;

        if (blocks_w.has_more_blocks) {
            m_window = m_window.move();
        } else {
            m_done = true;
        }

        return blocks_w.block_ranges;
    }

    bool has_next() { return !m_done && m_blocks_with_offsets.size() > 0; }

private:
    std::vector<block_t> m_blocks;
    std::size_t m_window_size;
    block_window m_window;
    bool m_done;
    std::vector<block_with_offset> m_blocks_with_offsets;
    int m_last_seen_block;
};

struct bounce_buffer {
    uint64_t* address;
    uint64_t size;
};

struct receive_block {
    uint64_t size;
    int id;
};

struct buffer_receive_result {
    int block_id;
    uint64_t * packed_buffer;
    uint64_t size;
};

class buffer_receive_state {
private:
    struct copy_action {
        uint64_t* dst_base;
        uint64_t src_offset;
        uint64_t dst_offset;
        uint64_t copy_size;
        bool complete;
        int id;
        uint64_t size;
    };

public:
    buffer_receive_state(
        bounce_buffer const & bb, 
        std::vector<receive_block> const& blocks, 
        rmm::cuda_stream_view stream,
        rmm::mr::device_memory_resource* mr)
        : m_bb(bb), 
          m_stream(stream),
          m_mr(mr),
          m_window_iter(blocks, bb.size),
          m_bounce_buffer_byte_offset(0),
          m_working_on_offset(0)
    {
        m_iterated = false;
        m_has_more_blocks = m_window_iter.has_next();
    }

    bool has_more_blocks() { return m_has_more_blocks; } 

    void advance() {
        if (!m_has_more_blocks) {
            throw std::runtime_error("buffer_receive_state is done yet it received a call to next()");
        }

        if (!m_iterated) {
            m_next_blocks = m_window_iter.next();
            m_iterated = true;
        }

        m_current_blocks = m_next_blocks;

        if (m_window_iter.has_next()) {
            m_next_blocks = m_window_iter.next();
        } else {
            m_next_blocks.clear();
            m_has_more_blocks = false;
        }
    }

    std::vector<buffer_receive_result> consume() {
        std::cout << "before advance" << std::endl;
        advance();
        std::cout << "after advance" << std::endl;

        // TODO: maybe this is in java?
        // TODO: shuffle_thread_working_on_tasks(tasks)

        std::vector<copy_action> copy_actions;
        for (const block_range<receive_block>& br : m_current_blocks) {
            uint64_t full_size = br.block.size;
            if (full_size == br.range_size()) {
                // add copy action for a full buffer
                copy_actions.push_back(copy_action {
                    reinterpret_cast<uint64_t*>(m_mr->allocate(full_size, m_stream)),
                    m_bounce_buffer_byte_offset,
                    0,
                    full_size,
                    true,
                    br.block.id,
                    br.block.size});
            } else {
                // copy actions for a partial buffer
                if (m_working_on_offset != 0) {
                    copy_actions.push_back(copy_action {
                        m_working_on_buffer, 
                        m_bounce_buffer_byte_offset,
                        m_working_on_offset,
                        br.range_size(),
                        m_working_on_offset + br.range_size() == full_size,
                        br.block.id,
                        br.block.size});

                    m_working_on_offset += br.range_size();
                    if (m_working_on_offset == full_size) {
                        m_working_on_offset = 0;
                    }
                } else {
                    m_working_on_buffer = reinterpret_cast<uint64_t*>(
                        m_mr->allocate(full_size, m_stream));
                    copy_actions.push_back(copy_action{
                        m_working_on_buffer, 
                        m_bounce_buffer_byte_offset,
                        m_working_on_offset,
                        br.range_size(),
                        m_working_on_offset + br.range_size() == full_size,
                        br.block.id,
                        br.block.size});
                    m_working_on_offset += br.range_size();
                }
            }
            m_bounce_buffer_byte_offset += br.range_size();
            if (m_bounce_buffer_byte_offset >= m_bb.size) {
                m_bounce_buffer_byte_offset = 0;
            }
        }

        std::cout << "copy actions " << copy_actions.size() << std::endl;
        std::vector<uint64_t*> src_addresses;
        std::vector<uint64_t*> dst_addresses;
        std::vector<uint64_t> buffer_sizes;
        for (const copy_action& ca : copy_actions) {
            src_addresses.push_back(m_bb.address + ca.src_offset);
            dst_addresses.push_back(ca.dst_base + ca.dst_offset);
            buffer_sizes.push_back(ca.copy_size);
        }
        std::cout << "batch memcpy: " << src_addresses.size() << std::endl;
        cudf::batch_memcpy(
            src_addresses.data(),
            dst_addresses.data(),
            buffer_sizes.data(),
            src_addresses.size(),
            m_stream,
            m_mr);

        std::cout << "results" << std::endl;
        std::vector<buffer_receive_result> result;
        for (const copy_action& ca : copy_actions) {
            if (ca.complete) { 
                result.push_back(buffer_receive_result{
                    ca.id, ca.dst_base, ca.size
                });
            }
        }
        if (m_working_on_offset == 0) {
            m_working_on_buffer = nullptr;
        }

        m_stream.synchronize();

        // TODO maybe these two are in java
        // TODO: rmmspark stop working on tasks
        // TODO: tofinalize needs to be called at some point

        return result;
    }

private:
    bounce_buffer m_bb;
    rmm::cuda_stream_view m_stream;
    rmm::mr::device_memory_resource* m_mr;
    windowed_block_iterator<receive_block> m_window_iter;
    std::vector<block_range<receive_block>> m_next_blocks;
    std::vector<block_range<receive_block>> m_current_blocks;
    bool m_iterated;
    bool m_has_more_blocks;
    uint64_t m_bounce_buffer_byte_offset;
    uint64_t m_working_on_offset;
    uint64_t* m_working_on_buffer;
};