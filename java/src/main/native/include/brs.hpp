#pragma once

#include <vector>
#include "jni_utils.hpp"

/**
 * block_t must conform to the block_with_size trait

    class block_with_size {
        virtual uint64_t size() const;
    };
*/

template <typename block_t> 
struct block_range {
  block_t block;
  uint64_t range_start;
  uint64_t range_end;

  uint64_t range_size() { return range_end - range_start; }

  bool is_complete() { return range_end == block.size(); }
};

template <typename block_t>
class windowed_block_iterator {
private:
    struct block_with_offset { 
        block_t block;
        uint64_t start_offset;
        uint54_t end_offset;
    };

    struct block_window {
        block_window(uint64_t _start, uint64_t _size)
            : start(_start),
              size(_size),
              end(_start + _size)
        {}
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
          m_window{0, window_size},
          m_done(false),
          m_last_seen_block(0)
    {
        auto it = m_blocks.begin();
        uint64_t last_offset = 0;
        while (it != m_blocks.end()) {
            uint64_t start_offset = last_offset;
            uint64_t end_offset = start_offset + it->size;
            last_offset = end_offset;
            m_blocks_with_offsets.emplace_back{*it, start_offset, end_offset};
            it++;
        }
    }

    blocks_for_window get_blocks_for_window(
        const block_window& window, 
        int starting_block) {
        std::vector<block_range<block_t>> block_ranges_in_window;
        bool do_continue = true;
        int this_block = starting_block;
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
                block_ranges_in_window.emplace_back {b.block, range_start, range_end};
                last_block = this_block;
            } else {
                do_continue = b.end_offset <= window.start;
            }
            this_block++;
        }
        int last_block = block_ranges_in_window.last;
        return {last_block, block_ranges_in_window, 
                    !do_continue || !last_block.is_complete};
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

class bounce_buffer {
    uint64_t* address;
    uint64_t size;
};

class receive_block {
public: 
    uint64_t size() { 
    }

    void* user_data() {
    }
};

struct buffer_receive_result {
    // packed_tables_with_metas
    // spillable??

};

class buffer_receive_state {
private:
    struct copy_action {
        uint64_t* dst_base;
        uint64_t src_offset;
        uint64_t dst_offset;
        uint64_t copy_size;
        bool complete;
        void* user_data;
    };

public:
    buffer_receive_state(
        bounce_buffer const & bb, 
        std::vector<receive_block> const& blocks, 
        cudf::cuda_stream_view stream)
        : m_bb(bb), m_blocks(blocks), m_stream(stream), m_iterated(false)
    {
        m_window_iter = windowed_block_iterator<receive_block>(blocks, bb.size);
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
        advance();

        // TODO: maybe this is in java?
        // TODO: shuffle_thread_working_on_tasks(tasks)

        std::vector<copy_action> copy_actions;
        for (const block_range<receive_block>& br : m_current_blocks) {
            uint64_t full_size = br.block.size();
            if (full_size == br.range_size()) {
                // add copy action for a full buffer
                copy_actions.emplace_back {
                    allocate(full_size, stream),
                    bounce_buffer_byte_offset,
                    0,
                    full_size,
                    true,
                    br.block.user_data()};
            } else {
                // copy actions for a partial buffer
                if (working_on_offset != 0) {
                    copy_actions.emplace_back {
                        working_on_buffer, 
                        bounce_buffer_byte_offset,
                        working_on_offset,
                        br.range_size(),
                        working_on_offset + br.range_size() == full_size,
                        br.block.user_data()};

                    working_on_offset += br.range_size();
                    if (working_on_offset == full_size) {
                        working_on_offset = 0;
                    }
                } else {
                    working_on_buffer = allocate(full_size, stream);
                    copy_actions.emplace_back {
                        working_on_buffer, 
                        bounce_buffer_byte_offset,
                        working_on_offset,
                        br.range_size(),
                        working_on_offset + br.range_size() == full_size,
                        br.block.user_data()};
                    working_on_offset += br.range_size();
                }
            }
            bounce_buffer_byte_offset += br.range_size();
            if (bounce_buffer_byte_offset >= m_bb.size()) {
                bounce_buffer_byte_offset = 0;
            }
        }

        std::vector<uint64_t*> src_addresses;
        std::vector<uint64_t*> dst_addresses;
        std::vector<uint64_t> buffer_sizes;
        for (const copy_action& ca : copy_actions) {
            src_addresses.push_back(m_bb.address + ca.src_offset);
            dst_addresses.push_back(ca.dst_base + ca.dst_offset);
            buffer_sizes.push_back(ca.copy_size);
        }
        cudf::batch_memcpy(
            src_addresses.data(),
            dst_addresses.data(),
            dst_addresses.data(),
            src_addresses.size(),
            stream,
            mr);

        std::vector<buffer_receive_result> result;
        for (const copy_action& ca : copy_actions) {
            if (ca.complete) { 
                result.emplace_back {ca.dst_base, ca.user_data};
            }
        }
        if (working_on_offset == 0) {
            working_on_buffer = nullptr;
        }

        stream.synchronize();

        // TODO maybe these two are in java
        // TODO: rmmspark stop working on tasks
        // TODO: tofinalize needs to be called at some point

        return result;
    }

private:
    bounce_buffer m_bb;
    cudf::cuda_stream_view m_stream;
    windowed_block_iterator<receive_block> m_window_iter;
    std::vector<block_range<receive_block>> m_next_blocks;
    std::vector<block_range<receive_block>> m_current_blocks;
    bool m_iterated;
    bool m_has_more_blocks;
};