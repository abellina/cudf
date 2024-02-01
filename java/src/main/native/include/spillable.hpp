#pragma once

enum spill_location_t { 
    DEVICE,
    HOST,
    DISK
};

struct spill_buffer_t {
    uint64_t id;
    void * data;
    uint64_t size;
    spill_location_t location;
};

class spill_store_t {
public:
    spill_store_t(spill_store_t& downstream_store)
        : my_spill_store(downstream_store) {}

private:
    std::unordered_map<uint64_t, spill_buffer_t> buffers;
    spill_store_t& my_spill_store;
};

class spill_catalog {
public:
    spill_catalog(): id(0) {
        spill_stores[spill_location_t::DISK]   = spill_store_t();
        spill_stores[spill_location_t::HOST]   = spill_store_t(spill_stores[spill_location_t::DISK]);
        spill_stores[spill_location_t::DEVICE] = spill_store_t(spill_stores[spill_location_t::HOST]);
    }

    uint64_t add_buffer(void * data, uint64_t size, spill_location_t location) {
        auto new_id = create_id();
        buffers[new_id] = {new_id, data, size, location};
        return new_id;
    };

    uint64_t create_id() { 
        return id++;
    }

    spill_buffer_t get_buffer(uint64_t id) {
        return buffers[new_id];
    }

    uint64_t spill(uint64_t amount_needed, spill_location_t location) {
        return 0;
    }

private:
    std::unordered_map<spill_location_t, spill_store_t> spill_stores;
    uint64_t id;
};
