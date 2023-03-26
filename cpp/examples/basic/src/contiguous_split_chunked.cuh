#include <vector>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace cudf { 

class table_view;
    
namespace chunked {

namespace detail {
  struct contiguous_split_state;
};

class chunked_contiguous_split {
  public:
    explicit chunked_contiguous_split(
        cudf::table_view const& input,
        cudf::device_span<uint8_t> const& user_buffer,
        rmm::cuda_stream_view stream,
        rmm::mr::device_memory_resource* mr);

    ~chunked_contiguous_split();

    bool has_next() const;

    std::size_t next();

    std::vector<packed_columns::metadata> const& make_packed_columns() const;

  private:
    // internal state of contiguous split
    std::unique_ptr<detail::contiguous_split_state> state;
};

class contiguous_split {
  public:
    explicit contiguous_split(
        cudf::table_view const& input,
        std::vector<size_type> const& splits,
        rmm::cuda_stream_view stream,
        rmm::mr::device_memory_resource* mr);

    ~contiguous_split();

    std::vector<std::pair<packed_columns::metadata, rmm::device_buffer>> make_packed_columns();

  private:
    // internal state of contiguous split
    std::unique_ptr<detail::contiguous_split_state> state;
};

}};