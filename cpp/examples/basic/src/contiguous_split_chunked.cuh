#include <vector>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace cudf { 

class table_view;
    
namespace chunked {

namespace detail {
  struct the_state;
}

class chunked_contiguous_split {
  public:
    explicit chunked_contiguous_split(
        cudf::table_view& const input,
        uint8_t* user_buffer,
        std::size_t user_buffer_size,
        rmm::cuda_stream_view stream,
        rmm::mr::device_memory_resource* mr);

    bool has_next() const;

    std::size_t next();

    std::vector<packed_columns::metadata> make_packed_columns();
  private:
    detail::the_state state;
};

//std::pair<bool, cudf::size_type> contiguous_split(
//  cudf::table_view const& input,
//  std::vector<size_type> const& splits,
//  rmm::device_buffer* user_provided_out_buffer,
//  detail::the_state*& user_state,
//  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

//std::vector<packed_columns::metadata> make_packed_columns(detail::the_state* state);

}};