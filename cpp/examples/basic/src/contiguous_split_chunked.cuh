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
};

class chunked_contiguous_split {
  public:
    explicit chunked_contiguous_split(
        cudf::table_view const& input,
        void* user_buffer,
        std::size_t user_buffer_size,
        rmm::cuda_stream_view stream,
        rmm::mr::device_memory_resource* mr);

    ~chunked_contiguous_split();

    bool has_next() const;

    std::size_t next();

    std::vector<packed_columns::metadata> make_packed_columns();
  private:
    detail::the_state* state;
};

}};