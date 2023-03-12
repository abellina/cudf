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

std::vector<packed_table> contiguous_split(cudf::table_view const& input,
                                           std::vector<size_type> const& splits,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::vector<packed_table> contiguous_split(cudf::table_view const& input,
                                           std::vector<size_type> const& splits,
                                           rmm::device_buffer* user_provided_out_buffer,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

   //detail::the_state* chunked_contiguous_split(cudf::table_view const& input,
   //                                            std::vector<size_type> const& splits,
   //                                            rmm::device_buffer* user_provided_out_buffer,
   //                                        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}};