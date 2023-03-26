/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>
#include <cudf/copying.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "contiguous_split_chunked.cuh"

cudf::io::table_with_metadata read_csv(std::string const& file_path)
{
  auto source_info = cudf::io::source_info(file_path);
  auto builder     = cudf::io::csv_reader_options::builder(source_info);
  auto options     = builder.build();
  return cudf::io::read_csv(options);
}

void write_csv(cudf::table_view const& tbl_view, std::string const& file_path)
{
  auto sink_info = cudf::io::sink_info(file_path);
  auto builder   = cudf::io::csv_writer_options::builder(sink_info, tbl_view);
  auto options   = builder.build();
  cudf::io::write_csv(options);
}

std::vector<cudf::groupby::aggregation_request> make_single_aggregation_request(
  std::unique_ptr<cudf::groupby_aggregation>&& agg, cudf::column_view value)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].aggregations.push_back(std::move(agg));
  requests[0].values = value;
  return requests;
}

std::unique_ptr<cudf::table> average_closing_price(cudf::table_view stock_info_table)
{
  // Schema: | Company | Date | Open | High | Low | Close | Volume |
  auto keys = cudf::table_view{{stock_info_table.column(0)}};  // Company
  //auto val  = stock_info_table.column(5);                      // Close
  auto val  = stock_info_table.column(1);                      // Close

  // Compute the average of each company's closing price with entire column
  cudf::groupby::groupby grpby_obj(keys);
  auto requests =
    make_single_aggregation_request(cudf::make_mean_aggregation<cudf::groupby_aggregation>(), val);

  auto agg_results = grpby_obj.aggregate(requests);

  // Assemble the result
  auto result_key = std::move(agg_results.first);
  auto result_val = std::move(agg_results.second[0].results[0]);
  std::vector<cudf::column_view> columns{result_key->get_column(0), *result_val};
  return std::make_unique<cudf::table>(cudf::table_view(columns));
}

int main(int argc, char** argv)
{
  // Construct a CUDA memory resource using RAPIDS Memory Manager (RMM)
  // This is the default memory resource for libcudf for allocating device memory.
  std::cout << "initializing memory resource" << std::endl;
  rmm::mr::cuda_memory_resource cuda_mr{};
  // Construct a memory pool using the CUDA memory resource
  // Using a memory pool for device memory allocations is important for good performance in libcudf.
  // The pool defaults to allocating half of the available GPU memory.
  std::cout << "initializing pool" << std::endl;
  rmm::mr::pool_memory_resource mr{&cuda_mr};

  // Set the pool resource to be used by default for all device memory allocations
  // Note: It is the user's responsibility to ensure the `mr` object stays alive for the duration of
  // it being set as the default
  // Also, call this before the first libcudf API call to ensure all data is allocated by the same
  // memory resource.
  std::cout << "setting current resource" << std::endl;
  rmm::mr::set_current_device_resource(&mr);

  // Read data
  std::cout << "reading some csv" << std::endl;
  auto stock_table_with_metadata = read_csv("/home/abellina/mock.csv");

  // Process
  std::cout << "doing some compute" << std::endl;
  auto result = average_closing_price(*stock_table_with_metadata.tbl);
  // write it the normal way
  write_csv(*result, "4stock_5day_avg_close.csv");

  // contig split it
  auto tv = result->select(std::vector<cudf::size_type>{0,1});
  std::cout << "calling contig split" << std::endl;

  rmm::device_buffer bounce_buff(100000000, cudf::get_default_stream(), &mr);
  rmm::device_buffer final_buff(500000000, cudf::get_default_stream(), &mr);

  // TODO: we'd new this up in JNI
  auto cs = cudf::chunked::chunked_contiguous_split(
    tv, 
    cudf::device_span<uint8_t>(static_cast<uint8_t*>(bounce_buff.data()), bounce_buff.size()), 
    cudf::get_default_stream(),
    rmm::mr::get_current_device_resource());

  cudf::size_type final_buff_offset = 0;
  while(cs.has_next()) {
    auto bytes_copied = cs.next();
    cudaMemcpyAsync(
      (uint8_t*)final_buff.data() + final_buff_offset,
      bounce_buff.data(),
      bytes_copied,
      cudaMemcpyDefault,
      cudf::get_default_stream());
    std::cout << "copied in this iteration: " << bytes_copied << ". have next? " << cs.has_next() << std::endl;
    final_buff_offset += bytes_copied;
  }
  auto packed_columns = cs.make_packed_columns();

  // Write out result
  std::cout << "writing result out, see " 
            << packed_columns.size() << " results" << std::endl;
  
  auto meta = packed_columns[0].data();

  // TODO: revisit unpack iterface passing the packed_columns themselves
  auto unpacked= cudf::unpack(
    meta,
    (const uint8_t*)final_buff.data());
  write_csv(unpacked, "4stock_5day_avg_close_cs.csv");


  std::vector<cudf::size_type> splits{tv.num_rows()/2};

  auto reg_cs = cudf::chunked::contiguous_split(
    tv, 
    splits,
    cudf::get_default_stream(),
    rmm::mr::get_current_device_resource());

  auto packed_tables = reg_cs.make_packed_columns();

  // Write out result
  std::cout << "reg: writing result out, see " 
            << packed_columns.size() << " results" << std::endl;

  auto unpacked2 = cudf::unpack(packed_tables[0].first.data(), (uint8_t*)packed_tables[0].second.data());
  auto unpacked3 = cudf::unpack(packed_tables[1].first.data(), (uint8_t*)packed_tables[1].second.data());

  write_csv(unpacked2, "4stock_5day_avg_close_cs_reg_first.csv");
  write_csv(unpacked3, "4stock_5day_avg_close_cs_reg_second.csv");

  return 0;
}
