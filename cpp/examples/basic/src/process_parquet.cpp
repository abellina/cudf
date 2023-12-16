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

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

cudf::io::table_with_metadata read_parquet(std::string const& file_path)
{
  std::cout << "reading parquet file: " << file_path << std::endl;
  auto source_info = cudf::io::source_info(file_path);
  auto builder     = cudf::io::parquet_reader_options::builder(source_info);
  auto options     = builder.build();
  return cudf::io::read_parquet(options);
}

void simple_int_column(int num_rows)
{  
  std::string filepath("/home/abellina/table_with_dict.parquet");
  auto valids = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0 ? 1 : 0; });
  auto iter1 = cudf::detail::make_counting_transform_iterator(0, [](int i) { return 1; });
  cudf::test::fixed_width_column_wrapper<int> col1(iter1, iter1 + num_rows, valids);
  auto tbl = cudf::table_view{{col1}}; 
  
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)    
    .dictionary_policy(cudf::io::dictionary_policy::ALWAYS);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);
}

int main(int argc, char** argv)
{
  // Construct a CUDA memory resource using RAPIDS Memory Manager (RMM)
  // This is the default memory resource for libcudf for allocating device memory.
  rmm::mr::cuda_memory_resource cuda_mr{};
  // Construct a memory pool using the CUDA memory resource
  // Using a memory pool for device memory allocations is important for good performance in libcudf.
  // The pool defaults to allocating half of the available GPU memory.
  rmm::mr::pool_memory_resource mr{&cuda_mr};

  // Set the pool resource to be used by default for all device memory allocations
  // Note: It is the user's responsibility to ensure the `mr` object stays alive for the duration of
  // it being set as the default
  // Also, call this before the first libcudf API call to ensure all data is allocated by the same
  // memory resource.
  rmm::mr::set_current_device_resource(&mr);

  // Read data
  auto store_sales = read_parquet("/home/abellina/part-00191-9dcfb50c-76b0-4dbf-882b-b60e7ad5b925.c000.snappy.parquet");
  int num_rows = 128;
  if (argc > 1) {
    num_rows = atoi(argv[1]);
  }
  //simple_int_column(num_rows);
  //auto simple = read_parquet("/home/abellina/table_with_dict.parquet");

  std::cout << "over here" << std::endl;


  return 0;
}
