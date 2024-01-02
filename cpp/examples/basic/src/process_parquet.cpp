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
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>

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
  auto res = cudf::io::read_parquet(options);
  std::cout << "table of " << res.tbl->num_rows() << " rows scanned" << std::endl;
  return res;
}

void simple_int_column(int num_rows)
{  
  std::string filepath("/home/abellina/table_with_dict.parquet");

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i == 123 || i == 555 ? 0 : 1; });
  /// 0, [](auto i) { return 1; });
  //  0, [](auto i) { return i == 123 || i == 777 ? 0 : 1; });
  auto iter1 = cudf::detail::make_counting_transform_iterator(0, [](int i) { return i % 10; });
  cudf::test::fixed_width_column_wrapper<int> col1(iter1, iter1 + num_rows, valids);
  //cudf::test::fixed_width_column_wrapper<int> col1(iter1, iter1 + num_rows);
  auto tbl = cudf::table_view{{col1}}; 
  
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)    
    .dictionary_policy(cudf::io::dictionary_policy::ALWAYS);
  cudf::io::write_parquet(out_opts);
}

int main(int argc, char** argv)
{
  cudaSetDevice(0);
  //auto resource       = cudf::test::create_memory_resource("pool");
  auto resource = cudf::test::create_memory_resource("cuda");
  // auto resource       = cudf::test::create_memory_resource("async");
  rmm::mr::set_current_device_resource(resource.get());

  // Read data
  //auto store_sales = read_parquet("/home/abellina/part-00191-9dcfb50c-76b0-4dbf-882b-b60e7ad5b925.c000.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/part-00000-f7d7d84a-8f00-4921-95d1-ef17638a1c83-c000.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/second_mill.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/s_1.5.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/s_1.25.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/s_1.125.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/s_1.005.snappy.parquet");
  //auto store_sales = read_parquet("/home/abellina/cudf/part-00000-f7d7d84a-8f00-4921-95d1-ef17638a1c83-c000.snappy.parquet");
  ///cudaDeviceSynchronize();
// std::cout <<"done1"<<std::endl;
  //auto store_sales = read_parquet("/home/abellina/cudf/first_1m.snappy.parquet");

// ENABLE THIS
 const char* name = nullptr;
 if (argc > 1){
   name = argv[1];
   auto store_sales2 = read_parquet(name);
   cudaDeviceSynchronize();
 } else {
   auto store_sales2 = read_parquet(
     "/home/abellina/cudf/s_1_1.snappy.parquet");
   cudaDeviceSynchronize();
 }
  

 //[[maybe_unused]] int num_rows = 128;
 //if (argc > 1) {
 //  num_rows = atoi(argv[1]);
 //}
 //simple_int_column(num_rows);
 //auto simple = read_parquet("/home/abellina/table_with_dict.parquet");

  //std::cout << "over here: " << cudf::test::to_string(simple.tbl->get_column(0).view(), std::string(",")) << std::endl;
  std::cout << "done" << std::endl;

  return 0;
}
