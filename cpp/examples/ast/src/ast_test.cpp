/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/transform.hpp>
#include <cudf_test/column_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <vector>

int main(int argc, char** argv)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource mr{&cuda_mr, rmm::percent_of_free_device_memory(50)};
  rmm::mr::set_current_device_resource(&mr);

  auto expressions = std::list<cudf::ast::operation>();
  auto column_refs = std::vector<cudf::ast::column_reference>();

  int row_count = 100000;

  auto col = 
    cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, row_count);

  auto real_col =  
    cudf::fill(col->view(), 0, row_count, cudf::numeric_scalar<int32_t>(1));
 
  auto col_view = real_col->view();
  auto cols = std::vector<cudf::column_view>{col_view, col_view};
  auto table = cudf::table_view(cols);

  std::unique_ptr<cudf::column> binop_result;

  for (int i = 0; i < 100; i++) { 
      auto col = cudf::binary_operation(
      cols.at(0), 
      cols.at(1), 
      cudf::binary_operator::ADD, 
      cudf::data_type{cudf::type_id::INT32});

      auto col2 = cudf::binary_operation(
      col->view(), 
      cols.at(1), 
      cudf::binary_operator::ADD, 
      cudf::data_type{cudf::type_id::INT32});

      auto col3 = cudf::binary_operation(
      col2->view(), 
      cols.at(1), 
      cudf::binary_operator::ADD, 
      cudf::data_type{cudf::type_id::INT32});

      auto col4 = cudf::binary_operation(
      col3->view(), 
      cols.at(1), 
      cudf::binary_operator::ADD, 
      cudf::data_type{cudf::type_id::INT32});

      auto col5 = cudf::binary_operation(
      col4->view(), 
      cols.at(1), 
      cudf::binary_operator::ADD, 
      cudf::data_type{cudf::type_id::INT32});

      binop_result = cudf::binary_operation(
      col5->view(), 
      cols.at(1), 
      cudf::binary_operator::ADD, 
      cudf::data_type{cudf::type_id::INT32});
  }

  {
    auto col_null_mask = cudf::test::to_host<int32_t>(binop_result->view());
    std::cout << "binop result: " << std::endl;
    for (int i = 0; i < 10; i++) {
      std::cout << i << ": " << col_null_mask.first[i] << std::endl;
    }
  }

  column_refs.push_back(cudf::ast::column_reference(0));
  column_refs.push_back(cudf::ast::column_reference(1));

  expressions.push_back(
    cudf::ast::operation(
      cudf::ast::ast_operator::ADD,
      column_refs.at(0),
      column_refs.at(1)));

  expressions.push_back(
    cudf::ast::operation(
      cudf::ast::ast_operator::ADD,
      expressions.back(),
      column_refs.at(1)));

  expressions.push_back(
    cudf::ast::operation(
      cudf::ast::ast_operator::ADD,
      expressions.back(),
      column_refs.at(1)));

  expressions.push_back(
    cudf::ast::operation(
      cudf::ast::ast_operator::ADD,
      expressions.back(),
      column_refs.at(1)));

  expressions.push_back(
    cudf::ast::operation(
      cudf::ast::ast_operator::ADD,
      expressions.back(),
      column_refs.at(1)));

  expressions.push_back(
    cudf::ast::operation(
      cudf::ast::ast_operator::ADD,
      expressions.back(),
      column_refs.at(1)));

  std::unique_ptr<cudf::column> ast_result;
  auto const& root = expressions.back();
  for (int i = 0; i < 100; i++) { 
    ast_result = cudf::compute_column(table, root);
  }
  {
    auto col_null_mask = cudf::test::to_host<int32_t>(ast_result->view());
    std::cout << "ast result: " << std::endl;
    for (int i = 0; i < 10; i++) {
      std::cout << i << ": " << col_null_mask.first[i] << std::endl;
    }
  }

  return 0;
}
