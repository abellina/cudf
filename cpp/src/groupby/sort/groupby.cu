/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Alexander Ocsa <alexander@blazingdb.com>
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

#include <algorithm>
#include <cassert>
#include <thrust/fill.h>
#include <tuple>

#include <bitmask/legacy/bit_mask.cuh>
#include <cudf/copying.hpp>
#include <cudf/cudf.h>
#include <cudf/groupby.hpp>
#include <cudf/legacy/bitmask.hpp>
#include <cudf/legacy/table.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <utilities/column_utils.hpp>
#include <utilities/cuda_utils.hpp>

#include "../common/aggregation_requests.hpp"
#include "../common/utils.hpp"
#include "groupby.hpp"
#include "groupby_kernels.cuh"

#include <quantiles/groupby.hpp>
#include <quantiles/quantiles.hpp>

namespace cudf {
namespace groupby {
namespace sort {

using index_vector = rmm::device_vector<gdf_size_type>;

namespace {
 
struct quantiles_functor {

  template <typename T>
  std::enable_if_t<std::is_arithmetic<T>::value, void >
  operator()(gdf_column const& values_col,
             rmm::device_vector<gdf_size_type> const& group_indices,
             rmm::device_vector<gdf_size_type> const& group_sizes,
             gdf_column& result_col, rmm::device_vector<double> const& quantile,
             gdf_quantile_method interpolation)
  {
    // prepare args to be used by lambda below
    auto result = reinterpret_cast<double*>(result_col.data);
    auto values = reinterpret_cast<T*>(values_col.data);
    auto grp_id = group_indices.data().get();
    auto grp_size = group_sizes.data().get();
    auto d_quants = quantile.data().get();
    auto num_qnts = quantile.size();

    // For each group, calculate quantile
    thrust::for_each_n(thrust::device,
      thrust::make_counting_iterator(0),
      group_indices.size(),
      [=] __device__ (gdf_size_type i) {
        gdf_size_type segment_size = grp_size[i];

        for (gdf_size_type j = 0; j < num_qnts; j++) {
          gdf_size_type k = i * num_qnts + j;
          result[k] = cudf::detail::select_quantile(values + grp_id[i], segment_size,
                                              d_quants[j], interpolation);
        }
      }
    );
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic<T>::value, void >
  operator()(Args&&... args) {
    CUDF_FAIL("Only arithmetic types are supported in quantile");
  }
};

/**---------------------------------------------------------------------------*
 * @brief Computes the remaining original requests which were skipped in a previous  
 * process, in  the `compound_to_simple` function. Then combine these resutts with 
 * the set of output aggregation columns corresponding to not complex aggregation
 * requests
 *
 * @param groupby[in] The object for computing sort-based groupby
 * @param original_requests[in] The original set of potentially complex
 * aggregation requests
 * @param input_ops_args[in] The list of arguments fot each of the previous complex
 * aggregation requests
 * @param current_output_values[in] Set of output aggregation columns corresponding to
 * not complex aggregation requests
 * @param stream[in] CUDA stream on which to execute
 * @return vector of columns satisfying each of the original requests
 *---------------------------------------------------------------------------**/
std::vector<gdf_column*>  process_remaining_complex_request(
    cudf::detail::groupby &groupby,
    std::vector<AggRequestType> const& original_requests,
     std::vector<operation_args*> const& input_ops_args,
    cudf::table current_output_values,
    cudaStream_t stream) {

  std::vector<gdf_column*> output_value(original_requests.size());
  for (gdf_size_type i = 0; i < current_output_values.num_columns(); i++) {
    output_value[i] = current_output_values.get_column(i);
  }

  rmm::device_vector<gdf_size_type> group_indices = groupby.group_indices();
  index_vector group_labels = groupby.group_labels();

  for (size_t i = 0; i < original_requests.size(); ++i) {
    auto const& element = original_requests[i];
    if (is_complex_agg(element.second)) {
      std::vector<double> quantiles;
      gdf_quantile_method interpolation;

      if (element.second == MEDIAN) {
        quantiles.push_back(0.5);
        interpolation = GDF_QUANT_LINEAR;
      } else if (element.second == QUANTILE){
        quantile_args * args = static_cast<quantile_args*>(input_ops_args[i]);
        quantiles = args->quantiles;
        interpolation = args->interpolation;
      }
      gdf_column * value_col = element.first;
      gdf_column sorted_values;
      rmm::device_vector<gdf_size_type> group_sizes;

      std::tie(sorted_values, group_sizes) = groupby.sort_values(*value_col);
      gdf_column* result_col = new gdf_column;
      *result_col = cudf::allocate_column(
          GDF_FLOAT64, quantiles.size() * groupby.num_groups(), false);
      rmm::device_vector<double> dv_quantiles(quantiles);

      cudf::type_dispatcher(sorted_values.dtype, quantiles_functor{},
                          sorted_values, group_indices, group_sizes, 
                          *result_col,
                          dv_quantiles, interpolation);
      output_value[i] = result_col;
      gdf_column_free(&sorted_values);
    }
  }

  // Update size and null count of output columns
  std::transform(output_value.begin(), output_value.end(), output_value.begin(),
                 [](gdf_column *col) {
                   CUDF_EXPECTS(col != nullptr, "Attempt to update Null column.");
                   set_null_count(*col);
                   return col;
                 });
  return output_value;
}

/**---------------------------------------------------------------------------*
 * @brief Prepare input parameters for invoking the `aggregate_all_rows` kernel
 * which compute the simple aggregation(s) of corresponding rows in the output
 * `values` table.
 * @param input_keys The table of keys
 * @param options The options for controlling behavior of the groupby operation.
 * @param groupby The object for computing sort-based groupby
 * @param simple_values_columns The list of simple values columns
 * @param simple_operators The list of simple aggregation operations
 * @param stream[in] CUDA stream on which to execute
 * @return output value table with the aggregation(s) computed 
 *---------------------------------------------------------------------------**/
template <bool keys_have_nulls, bool values_have_nulls>
cudf::table compute_simple_request(const cudf::table &input_keys,
                               const Options &options,
                               cudf::detail::groupby &groupby,
                               const std::vector<gdf_column *> &simple_values_columns,
                               const std::vector<operators> &simple_operators,
                               cudaStream_t &stream) {
  gdf_column key_sorted_order = groupby.key_sorted_order();
  index_vector group_labels = groupby.group_labels();
  gdf_size_type num_groups = (gdf_size_type)groupby.num_groups();
  
  cudf::table simple_values_table{simple_values_columns};

  cudf::table simple_output_values{
      num_groups, target_dtypes(column_dtypes(simple_values_table), simple_operators),
      column_dtype_infos(simple_values_table), values_have_nulls, false, stream};

  initialize_with_identity(simple_output_values, simple_operators, stream);

  auto d_input_values = device_table::create(simple_values_table);
  auto d_output_values = device_table::create(simple_output_values, stream);
  rmm::device_vector<operators> d_ops(simple_operators);

  auto row_bitmask = cudf::row_bitmask(input_keys, stream);

  cudf::util::cuda::grid_config_1d grid_params{input_keys.num_rows(), 256};

  //Aggregate all rows for simple requests using the key sorted order (indices) and the group labels
  cudf::groupby::sort::aggregate_all_rows<keys_have_nulls, values_have_nulls><<<
      grid_params.num_blocks, grid_params.num_threads_per_block, 0, stream>>>(
      input_keys.num_rows(), *d_input_values, *d_output_values, (gdf_index_type *)key_sorted_order.data,
          group_labels.data().get(), options.ignore_null_keys,
          d_ops.data().get(), row_bitmask.data().get());

  return simple_output_values;
}

template <bool keys_have_nulls, bool values_have_nulls>
auto compute_sort_groupby(cudf::table const& input_keys, cudf::table const& input_values,
                          std::vector<operators> const& input_ops,
                          std::vector<operation_args*> const& input_ops_args,
                          Options options,
                          cudaStream_t stream) {
  auto include_nulls = not options.ignore_null_keys;
  auto groupby = cudf::detail::groupby(input_keys, include_nulls, options.null_sort_behavior, options.input_sorted);

  if (groupby.num_groups() == 0) {
    cudf::table output_values(0, target_dtypes(column_dtypes(input_values), input_ops), column_dtype_infos(input_values));
    return std::make_pair(
        cudf::empty_like(input_keys),
        output_values.get_columns()
        );
  }
  // An "aggregation request" is the combination of a `gdf_column*` to a column
  // of values, and an aggregation operation enum indicating the aggregation
  // requested to be performed on the column
  std::vector<AggRequestType> original_requests(input_values.num_columns());
  std::transform(input_values.begin(), input_values.end(), input_ops.begin(),
                 original_requests.begin(),
                 [](gdf_column const* col, operators op) {
                   return std::make_pair(const_cast<gdf_column*>(col), op);
                 });

  // Some aggregations are "compound", meaning they need be satisfied via the
  // composition of 1 or more "simple" aggregation requests. For example, MEAN
  // is satisfied via the division of the SUM by the COUNT aggregation. We
  // translate these compound requests into simple requests, and compute the
  // groupby operation for these simple requests. Later, we translate the simple
  // requests back to compound request results.
  std::vector<SimpleAggRequestCounter> simple_requests =
      compound_to_simple(original_requests);

  std::vector<gdf_column*> simple_values_columns;
  std::vector<operators> simple_operators;
  for (auto const& p : simple_requests) {
    const AggRequestType& agg_req_type = p.first;
    simple_values_columns.push_back(
        const_cast<gdf_column*>(agg_req_type.first));
    simple_operators.push_back(agg_req_type.second);
  }

  // If there are "simple" aggregation requests, compute the aggregations 
  cudf::table current_output_values{};
  if (simple_values_columns.size() > 0) {
    // Step 1: Aggregate all rows for simple requests 
    cudf::table simple_output_values = compute_simple_request<keys_have_nulls, values_have_nulls>(input_keys,
                              options,
                              groupby,
                              simple_values_columns,
                              simple_operators,
                              stream);
    // Step 2: If any of the original requests were compound, compute them from the
    // results of simple aggregation requests
    current_output_values = compute_original_requests(original_requests, simple_requests, simple_output_values, stream);
  }
  // If there are "complex" aggregation requests like MEDIAN, QUANTILE, compute these aggregations 
  std::vector<gdf_column*> final_output_values = process_remaining_complex_request(groupby, original_requests, input_ops_args, current_output_values, stream);
  return std::make_pair(groupby.unique_keys(), final_output_values);
}

/**---------------------------------------------------------------------------*
 * @brief Returns appropriate callable instantiation of `compute_sort_groupby`
 * based on presence of null values in keys and values.
 *
 * @param keys The groupby key columns
 * @param values The groupby value columns
 * @return Instantiated callable of compute_sort_groupby
 *---------------------------------------------------------------------------**/
auto groupby_null_specialization(table const& keys, table const& values) {
  if (cudf::has_nulls(keys)) {
    if (cudf::has_nulls(values)) {
      return compute_sort_groupby<true, true>;
    } else {
      return compute_sort_groupby<true, false>;
    }
  } else {
    if (cudf::has_nulls(values)) {
      return compute_sort_groupby<false, true>;
    } else {
      return compute_sort_groupby<false, false>;
    }
  }
}
} // anonymous namespace

namespace detail { 
 

/**---------------------------------------------------------------------------*
 * @brief Verifies the requested aggregation is valid for the arguments of the 
 * operator.
 *
 * @throw cudf::logic_error if an invalid combination of argument and operator
 * is requested.
 *
 * @param ops The aggregation operators
 * @param ops The aggregation arguments
 *---------------------------------------------------------------------------**/
static void verify_operators_with_arguments(std::vector<operators> const& ops, std::vector<operation_args*> const& args) {
   CUDF_EXPECTS(ops.size() == args.size(),
               "Size mismatch between ops and args");
  for (size_t i = 0; i < ops.size(); i++) {
    if(ops[i] == QUANTILE) { 
      quantile_args* q_args = static_cast<quantile_args*>(args[i]); 
      if (q_args == nullptr or q_args->quantiles.size() == 0) {
        CUDF_FAIL(
                "Cannot compute QUANTILE aggregation. It requires quantiles argument.");
      }
    } 
  }
}

std::pair<cudf::table, std::vector<gdf_column*>> groupby(cudf::table const& keys,
                                            cudf::table const& values,
                                            std::vector<operation> const& ops,
                                            Options options,
                                            cudaStream_t stream) {
  CUDF_EXPECTS(keys.num_rows() == values.num_rows(),
               "Size mismatch between number of rows in keys and values.");
  std::vector<operators> optype_list(ops.size());
  std::transform(ops.begin(), ops.end(), optype_list.begin(), [](auto const& op) {
    return op.op_name;
  });
  std::vector<operation_args*> ops_args(ops.size());
  std::transform(ops.begin(), ops.end(), ops_args.begin(), [](auto const& op) {
    return op.args.get();
  });
  verify_operators(values, optype_list);
  verify_operators_with_arguments(optype_list, ops_args);

  // Empty inputs
  if (keys.num_rows() == 0) {
    cudf::table output_values(0, target_dtypes(column_dtypes(values), optype_list), column_dtype_infos(values));
    return std::make_pair(
        cudf::empty_like(keys),
        output_values.get_columns()
        );
  }

  auto compute_groupby = groupby_null_specialization(keys, values);

  cudf::table output_keys;
  std::vector<gdf_column*> output_values;
  std::tie(output_keys, output_values) =
      compute_groupby(keys, values, optype_list, ops_args, options, stream);
  
  cudf::table table_output_values(output_values);
  
  update_nvcategories(keys, output_keys, values, table_output_values);
  return std::make_pair(output_keys, output_values);                                              
}

} // namespace detail
 

std::pair<cudf::table, std::vector<gdf_column*>> groupby(cudf::table const &keys,
                                            cudf::table const &values,
                                            std::vector<operation> const &ops,
                                            Options options) {
  return detail::groupby(keys, values, ops, options);
}


} // END: namespace sort
} // END: namespace groupby
} // END: namespace cudf