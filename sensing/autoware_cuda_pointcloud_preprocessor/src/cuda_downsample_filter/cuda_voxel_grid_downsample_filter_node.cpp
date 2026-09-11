// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/cuda_pointcloud_preprocessor/cuda_downsample_filter/cuda_voxel_grid_downsample_filter_node.hpp"

#include "autoware/pointcloud_preprocessor/utility/memory.hpp"

namespace autoware::cuda_pointcloud_preprocessor
{
CudaVoxelGridDownsampleFilterNode::CudaVoxelGridDownsampleFilterNode(
  const rclcpp::NodeOptions & node_options)
: Node("cuda_voxel_grid_downsample_filter", node_options)
{
  // set initial parameters
  float voxel_size_x = declare_parameter<float>("voxel_size_x");
  float voxel_size_y = declare_parameter<float>("voxel_size_y");
  float voxel_size_z = declare_parameter<float>("voxel_size_z");
  int64_t max_mem_pool_size_in_byte = declare_parameter<int64_t>(
    "max_mem_pool_size_in_byte",
    1e9);  // 1GB in default
  if (max_mem_pool_size_in_byte < 0) {
    RCLCPP_ERROR(
      this->get_logger(), "Invalid pool size was specified. The value should be positive");
    return;
  }

  sub_ =
    std::make_shared<cuda_blackboard::CudaBlackboardSubscriber<cuda_blackboard::CudaPointCloud2>>(
      *this, "~/input/pointcloud",
      std::bind(
        &CudaVoxelGridDownsampleFilterNode::cudaPointcloudCallback, this, std::placeholders::_1));

  pub_ =
    std::make_unique<cuda_blackboard::CudaBlackboardPublisher<cuda_blackboard::CudaPointCloud2>>(
      *this, "~/output/pointcloud");

  cuda_voxel_grid_downsample_filter_ = std::make_unique<CudaVoxelGridDownsampleFilter>(
    voxel_size_x, voxel_size_y, voxel_size_z, max_mem_pool_size_in_byte);
}

void CudaVoxelGridDownsampleFilterNode::cudaPointcloudCallback(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr msg)
{
  // Accept any of the layouts this package produces and consumes, not only
  // PointXYZI.
  //
  // is_data_layout_compatible_with_point_xyzi requires intensity to be FLOAT32,
  // while PointXYZIRC and PointXYZIRCAEDT -- the layouts every other node in
  // this package publishes -- store it as UINT8. Checking xyzi alone therefore
  // warned on every cloud of a correctly configured pipeline: a 30-second
  // replay produced over 1500 of these, each saying the result may be wrong
  // about output that was right. A warning that is always on is a warning
  // nobody reads.
  //
  // The filter itself does not need any of these layouts. It resolves x, y, z
  // and intensity by field name at runtime (see VoxelInfo::input_xyzi_offset)
  // and treats return_type and channel as optional, throwing only when a
  // mandatory field is absent. The check is therefore advisory, and the
  // throttle below keeps a genuinely unknown layout visible without flooding.
  const auto & fields = msg->fields;
  const bool layout_is_known =
    pointcloud_preprocessor::utils::is_data_layout_compatible_with_point_xyzi(fields) ||
    pointcloud_preprocessor::utils::is_data_layout_compatible_with_point_xyzirc(fields) ||
    pointcloud_preprocessor::utils::is_data_layout_compatible_with_point_xyzircaedt(fields);
  if (!layout_is_known) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 10000,
      "Input pointcloud layout matches none of PointXYZI, PointXYZIRC or "
      "PointXYZIRCAEDT. The filter resolves x, y, z and intensity by field name, "
      "so this is only a problem if one of them is missing or is not FLOAT32 "
      "for x, y and z.");
  }

  auto output_pointcloud_ptr = cuda_voxel_grid_downsample_filter_->filter(msg);
  pub_->publish(std::move(output_pointcloud_ptr));
}
}  // namespace autoware::cuda_pointcloud_preprocessor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
  autoware::cuda_pointcloud_preprocessor::CudaVoxelGridDownsampleFilterNode)
