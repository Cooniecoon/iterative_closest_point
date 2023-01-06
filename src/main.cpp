#include "iterative_closest_point/iterative_closest_point.hpp"



#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>

#include "pcl/sample_consensus/method_types.h"
#include "pcl/sample_consensus/model_types.h"
#include "pcl/segmentation/sac_segmentation.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/filters/passthrough.h"


typedef pcl::PointXYZ PointType;

pcl::SACSegmentation<PointType> seg_;
pcl::ExtractIndices<PointType> extract_;
void setSegmentationParam()
{
    seg_.setOptimizeCoefficients(true);
    seg_.setModelType(pcl::SACMODEL_PLANE);
    seg_.setMethodType(pcl::SAC_RANSAC);
    seg_.setMaxIterations(1000);
    seg_.setDistanceThreshold(0.05);
    seg_.setRadiusLimits(0, 0.01);
    extract_.setNegative(true);
}

void groundSegmentation(pcl::PointCloud<PointType>::Ptr &input_cloud ,pcl::PointCloud<PointType>::Ptr &output_cloud)
{
    pcl::ModelCoefficients::Ptr coefficients_(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_(new pcl::PointIndices);
    seg_.setInputCloud(input_cloud);
    seg_.segment(*inliers_, *coefficients_);

    extract_.setInputCloud(input_cloud);
    extract_.setIndices(inliers_);
    extract_.filter(*output_cloud);
}


void voxelize(pcl::PointCloud<PointType>::Ptr pc_src, pcl::PointCloud<PointType>& pc_dst, double var_voxel_size)
{
    pcl::VoxelGrid<PointType> voxel_filter_;
    voxel_filter_.setInputCloud(pc_src);
    voxel_filter_.setLeafSize(var_voxel_size, var_voxel_size,var_voxel_size);
    voxel_filter_.filter(pc_dst);
}

void roiFilter(pcl::PointCloud<PointType>::Ptr &input_cloud,pcl::PointCloud<PointType>::Ptr &output_cloud, std::string field_name, float min_val, float max_val)
{
    pcl::PassThrough<PointType> roi_filter_;   

    roi_filter_.setInputCloud(input_cloud);
    roi_filter_.setFilterFieldName (field_name);
    roi_filter_.setFilterLimits (min_val, max_val);
    roi_filter_.filter (*output_cloud);
}

int main()
{
    setSegmentationParam();

    std::string file_2 = "/home/tony/iterative_closest_point/data/front_rear/rear_-179.596790.pcd";
    std::string file_1 = "/home/tony/iterative_closest_point/data/front_rear/rear_-185.324076.pcd";

    pcl::PointCloud<PointType>::Ptr source_cloud(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr target_cloud(new pcl::PointCloud<PointType>);

    pcl::io::loadPCDFile<PointType> (file_1, *source_cloud);
    pcl::io::loadPCDFile<PointType> (file_2, *target_cloud);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*source_cloud,*source_cloud, indices); 
    pcl::removeNaNFromPointCloud(*target_cloud,*target_cloud, indices); 

    voxelize(source_cloud, *source_cloud,0.03);
    voxelize(target_cloud, *target_cloud,0.03);


    groundSegmentation(source_cloud,source_cloud);
    groundSegmentation(target_cloud,target_cloud);
    
    roiFilter(source_cloud, source_cloud, "z", 0.0, 2.0);
    roiFilter(target_cloud, target_cloud, "z", 0.0, 2.0);

    iterative_closest_point::ICP icp;
    icp.setSourceData(source_cloud);
    icp.setTargetData(target_cloud);
    icp.setEuclidianThreshold(0.08);
    icp.setMaxIteration(100);

    Eigen::Vector3d xyz;
    Eigen::Vector3d rpy;
    // xyz << -0.04106184, 0.00213617, 0.0;
    // rpy << 0.00239408, -0.09553752, -0.07596487;
    // icp.setInitialTransformation(xyz, rpy);


    icp.computeTransformation();
    Eigen::Matrix4d tf = icp.getFinalTransformation();
    pcl::PointCloud<PointType>::Ptr transformed_cloud (new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*source_cloud, *transformed_cloud, tf);


    // Initializing point cloud visualizer
    pcl::visualization::PCLVisualizer::Ptr
    viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer_final->setBackgroundColor (0, 0, 0);

    // Coloring and visualizing target cloud (red).
    pcl::visualization::PointCloudColorHandlerCustom<PointType>
    transformed_color (transformed_cloud, 255, 255, 0);
    viewer_final->addPointCloud<PointType> (transformed_cloud, transformed_color, "transformed cloud");
    viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    2, "transformed cloud");

    pcl::visualization::PointCloudColorHandlerCustom<PointType>
    input_color (source_cloud, 255, 0, 0);
    viewer_final->addPointCloud<PointType> (source_cloud, input_color, "input cloud");
    viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    2, "input cloud");

    pcl::visualization::PointCloudColorHandlerCustom<PointType>
    target_color (target_cloud, 0, 0, 255);
    viewer_final->addPointCloud<PointType> (target_cloud, target_color, "target cloud");
    viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    2, "target cloud");

    // Starting visualizer
    viewer_final->addCoordinateSystem (1.0, "global");
    viewer_final->initCameraParameters ();

    // Wait until visualizer window is closed.
    while (!viewer_final->wasStopped ())
    {
    viewer_final->spinOnce (100000000);
    }

    return (0);
}