#ifndef ITERATIVE_CLOSEST_POINT__ITERATIVE_CLOSEST_POINT_HPP__
#define ITERATIVE_CLOSEST_POINT__ITERATIVE_CLOSEST_POINT_HPP__

#include <iostream>


#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "pcl/common/transforms.h"

namespace iterative_closest_point
{
class ICP
{
public:
    typedef Eigen::Matrix<double, 3, 6> Jacobian;
    typedef Eigen::Matrix<double, 6, 6> Hessian;
    typedef std::vector<std::pair<int,int>> Correspondences;
    typedef pcl::PointXYZ PointT;

    ICP()
    {
        H_ = Hessian::Zero();
        g_ = Eigen::Matrix<double, 6, 1>::Zero();
        chi_ = 0.0;
        xyz_ =  Eigen::Vector3d::Zero();
        rpy_ =  Eigen::Vector3d::Zero();
    }

    void setEuclidianThreshold(const double &threshold)
    {
        euclidian_threshold_ = threshold;
    }

    void setMaxIteration(const int &i)
    {
        max_iteration_ = i;
    }

    void setSourceData(const pcl::PointCloud<PointT>::Ptr &source_cloud)
    {
        source_cloud_ = source_cloud;
        std::cout << "Loaded " << source_cloud_->size () << " points from source cloud" << std::endl;
    }

    void setTargetData(const pcl::PointCloud<PointT>::Ptr &target_cloud)
    {
        target_cloud_ = target_cloud;
        std::cout << "Loaded " << target_cloud_->size () << " points from target cloud" << std::endl;
    }

    void setInitialTransformation(const Eigen::Vector3d &xyz, const Eigen::Vector3d &rpy)
    {
        xyz_ = xyz;
        rpy_ = rpy;
    }



    void computeTransformation()
    {
        pcl::PointCloud<PointT>::Ptr transformed_cloud (new pcl::PointCloud<PointT>);
        *transformed_cloud = *source_cloud_;
        for(std::size_t i=0; i<max_iteration_; i++) {
            auto tf = getTransformation();
            Correspondences correspondences = getCorrespondenceIndices(transformed_cloud, target_cloud_);
            optimize(correspondences);
            auto dparam = (H_.transpose() * H_).ldlt().solve(H_.transpose() * -g_);
            rpy_(0)+=dparam(0);
            rpy_(1)+=dparam(1);
            rpy_(2)+=dparam(2);
            xyz_(0)+=dparam(3);
            xyz_(1)+=dparam(4);
            xyz_(2)+=dparam(5);
            pcl::transformPointCloud(*source_cloud_, *transformed_cloud, tf);
            std::cout << "iter: " << i << " err: " << chi_ << std::endl;
        }
        std::cout << "xyz";
        std::cout << xyz_.transpose() << std::endl;
        std::cout << "rpy";
        std::cout << rpy_.transpose() << std::endl;
    }

    Eigen::Matrix4d getTransformation()
    {
        Eigen::Matrix3d rotation;
        rotation = Eigen::AngleAxisd(rpy_(2), Eigen::Vector3d::UnitZ())
                  * Eigen::AngleAxisd(rpy_(1), Eigen::Vector3d::UnitY())
                  * Eigen::AngleAxisd(rpy_(0), Eigen::Vector3d::UnitX());
        Eigen::Translation3d translation (xyz_(0),xyz_(1),xyz_(2));
        Eigen::Matrix4d tf = (translation * rotation).matrix ();
        return tf;
    }

    void optimize(const Correspondences &correspondences)
    {
        H_ = Hessian::Zero();
        g_ = Eigen::Matrix<double, 6, 1>::Zero();
        chi_ = 0.0;
        double J_sum(0.0);
        double final_err;
        for(int corr_idx = 0; corr_idx < correspondences.size(); corr_idx++) {
            int i = correspondences[corr_idx].first;
            int j = correspondences[corr_idx].second;
            PointT p_src = source_cloud_->points[i];
            PointT p_tgt = target_cloud_->points[j];

            Eigen::Vector3d err = getError(p_src, p_tgt);
            double weight = (err.norm() > euclidian_threshold_) ? 0.0 : 1.0;
            // double weight(1.0);
            Jacobian J = getJacobian(p_src);
            J_sum+=J.norm();
            H_ += weight*(J.transpose() * J);
            g_ += weight*(J.transpose() * err);
            chi_ += weight* err.transpose() * err;
            final_err = err.norm();
            // std::cout <<err.transpose() * err << ", ";
        }

        std::cout << J_sum<<  std::endl;
    }
protected:

    pcl::PointCloud<PointT>::Ptr source_cloud_;
    pcl::PointCloud<PointT>::Ptr target_cloud_;
    Eigen::Vector3d xyz_;
    Eigen::Vector3d rpy_;

    int max_iteration_;
    double euclidian_threshold_;

    Hessian H_;
    Eigen::Matrix<double, 6, 1> g_;
    double chi_;
    Eigen::Matrix3d  getRotationMatrix()
    {
        using std::cos, std::sin;
        // Eigen::AngleAxisd rollAngle(rpy_(0), Eigen::Vector3d::UnitX());
        // Eigen::AngleAxisd pitchAngle(rpy_(1), Eigen::Vector3d::UnitY());
        // Eigen::AngleAxisd yawAngle(rpy_(2), Eigen::Vector3d::UnitZ());

        // Eigen::Matrix3d rotationMatrix = (yawAngle * pitchAngle * rollAngle).matrix();

        Eigen::Matrix3d rotationMatrix;
        rotationMatrix << cos(rpy_(1))*cos(rpy_(2)), sin(rpy_(1))*sin(rpy_(0))*cos(rpy_(2)) - sin(rpy_(2))*cos(rpy_(0)), sin(rpy_(1))*cos(rpy_(0))*cos(rpy_(2)) + sin(rpy_(0))*sin(rpy_(2)), 
                          sin(rpy_(2))*cos(rpy_(1)), sin(rpy_(1))*sin(rpy_(0))*sin(rpy_(2)) + cos(rpy_(0))*cos(rpy_(2)), sin(rpy_(1))*sin(rpy_(2))*cos(rpy_(0)) - sin(rpy_(0))*cos(rpy_(2)), 
                          -sin(rpy_(1)), sin(rpy_(0))*cos(rpy_(1)), cos(rpy_(1))*cos(rpy_(0));

        return rotationMatrix;
    }

    Jacobian getJacobian(const PointT &point)
    {
        using std::cos, std::sin;
        double x(point.x), y(point.y), z(point.z);
        Jacobian J;
        J   << y*(sin(rpy_(1))*cos(rpy_(0))*cos(rpy_(2)) + sin(rpy_(0))*sin(rpy_(2))) + z*(-sin(rpy_(1))*sin(rpy_(0))*cos(rpy_(2)) + sin(rpy_(2))*cos(rpy_(0))), -x*sin(rpy_(1))*cos(rpy_(2)) + y*sin(rpy_(0))*cos(rpy_(1))*cos(rpy_(2)) + z*cos(rpy_(1))*cos(rpy_(0))*cos(rpy_(2)), -x*sin(rpy_(2))*cos(rpy_(1)) + y*(-sin(rpy_(1))*sin(rpy_(0))*sin(rpy_(2)) - cos(rpy_(0))*cos(rpy_(2))) + z*(-sin(rpy_(1))*sin(rpy_(2))*cos(rpy_(0)) + sin(rpy_(0))*cos(rpy_(2))), 1.0, 0.0, 0.0,
               y*(sin(rpy_(1))*sin(rpy_(2))*cos(rpy_(0)) - sin(rpy_(0))*cos(rpy_(2))) + z*(-sin(rpy_(1))*sin(rpy_(0))*sin(rpy_(2)) - cos(rpy_(0))*cos(rpy_(2))), -x*sin(rpy_(1))*sin(rpy_(2)) + y*sin(rpy_(0))*sin(rpy_(2))*cos(rpy_(1)) + z*sin(rpy_(2))*cos(rpy_(1))*cos(rpy_(0)), x*cos(rpy_(1))*cos(rpy_(2)) + y*(sin(rpy_(1))*sin(rpy_(0))*cos(rpy_(2)) - sin(rpy_(2))*cos(rpy_(0))) + z*(sin(rpy_(1))*cos(rpy_(0))*cos(rpy_(2)) + sin(rpy_(0))*sin(rpy_(2))), 0.0, 1.0, 0.0,
               y*cos(rpy_(1))*cos(rpy_(0)) - z*sin(rpy_(0))*cos(rpy_(1)), -x*cos(rpy_(1)) - y*sin(rpy_(1))*sin(rpy_(0)) - z*sin(rpy_(1))*cos(rpy_(0)), 0.0, 0.0, 0.0, 1.0;
        return J;
    }

    Eigen::Vector3d getError(const PointT &src_point, const PointT &tgt_point)
    {
        Eigen::Vector3d p_src, p_tgt;
        p_src<<src_point.x,src_point.y,src_point.z;
        p_tgt<<tgt_point.x,tgt_point.y,tgt_point.z;
        Eigen::Matrix3d R = getRotationMatrix();
        return (R*p_src + xyz_) - p_tgt;
    }

    Correspondences getCorrespondenceIndices(const pcl::PointCloud<PointT>::Ptr &p, const pcl::PointCloud<PointT>::Ptr &q)
    {
        int p_size = p->size();
        int q_size = q->size();
        int size = (p_size>q_size)? q_size : p_size;
        Correspondences correspondences;

        for(int i=0; i<size; i++) {
            PointT p_pnt = p->points[i];
            double min_dist = std::numeric_limits<double>::max();
            int chosen_idx(-1);
            for(int j=0; j<size; j++) {
                PointT q_pnt = q->points[j];
                double dist = std::sqrt((p_pnt.x-q_pnt.x)*(p_pnt.x-q_pnt.x) + (p_pnt.y-q_pnt.y)*(p_pnt.y-q_pnt.y) + (p_pnt.z-q_pnt.z)*(p_pnt.z-q_pnt.z));
                // std::cout << dist << std::endl;
                if(dist<min_dist) {
                    min_dist = dist;
                    chosen_idx = j;
                }
            }
            correspondences.push_back(std::make_pair(i,chosen_idx));
        }
        return correspondences;
    }

};

} // namespace iterative_closest_point
#endif 