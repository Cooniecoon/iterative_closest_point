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
        lm_param = {0.1,true,10.0,0.1,1e-9};
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
        double prev_err = std::numeric_limits<double>::max();
        auto xyz = xyz_;
        auto rpy = rpy_;
        for(std::size_t i=0; i<max_iteration_; i++) {
            double curr_err(0.0);
            auto tf = getTransformation(xyz, rpy);
            pcl::transformPointCloud(*source_cloud_, *transformed_cloud, tf);
            Correspondences correspondences = getCorrespondenceIndices(transformed_cloud, target_cloud_);
            H_ = Hessian::Zero();
            g_ = Eigen::Matrix<double, 6, 1>::Zero();
            chi_ = 0.0;
            Jacobian J;
            for(int corr_idx = 0; corr_idx < correspondences.size(); corr_idx++) {
                int i = correspondences[corr_idx].first;
                int j = correspondences[corr_idx].second;
                PointT p_src = source_cloud_->points[i];
                PointT p_tgt = target_cloud_->points[j];

                Eigen::Vector3d err = getError(xyz, rpy, p_src, p_tgt);
                double weight = (err.norm() > euclidian_threshold_) ? 0.0 : 1.0;
                // double weight(1.0);
                J = getJacobian(rpy_, p_src);
                H_ += weight*(J.transpose() * J);
                g_ += weight*(J.transpose() * err);
                chi_ += weight* err.transpose() * err;
                curr_err+=weight*err.norm();
                // std::cout <<err.transpose() * err << ", ";
            }
            chi_/=correspondences.size();
            Hessian H_diag = H_.diagonal().matrix().asDiagonal();
            auto dparam = -(H_+lm_param.lambda*H_diag).inverse()*g_;

            rpy(0)+=dparam(0);
            rpy(1)+=dparam(1);
            rpy(2)+=dparam(2);
            xyz(0)+=dparam(3);
            xyz(1)+=dparam(4);
            xyz(2)+=dparam(5);
            if (prev_err>chi_) {
                prev_err = chi_; 
                lm_param.lambda = lm_param.lambda/lm_param.decrease_factor;
                rpy_=rpy;
                xyz_=xyz;

            } else {
                lm_param.lambda = lm_param.lambda/lm_param.increase_factor;
            }

            std::cout << "iter: " << i << " err: " << chi_ << " lambda: " << lm_param.lambda << std::endl;
            if(chi_ < lm_param.tollerance) {
                std::cout << "converged!" << std::endl;
            }
        }
        std::cout << "xyz";
        std::cout << xyz_.transpose() << std::endl;
        std::cout << "rpy";
        std::cout << rpy_.transpose() << std::endl;

    }

    Eigen::Matrix4d getTransformation(const Eigen::Vector3d &xyz, const Eigen::Vector3d &rpy)
    {
        Eigen::Matrix3d rotation;
        rotation = Eigen::AngleAxisd(rpy(2), Eigen::Vector3d::UnitZ())
                  * Eigen::AngleAxisd(rpy(1), Eigen::Vector3d::UnitY())
                  * Eigen::AngleAxisd(rpy(0), Eigen::Vector3d::UnitX());
        Eigen::Translation3d translation (xyz(0),xyz(1),xyz(2));
        Eigen::Matrix4d tf = (translation * rotation).matrix ();
        return tf;
    }

    Eigen::Matrix4d getFinalTransformation()
    {
        return getTransformation(xyz_, rpy_);
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

    struct LM_param_
    {
        double lambda;
        bool update_j;
        double increase_factor;
        double decrease_factor;
        double tollerance;
    };

    LM_param_ lm_param;
    Eigen::Matrix3d  getRotationMatrix(const Eigen::Vector3d &rpy)
    {
        using std::cos, std::sin;

        Eigen::Matrix3d rotationMatrix;
        rotationMatrix << cos(rpy(1))*cos(rpy(2)), sin(rpy(1))*sin(rpy(0))*cos(rpy(2)) - sin(rpy(2))*cos(rpy(0)), sin(rpy(1))*cos(rpy(0))*cos(rpy(2)) + sin(rpy(0))*sin(rpy(2)), 
                          sin(rpy(2))*cos(rpy(1)), sin(rpy(1))*sin(rpy(0))*sin(rpy(2)) + cos(rpy(0))*cos(rpy(2)), sin(rpy(1))*sin(rpy(2))*cos(rpy(0)) - sin(rpy(0))*cos(rpy(2)), 
                          -sin(rpy(1)), sin(rpy(0))*cos(rpy(1)), cos(rpy(1))*cos(rpy(0));

        return rotationMatrix;
    }

    Jacobian getJacobian(const Eigen::Vector3d &rpy, const PointT &point)
    {
        using std::cos, std::sin;
        double x(point.x), y(point.y), z(point.z);
        Jacobian J;
        J   << y*(sin(rpy(1))*cos(rpy(0))*cos(rpy(2)) + sin(rpy(0))*sin(rpy(2))) + z*(-sin(rpy(1))*sin(rpy(0))*cos(rpy(2)) + sin(rpy(2))*cos(rpy(0))), -x*sin(rpy(1))*cos(rpy(2)) + y*sin(rpy(0))*cos(rpy(1))*cos(rpy(2)) + z*cos(rpy(1))*cos(rpy(0))*cos(rpy(2)), -x*sin(rpy(2))*cos(rpy(1)) + y*(-sin(rpy(1))*sin(rpy(0))*sin(rpy(2)) - cos(rpy(0))*cos(rpy(2))) + z*(-sin(rpy(1))*sin(rpy(2))*cos(rpy(0)) + sin(rpy(0))*cos(rpy(2))), 1.0, 0.0, 0.0,
               y*(sin(rpy(1))*sin(rpy(2))*cos(rpy(0)) - sin(rpy(0))*cos(rpy(2))) + z*(-sin(rpy(1))*sin(rpy(0))*sin(rpy(2)) - cos(rpy(0))*cos(rpy(2))), -x*sin(rpy(1))*sin(rpy(2)) + y*sin(rpy(0))*sin(rpy(2))*cos(rpy(1)) + z*sin(rpy(2))*cos(rpy(1))*cos(rpy(0)), x*cos(rpy(1))*cos(rpy(2)) + y*(sin(rpy(1))*sin(rpy(0))*cos(rpy(2)) - sin(rpy(2))*cos(rpy(0))) + z*(sin(rpy(1))*cos(rpy(0))*cos(rpy(2)) + sin(rpy(0))*sin(rpy(2))), 0.0, 1.0, 0.0,
               y*cos(rpy(1))*cos(rpy(0)) - z*sin(rpy(0))*cos(rpy(1)), -x*cos(rpy(1)) - y*sin(rpy(1))*sin(rpy(0)) - z*sin(rpy(1))*cos(rpy(0)), 0.0, 0.0, 0.0, 1.0;
        return J;
    }

    Eigen::Vector3d getError(const Eigen::Vector3d &xyz, const Eigen::Vector3d &rpy, const PointT &src_point, const PointT &tgt_point)
    {
        Eigen::Vector3d p_src, p_tgt;
        p_src<<src_point.x,src_point.y,src_point.z;
        p_tgt<<tgt_point.x,tgt_point.y,tgt_point.z;
        Eigen::Matrix3d R = getRotationMatrix(rpy);
        return (R*p_src + xyz) - p_tgt;
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