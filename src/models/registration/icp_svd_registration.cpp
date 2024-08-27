/*
 * @Description: ICP SVD lidar odometry
 * @Author: Ge Yao
 * @Date: 2020-10-24 21:46:45
 */
#include <ros/ros.h>

#include <pcl/common/transforms.h>
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "glog/logging.h"

#include "lidar_localization/models/registration/icp_svd_registration.hpp"

namespace lidar_localization {

ICPSVDRegistration::ICPSVDRegistration(
    const YAML::Node& node
) : input_target_kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZ>()) {
    // parse params:
    float max_corr_dist = node["max_corr_dist"].as<float>();
    float trans_eps = node["trans_eps"].as<float>();
    float euc_fitness_eps = node["euc_fitness_eps"].as<float>();
    int max_iter = node["max_iter"].as<int>();

    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

ICPSVDRegistration::ICPSVDRegistration(
    float max_corr_dist, 
    float trans_eps, 
    float euc_fitness_eps, 
    int max_iter
) : input_target_kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZ>()) {
    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

bool ICPSVDRegistration::SetRegistrationParam(
    float max_corr_dist, 
    float trans_eps, 
    float euc_fitness_eps, 
    int max_iter
) {
    // set params:
    max_corr_dist_ = max_corr_dist;
    trans_eps_ = trans_eps;
    euc_fitness_eps_ = euc_fitness_eps;
    max_iter_ = max_iter;

    LOG(INFO) << "ICP SVD params:" << std::endl
              << "max_corr_dist: " << max_corr_dist_ << ", "
              << "trans_eps: " << trans_eps_ << ", "
              << "euc_fitness_eps: " << euc_fitness_eps_ << ", "
              << "max_iter: " << max_iter_ 
              << std::endl << std::endl;

    return true;
}

bool ICPSVDRegistration::SetInputTarget(const CloudData::CLOUD_PTR& input_target) {
    input_target_ = input_target;
    input_target_kdtree_->setInputCloud(input_target_);

    return true;
}

bool ICPSVDRegistration::ScanMatch(
    const CloudData::CLOUD_PTR& input_source, 
    const Eigen::Matrix4f& predict_pose, 
    CloudData::CLOUD_PTR& result_cloud_ptr,
    Eigen::Matrix4f& result_pose
) {
    input_source_ = input_source;

    // pre-process input source:
    CloudData::CLOUD_PTR transformed_input_source(new CloudData::CLOUD());
    pcl::transformPointCloud(*input_source_, *transformed_input_source, predict_pose);//应用初始位姿旋转之后的点云

    // init estimation:
    transformation_.setIdentity();
    
    //
    // TODO: first option -- implement all computing logic on your own
    //
    // do estimation:
    int curr_iter = 0;
    while (curr_iter < max_iter_) {
        // TODO: apply current estimation:
        ROS_INFO("apply current estimation");
        CloudData::CLOUD_PTR input_pointcloud_after_iter_estimate(new CloudData::CLOUD());
        pcl::transformPointCloud(*transformed_input_source,*input_pointcloud_after_iter_estimate,transformation_);

        // TODO: get correspondence:
        std::vector<Eigen::Vector3f> xs;
        std::vector<Eigen::Vector3f> ys;
        ROS_INFO("get correspondence");
        // TODO: do not have enough correspondence -- break:
        if(GetCorrespondence(input_pointcloud_after_iter_estimate,xs,ys)<3)
            break;  
        // TODO: update current transform:
        Eigen::Matrix4f delta_transform;
        GetTransform(xs,ys,delta_transform);
        ROS_INFO("update current transform");
        // TODO: whether the transformation update is significant:
        if(!IsSignificant(delta_transform,trans_eps_))
        {
            break;
        }
        // TODO: update transformation:
        transformation_ = delta_transform*transformation_;

        ++curr_iter;
    }

    // set output:
    result_pose = transformation_ * predict_pose;

    //归一化
    Eigen::Quaternionf qr(result_pose.block<3,3>(0,0));
    qr.normalize();//对四元数进行归一化操作
    Eigen::Vector3f t = result_pose.block<3,1>(0,3);
    result_pose.setIdentity();
    result_pose.block<3,3>(0,0) = qr.toRotationMatrix();
    result_pose.block<3,1>(0,3) = t;
    pcl::transformPointCloud(*input_source_, *result_cloud_ptr, result_pose);
    
    return true;
}

size_t ICPSVDRegistration::GetCorrespondence(
    const CloudData::CLOUD_PTR &input_source, 
    std::vector<Eigen::Vector3f> &xs,
    std::vector<Eigen::Vector3f> &ys
) {
    const float MAX_CORR_DIST_SQR = max_corr_dist_ * max_corr_dist_;

    size_t num_corr = 0;

    // TODO: set up point correspondence
    for (size_t i = 0; i < input_source->points.size(); ++i)
    {
        std::vector<int> corr_ind;
        std::vector<float> corr_sq_dis;
        input_target_kdtree_->nearestKSearch(//input_target_kdtree_是地图已经输入构建的kdtree了
            input_source->at(i),
            1,corr_ind,corr_sq_dis  
        );
        if(corr_sq_dis.at(0)>MAX_CORR_DIST_SQR)
        {
            continue;
        }
        ROS_INFO("156");
        Eigen::Vector3f x(
                    input_target_->at(corr_ind.at(0)).x,
                    input_target_->at(corr_ind.at(0)).y,
                    input_target_->at(corr_ind.at(0)).z
        );
        ROS_INFO("162");
        Eigen::Vector3f y(
                    input_source->at(i).x,
                    input_source->at(i).y,
                    input_source->at(i).z
                 
        );
        ROS_INFO("169");  
        xs.push_back(x);
        ys.push_back(y);
        ROS_INFO("172");
        ++num_corr;
    }
    
    ROS_INFO("175");

    return num_corr;
}

void ICPSVDRegistration::GetTransform(
    
    const std::vector<Eigen::Vector3f> &xs,
    const std::vector<Eigen::Vector3f> &ys,
    Eigen::Matrix4f &transformation_
) {
    const size_t N = xs.size();

    // TODO: find centroids of mu_x and mu_y:
    Eigen::Vector3f mu_x = Eigen::Vector3f::Zero();
    Eigen::Vector3f mu_y = Eigen::Vector3f::Zero();

    for (size_t i = 0; i < xs.size(); i++)
    {
        mu_x +=xs.at(i);
        mu_y +=ys.at(i);
    }
    mu_x /=N;
    mu_y /=N;
    ROS_INFO("find centroids of mu_x and mu_y");
    // TODO: build H:
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    for (size_t i = 0; i < N; i++)
    {
        H +=(ys.at(i)-mu_y) * (xs.at(i)-mu_x).transpose();
    }
    
    // TODO: solve R:
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(H,Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
    ROS_INFO("solve R");
    // TODO: solve t:
    Eigen::Vector3f t = mu_x-R*mu_y;
    // TODO: set output:
    transformation_.setIdentity();
    transformation_.block<3,3>(0,0) = R;
    transformation_.block<3,1>(0,3) = t;
    ROS_INFO("set output");
}

bool ICPSVDRegistration::IsSignificant(
    const Eigen::Matrix4f &transformation,
    const float trans_eps
) {
    // a. translation magnitude -- norm:
    float translation_magnitude = transformation.block<3, 1>(0, 3).norm();
    // b. rotation magnitude -- angle:
    float rotation_magnitude = fabs(
        acos(
            (transformation.block<3, 3>(0, 0).trace() - 1.0f) / 2.0f
        )
    );

    return (
        (translation_magnitude > trans_eps) || 
        (rotation_magnitude > trans_eps)
    );
}

} // namespace lidar_localization