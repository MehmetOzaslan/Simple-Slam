#pragma once
#include <vector>
#include <Eigen/Dense>


namespace ICP{
    Eigen::MatrixXd GetPose(std::vector<Eigen::Vector3<double>> point_cloud_1,
        std::vector<Eigen::Vector3<double>> point_cloud_2);
}