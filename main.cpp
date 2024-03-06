#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <iostream>
#include <pthread.h>
#include "include/icp.h"
#include "icp.h"
#include <icp.h>
#include <nanoflann.hpp>
using namespace nanoflann;

int main() {
    using num_t = double;
    const int n_points = 5;

    //Create point clouds
    Eigen::Matrix<num_t, n_points,3> x = Eigen::MatrixXd::Random(n_points,3);

    //Rotate by the matrix, and add some random vector to the second pointcloud.
    Eigen::Matrix<num_t, 3,3> R = Eigen::MatrixXd(3,3);
    R << 0.996, -0.087,  0,
        0.087 ,  0.996,  0.,
         0.   ,  0.   ,  1;

    Eigen::Matrix<num_t, n_points,3> p = x * R;
    Eigen::RowVector3d t = Eigen::RowVector3d::Random();
    p.rowwise() += t;

    Eigen::Matrix4d original_translation(4,4);
    original_translation << R, t.transpose(), 0,0,0,1;

    auto pose = ObtainPose(x, p, 1);

    std::cout << "Original: " << std::endl <<  original_translation << std::endl;
    std::cout << "Result: " << std::endl << pose << std::endl;

}
