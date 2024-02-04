#include <Eigen/Dense>
#include <iostream>
#include <nanoflann.hpp>
#include <pthread.h>
#include "icp.h"

using namespace nanoflann;

int main() {
    using num_t = double;
    const int n_points = 1000;
    const int max_leaf = 10;
    const int DIM = 3;
    typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<num_t,n_points,3> >  my_kd_tree_t;

    //Create point clouds
    Eigen::Matrix<num_t, n_points,3> pointcloud1 = Eigen::MatrixXd::Random(n_points,3);
    Eigen::Matrix<num_t, 3,3> rotationalMatrix = Eigen::MatrixXd(3,3);
    rotationalMatrix << 0.73842711,  0.6743333 ,  0,
                        -0.6743333 ,  0.73842711,  0.,
                        0.        ,  0.        ,  1.;

    //Rotate by the above matrix.
    Eigen::Matrix<num_t, n_points,3> pointcloud2 = pointcloud1 * rotationalMatrix;
    Eigen::Vector3d centerOfMass1 = pointcloud1.colwise().mean();
    Eigen::Vector3d centerOfMass2 = pointcloud2.colwise().mean();

    //Create point clouds.
    my_kd_tree_t point_cloud_index(DIM, pointcloud1, max_leaf);

    point_cloud_index.index_->buildIndex();
    

    for (int i = 0; i < pointcloud2.rows(); ++i) {
        // Query point
        num_t query_pt[DIM] = { pointcloud2(i,0), pointcloud2(i,1), pointcloud2(i,2) };

        // Results
        size_t ret_index;
        num_t out_dist_sqr;
        nanoflann::KNNResultSet<num_t> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);

        // Perform the search
        point_cloud_index.index_->findNeighbors(resultSet, &query_pt[0]);

        // Output result (for example purposes, adapt as needed)
        std::cout << "Nearest neighbor of point " << i << " is point " << ret_index
                << " with squared L2 distance " << out_dist_sqr << std::endl;
    }

}
