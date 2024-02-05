#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <iostream>
#include <nanoflann.hpp>
#include <pthread.h>
#include "icp.h"

using namespace nanoflann;

// Eigen::Matrix4d ICP(Eigen::Matrixx<num_t, n_points,3> pointcloud2){

// }


int main() {
    using num_t = double;
    const int n_points = 5;
    const int max_leaf = 10;
    const int DIM = 3;
    typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<num_t,n_points,3> >  my_kd_tree_t;

    //Create point clouds
    Eigen::Matrix<num_t, n_points,3> pointcloud1 = Eigen::MatrixXd::Random(n_points,3);

    //Rotate by the above matrix, and add some random vector to the second pointcloud.
    
    Eigen::Matrix<num_t, 3,3> rotationalMatrix = Eigen::MatrixXd(3,3);
    rotationalMatrix << 0.73842711,  0.6743333 ,  0,
                        -0.6743333 ,  0.73842711,  0.,
                        0.        ,  0.        ,  1.;

    Eigen::Matrix<num_t, n_points,3> pointcloud2 = pointcloud1 * rotationalMatrix;
    Eigen::RowVector3d random_offset = Eigen::RowVector3d::Random();
    pointcloud2.rowwise() += random_offset;

    //Assume that we have a function defined at this point.

    //Get the centroids.
    Eigen::Vector3d centerOfMass1 = pointcloud1.colwise().mean();
    Eigen::Vector3d centerOfMass2 = pointcloud2.colwise().mean();

    //Translate the point clouds so their center of mass is at the origin.
    pointcloud1.rowwise() -= centerOfMass1.transpose();
    pointcloud2.rowwise() -= centerOfMass2.transpose();        

    //Create point clouds.
    my_kd_tree_t point_cloud_index(DIM, pointcloud1, max_leaf);
    point_cloud_index.index_->buildIndex();
    
    Eigen::MatrixXd mapped_points = pointcloud2; 
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
        //the nearest neighbor is gonna be in the ith index <-> ret_index so update pointcloud2 so that they match up properly
        mapped_points.row(i) = pointcloud2.row(ret_index);
    }

    //Now that everything has been centralized and mapped, multiply stuff out.
    Eigen::MatrixXd W = mapped_points.transpose() * pointcloud1;

    //Now do SVD.
    // Eigen::BDCSVD<Eigen::MatrixXd> svd(W,Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(W,  Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV(); 

    //Determinant part of algorithm.
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(U.rows(), V.rows());
    double det = (U * V.transpose()).determinant();
    if (det < 0) {
        I(I.rows() - 1, I.cols() - 1) = -1; // Set the bottom-right element to -1 if det is negative
    }

    Eigen::MatrixXd R = U * I * V.transpose();
    Eigen::Vector3d t = centerOfMass1 - R * centerOfMass2;

    std::cout << "Rotational Matrix:" << std::endl << R << std::endl;
    std::cout << "Offset:" << std::endl << t << std::endl;

    Eigen::MatrixXd pose(4,4);
    pose << R, t, 0,0,0,1;
    
    std::cout << "Result: " << pose << std::endl;
}
