#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <iostream>
#include <nanoflann.hpp>
#include <pthread.h>
#include "icp.h"

using namespace nanoflann;


// const Eigen::Matrix<num_t, n_points, 3>& x, const Eigen::Matrix<num_t, n_points, 3>& p

template <typename num_t, int n_points>
Eigen::Matrix<num_t,n_points,3> CorrespondPoints(Eigen::Matrix<num_t,n_points,3> x, Eigen::Matrix<num_t,n_points,3> p){

    typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<num_t,n_points,3> >  my_kd_tree_t;
    const int DIM = 3;
    const int max_leaf = 10;


    //Create point clouds.
    my_kd_tree_t point_cloud_index(DIM, x, max_leaf);
    point_cloud_index.index_->buildIndex();
    
    Eigen::MatrixXd mapped_points = p; 
    for (int i = 0; i < p.rows(); ++i) {
        // Query point
        num_t query_pt[DIM] = { p(i,0), p(i,1), p(i,2) };

        // Results
        size_t ret_index;
        num_t out_dist_sqr;
        nanoflann::KNNResultSet<num_t> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);

        // Perform the search
        point_cloud_index.index_->findNeighbors(resultSet, &query_pt[0]);
        //the nearest neighbor is gonna be in the ith index <-> ret_index so update p so that they match up properly
        mapped_points.row(i) = p.row(ret_index);
    }

    return mapped_points;
}


template <typename num_t, int n_points>
Eigen::Matrix4d ObtainPose(Eigen::Matrix<num_t,n_points,3> x, Eigen::Matrix<num_t,n_points,3> p){

    //Get the centroids.
    Eigen::Vector3d centerOfMass1 = x.colwise().mean();
    Eigen::Vector3d centerOfMass2 = p.colwise().mean();

    //Translate the point clouds so their center of mass is at the origin.
    x.rowwise() -= centerOfMass1.transpose();
    p.rowwise() -= centerOfMass2.transpose();        
    
    Eigen::Matrix<num_t,n_points,3> mapped_points = CorrespondPoints(x,p);

    //Now that everything has been centralized and mapped, multiply stuff out.
    Eigen::MatrixXd W = mapped_points.transpose() * x;

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

    Eigen::MatrixXd pose(4,4);
    pose << R, t, 0,0,0,1;
    return pose;
}


int main() {
    using num_t = double;
    const int n_points = 5;

    //Create point clouds
    Eigen::Matrix<num_t, n_points,3> x = Eigen::MatrixXd::Random(n_points,3);

    //Rotate by the matrix, and add some random vector to the second pointcloud.
    Eigen::Matrix<num_t, 3,3> R = Eigen::MatrixXd(3,3);
    R << 0.73842711,  0.6743333 ,  0,
                        -0.6743333 ,  0.73842711,  0.,
                        0.        ,  0.        ,  1.;

    Eigen::Matrix<num_t, n_points,3> p = x * R;
    Eigen::RowVector3d t = Eigen::RowVector3d::Random();
    p.rowwise() += t;

    Eigen::Matrix4d original_translation(4,4);
    original_translation << R, t.transpose(), 0,0,0,1;


    

    Eigen::Matrix4d pose = ObtainPose(p, x);

    std::cout << "Original: " << std::endl <<  std::endl <<  original_translation << std::endl;
    std::cout << "Result: " << std::endl << pose << std::endl;

}
