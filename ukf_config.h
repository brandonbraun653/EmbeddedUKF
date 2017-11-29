#pragma once
#ifndef UKF_CONFIG_H_
#define UKF_CONFIG_H_

/* C++ Dependencies */
#include <stdlib.h>

/* Eigen Dependencies */
#include <eigen/Eigen>

/* Define a few variables that will set up the UKF matrix dimensions */
#define NUM_STATE_VARS 9
#define NUM_SIGMA_POINTS (2*NUM_STATE_VARS + 1)

/* Vectors & Matrices have to be defined at compile time otherwise it's impossible
 * to view their contents in the debugger. Dynamic matrices are nice, but a nuisance
 * in this case for debugging */
typedef Eigen::Matrix<float, NUM_STATE_VARS, 1> mfColumn;
typedef Eigen::Matrix<float, NUM_STATE_VARS, NUM_STATE_VARS> mfSquare;
typedef Eigen::Matrix<float, NUM_STATE_VARS, NUM_SIGMA_POINTS> mfSigma;
typedef Eigen::Matrix<float, 1, NUM_SIGMA_POINTS> mfWeight;
typedef Eigen::DiagonalMatrix<float, NUM_STATE_VARS, NUM_STATE_VARS> mfDiagonal;

typedef Eigen::Matrix<double, NUM_STATE_VARS, 1> mdColumn;
typedef Eigen::Matrix<double, NUM_STATE_VARS, NUM_STATE_VARS> mdSquare;
typedef Eigen::Matrix<double, NUM_STATE_VARS, NUM_SIGMA_POINTS> mdSigma;
typedef Eigen::Matrix<double, 1, NUM_SIGMA_POINTS> mdWeight;
typedef Eigen::DiagonalMatrix<double, NUM_STATE_VARS, NUM_STATE_VARS> mdDiagonal;

#endif