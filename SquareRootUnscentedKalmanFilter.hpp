#pragma once
#ifndef SQUARE_ROOT_UNSCENTED_KALMAN_FILTER_HPP
#define SQUARE_ROOT_UNSCENTED_KALMAN_FILTER_HPP

/* Note:
 * This implementation of the SR_UKF is not entirely my original work. Most of the
 * code has been forked from mherb on GitHub, linked here: https://github.com/mherb/kalman 
 * 
 * Modifications to the code have been primarily to condense everything down into 
 * a single managable file that works well on an embedded platform (aka no malloc).
 * Use of the original library found a few small bugs that prevented use on an embedded system.
 **/



/* C++ Dependencies */
#include <stdlib.h>
#include <memory>
#include <string>

/* Eigen Dependencies */
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#include <eigen/Eigen>

/* Boost Dependencies */



namespace Kalman 
{
	/**
     * @brief Cholesky square root decomposition of a symmetric positive-definite matrix
     * @param _MatrixType The matrix type
     * @param _UpLo Square root form (Eigen::Lower or Eigen::Upper)
     */
	template<typename _MatrixType, int _UpLo = Eigen::Lower>
	class Cholesky : public Eigen::LLT< _MatrixType, _UpLo >
	{
	public:
		Cholesky() : Eigen::LLT< _MatrixType, _UpLo >() {}
        
		/**
			* @brief Construct cholesky square root decomposition from matrix
			* @param m The matrix to be decomposed
			*/
		Cholesky(const _MatrixType& m) : Eigen::LLT< _MatrixType, _UpLo >(m) {}
        
		/**
			* @brief Set decomposition to identity
			*/
		Cholesky& setIdentity()
		{
			this->m_matrix.setIdentity();
			this->m_isInitialized = true;
			return *this;
		}
        
		/**
			* @brief Check whether the decomposed matrix is the identity matrix
			*/
		bool isIdentity() const
		{
			eigen_assert(this->m_isInitialized && "LLT is not initialized.");
			return this->m_matrix.isIdentity();
		}
        
		/**
			* @brief Set lower triangular part of the decomposition
			* @param matrix The lower part stored in a full matrix
			*/
		template<typename Derived>
			Cholesky& setL(const Eigen::MatrixBase <Derived>& matrix)
			{
				this->m_matrix = matrix.template triangularView<Eigen::Lower>();
				this->m_isInitialized = true;
				return *this;
			}
        
		/**
			* @brief Set upper triangular part of the decomposition
			* @param matrix The upper part stored in a full matrix
			*/
		template<typename Derived>
			Cholesky& setU(const Eigen::MatrixBase <Derived>& matrix)
			{
				this->m_matrix = matrix.template triangularView<Eigen::Upper>().adjoint();
				this->m_isInitialized = true;
				return *this;
			}
	};
	
	
	
	/**
	 * @brief Square Root Unscented Kalman Filter
	 * 
	 * @param T Scalar type used for numerical calculations 
	 * @param StateType Eigen::Matrix type used for the state vector
	 */
	template<typename T, typename StateType>
	class SquareRootUKF
	{
	public:
		typedef Eigen::Matrix<T, StateType::RowsAtCompileTime, 1> ColumnMatrix;
		typedef Eigen::Matrix<T, StateType::RowsAtCompileTime, StateType::RowsAtCompileTime> SquareMatrix;
		typedef Eigen::Matrix<T, StateType::RowsAtCompileTime, (2*StateType::RowsAtCompileTime + 1)> SigmaPointMatrix;
		typedef Eigen::Matrix<T, (2*StateType::RowsAtCompileTime + 1), 1> SigmaWeightMatrix;
		
		SquareRootUKF(T _alpha = T(1), T _beta = T(2), T _kappa = T(0)):			
			alpha(_alpha), beta(_beta), kappa(_kappa)
		{
			/* Pre-compute all the weights */
			computeWeights();
			
			
			/* Default square root covariance */
			S.setIdentity();
			
			/* Default State Vector */
			x.setZero();	
		}
		
		/* Matricies to be assigned by the user */
		ColumnMatrix x;	/* Initial State Vector */
		SquareMatrix A;		/* State Transition Matrix */
		SquareMatrix H;		/* Measurement */
		SquareMatrix Rv;	/* Process Noise Covariance (Square Root Form) */
		SquareMatrix Rn;	/* Measurement Noise Covariance (Square Root Form) */
	
	private:
		/* Weight Parameters */
        T alpha;      //!< Scaling parameter for spread of sigma points (usually \f$ 1E-4 \leq \alpha \leq 1 \f$)
        T beta;       //!< Parameter for prior knowledge about the distribution (\f$ \beta = 2 \f$ is optimal for Gaussian)
        T kappa;      //!< Secondary scaling parameter (usually 0)
        T gamma;      //!< \f$ \gamma = \sqrt{L + \lambda} \f$ with \f$ L \f$ being the state dimensionality
        T lambda;     //!< \f$ \lambda = \alpha^2 ( L + \kappa ) - L\f$ with \f$ L \f$ being the state dimensionality
		
		SigmaWeightMatrix sigmaWeights_m;
		SigmaWeightMatrix sigmaWeights_c;
		
		
		/* Sigma Point Parameters */
		static constexpr int SigmaPointCount = 2 * StateType::RowsAtCompileTime + 1;
		
		SigmaPointMatrix sigmaStatePoints;
		SigmaPointMatrix sigmaMeasurementPoints;
		
		
		using CovarianceSquareRoot = Cholesky<SquareMatrix>;
		CovarianceSquareRoot S;
		
		/* Matrices */
		
		
	public:
		const StateType& iterate(StateType& measured_data)
		{
			/*---------------------------------
			 * Predict Step
			 *--------------------------------*/
			//Eq(17)
			computeSigmaPoints();
			
			//Eq(18): Pass the sigma points through the system model transition matrix (A)
			computeSigmaPointTransition();
			
			//Eq(19): Use the weights and sigma points to generate a state prediction
			x = computePredictionFromSigmaPoints(sigmaStatePoints);
			
			//Eq(20): Compute the prediction covariance
			if(!computeCovarianceSquareRootFromSigmaPoints(x, sigmaStatePoints, Rv, S))
			{
				//TODO: handle numerical errors
				assert(false);
			}
			
			//computeCovarianceSquareRootFromSigmaPoints(x, sigmaStatePoints, Rv, SS);
			
			/*---------------------------------
			 * Update Step
			 *--------------------------------*/
			
			
			return measured_data;
		}
		
	private:
		void computeWeights()
		{
			T L = T(StateType::RowsAtCompileTime);
			lambda = alpha * alpha * (L + kappa) - L;
			
			// Make sure L != -lambda to avoid division by zero
            assert(std::abs(L + lambda) > T(1e-6f));
            
			// Make sure L != -kappa to avoid division by zero
			assert(std::abs(L + kappa) > T(1e-6f));
			
			T W_m_0 = lambda / (L + lambda);
			T W_c_0 = W_m_0 + (T(1) - alpha*alpha + beta);
			T W_i   = T(1) / (T(2) * alpha*alpha * (L + kappa));
			
			// Make sure W_i > 0 to avoid square-root of negative number
            assert(W_i > T(0));
			
			sigmaWeights_m(0) = W_m_0;
			sigmaWeights_c(0) = W_c_0;
			
			for (int i = 1; i < SigmaPointCount; ++i)
			{
				sigmaWeights_m(i) = W_i;
				sigmaWeights_c(i) = W_i;
			}
		}
		
		/**
         * @brief Compute sigma points from current state estimate and state covariance
         * 
         * @note This covers equations (17) and (22) of Algorithm 3.1 in the Paper
         */
		bool computeSigmaPoints()
		{
			//Get the square root of covariance
			SquareMatrix S_ = S.matrixL().toDenseMatrix();
			
			//Set the first column
			sigmaStatePoints.template leftCols<1>() = x;
			
			//Set the center block with (x + gamma * S)
			sigmaStatePoints.template block<StateType::RowsAtCompileTime, StateType::RowsAtCompileTime>(0, 1) = 
				(gamma * S_).colwise() + x;
			
			//Set the right block with (x - gamma * S)
			sigmaStatePoints.template rightCols<StateType::RowsAtCompileTime>() = 
				(-gamma * S_).colwise() + x;
			
			return true;
		}
		
		/**
         * @brief Predict expected sigma states from current sigma states using system model and control input
         * 
         * @note This covers equation (18) of Algorithm 3.1 in the Paper
         */
		void computeSigmaPointTransition()
		{
			for (int i = 0; i < SigmaPointCount; ++i)
			{	
				sigmaStatePoints.col(i) = A*sigmaStatePoints.col(i); //TODO: Add ' + B*u' as well 
			}
		}
		
		/**
         * @brief Predict the expected sigma measurements from predicted sigma states using measurement model
         * 
         * @note This covers equation (23) of Algorithm 3.1 in the Paper
         */
		void computeSigmaPointMeasurement()
		{
			for (int i = 0; i < SigmaPointCount; ++i)
			{
				//Same thing as transition version, but passing the sigma points through the measurement
				//function H
				
				//sigmaMeasurmentPoints.col(i) = H*sigmaMeasurementPoints.col(i);
			}
		}
		
		/**
		 * @brief Compute state or measurement prediciton from sigma points using pre-computed sigma weights
		 * 
		 * @note This covers equations (19) and (24) of Algorithm 3.1 in the Paper
		 *
		 * @param [in] sigmaPoints The computed sigma points of the desired type (state or measurement)
		 * @return The prediction
		 */
		StateType computePredictionFromSigmaPoints(const SigmaPointMatrix& sigmaPoints)
		{
			/* sigmaPoints:		NxL
			 * sigmaWeights:	Lx1
			 * 
			 * Where N is the state vector size and L is the number of sigma points */
			return sigmaPoints * sigmaWeights_m;
		}
		
		/**
         * @brief Compute the Covariance Square root from sigma points and noise covariance
         * 
         * @note This covers equations (20) and (21) as well as (25) and (26) of Algorithm 3.1 in the Paper
         * 
         * @param [in] mean The mean predicted state or measurement
         * @param [in] sigmaPoints the predicted sigma state or measurement points
         * @param [in] noiseCov The system or measurement noise covariance (as square root)
         * @param [out] cov The propagated state or innovation covariance (as square root)
         *
         * @return True on success, false if a numerical error is encountered when updating the matrix
         */
		bool computeCovarianceSquareRootFromSigmaPoints(
			const ColumnMatrix& mean,
			const SigmaPointMatrix& sigmaPoints, 
			const CovarianceSquareRoot& noiseCov,
			CovarianceSquareRoot& cov)
		{
			/* Compute the QR decomposition of (transposed) augmented matrix */
			Eigen::Matrix<T, (2*StateType::RowsAtCompileTime + StateType::RowsAtCompileTime), StateType::RowsAtCompileTime> tmp;
			
			tmp.template topRows<2*StateType::RowsAtCompileTime>() = std::sqrt(sigmaWeights_c(1)) * (sigmaPoints.template rightCols<SigmaPointCount - 1>().colwise() - mean).transpose();
			tmp.template bottomRows<StateType::RowsAtCompileTime>() = noiseCov.matrixU().toDenseMatrix();
			
			//TODO: Switch to ColPivHouseholderQR
			Eigen::HouseholderQR<decltype(tmp)> qr(tmp);
			
			/* Set R matrix as upper triangular square root */
			cov.setU(qr.matrixQR().template topRightCorner<StateType::RowsAtCompileTime, StateType::RowsAtCompileTime>());
			
			
			/* If the program fails at this line, it is due to memory allocation errors. To fix, replace line 240 of LLT.h
			 * (Eigen/src/Cholesky/LLT.h)
			 * 
			 * from: 
			 *		typedef Matrix<Scalar,Dynamic,1> TempVectorType;
			 * to: 
			 *		typedef Matrix<Scalar, VectorType::RowsAtCompileTime, 1> TempVectorType;
			 *	
			 * rankUpdate() eventually calls std::malloc(), which on an embedded system will fail. If using FreeRTOS & Thor, 
			 * only the new(), new[](), delete(), and delete[]() operators are overloaded for dynamic allocation. */
			cov.rankUpdate(sigmaPoints.template leftCols<1>() - mean, sigmaWeights_c(0));
			
			return (cov.info() == Eigen::Success);
		}
		
		
		
		void computePredictedState();
		void updateStateCovariance();
	};
	
	
	
}


#endif