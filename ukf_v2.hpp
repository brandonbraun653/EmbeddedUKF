#pragma once
#ifndef UKFV2_H_
#define UKFV2_H_

/* C++ Dependencies */
#include <stdlib.h>
#include <memory>
#include <string>

/* Eigen Dependencies */
#include <eigen/Eigen>

/* Boost Dependencies */
#include <boost/type_index.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/core/enable_if.hpp>
#include <boost/type_traits.hpp>

template<typename mType, int _StateVars, int _SigmaPts = (2*_StateVars + 1)>
	class UnscentedKalmanFilter
	{
	public:
		typedef Eigen::Matrix<mType, _StateVars, 1> mColumn;
		typedef Eigen::Matrix<mType, _StateVars, _StateVars> mSquare;
		typedef Eigen::Matrix<mType, _StateVars, _SigmaPts> mSigma;
		typedef Eigen::Matrix<mType, 1, _SigmaPts> mWeight;
		
		void initialize();
		Eigen::Matrix<mType, _StateVars, 1> iterate(Eigen::Matrix<mType, _StateVars, 1>& measured_data);
	
		UnscentedKalmanFilter(bool optimize_speed = false);
		~UnscentedKalmanFilter();
		
		struct UKF_InputMatrices
		{
			UKF_InputMatrices()
			{
				A.setZero(_StateVars, _StateVars);
				X.setZero(_StateVars, 1);
				B.setZero(_StateVars, _StateVars);
				P.setZero(_StateVars, _StateVars);
				Q.setZero(_StateVars, _StateVars);
				R.setZero(_StateVars, _StateVars);
				H.setZero(_StateVars, _StateVars);
			}
			mSquare A;      // State Transition Matrix (Phi in some papers)
			mColumn X;      // State Matrix (Also called "mean")
			mSquare B;      // Input Transition Matrix
			mSquare P;      // State Uncertainty/Covariance Matrix
			mSquare Q;      // Process Noise Matrix
			mSquare R;      // Measurement Uncertainty/Covariance Matrix
			mSquare H;      // Measurement Function
		};
		boost::shared_ptr<UKF_InputMatrices> Input;
		
		template<typename varType>
			struct UKF_MerweConstants
			{
				UKF_MerweConstants(){ cholesky.setZero(_StateVars, _StateVars); }

				varType alpha = 0.0;
				varType beta = 0.0;
				varType kappa = 0.0;
				varType lambda = 0.0;
				mSquare cholesky;
			};
		boost::shared_ptr<UKF_MerweConstants<mType>> Merwe;
		
	private:
		/*----------------------------
		 * Config & Debugging Variables
		 *---------------------------*/
		bool doublePrecision = false;
		bool enhanceSpeed = false;
		std::string error_message = "";
		
		/*----------------------------
		 * Runtime Functions 
		 *---------------------------*/
		bool dim_assert(size_t row_act, size_t row_exp, size_t col_act, size_t col_exp);
		void error_handler(std::string errorMessage);
		void merweSigmaPts();
		void takeMeasurement();
		void residual(Eigen::Matrix<mType, _StateVars, 1>& inputData);
		void kalmanGain();
		void update();
		void xCov(
			Eigen::Matrix<mType, _StateVars, 1>& meanH,
			Eigen::Matrix<mType, _StateVars, _SigmaPts>& sigmaH, 
			Eigen::Matrix<mType, _StateVars, 1>& meanJ,
			Eigen::Matrix<mType, _StateVars, _SigmaPts>& sigmaJ,
			Eigen::Matrix<mType, _StateVars, _StateVars>& xCov_out);
		
		/* Specialized functions that optimize speed depending on the data type used */
		template<typename U = mType, typename std::enable_if<std::is_same<U, float>::value, int>::type = 0>
			void unscentedTransform(
				Eigen::Matrix<mType, _StateVars, _SigmaPts>& sigmaPts, 
				Eigen::Matrix<mType, _StateVars, _StateVars>& noise_covariance,
				Eigen::Matrix<mType, _StateVars, 1>& X,
				Eigen::Matrix<mType, _StateVars, _StateVars>& P,
				bool allowSpeedup = true)
			{
				/* Calculate the new mean */
				for (int i = 0; i < _SigmaPts; i++)
					X += mResult->Wm(0, i)*sigmaPts.col(i);

				/* Calculate the new covariance */
				Eigen::Matrix<mType, _StateVars, 1> temp;
				Eigen::Matrix<mType, 1, _StateVars> tempT;

				for (int i = 0; i < _SigmaPts; i++)
				{
					temp = sigmaPts.col(i) - X;
					tempT = temp.transpose();

					P += mResult->Wc(0, i)*(temp)*tempT;
				}

				P += noise_covariance;
			}
		
		template<typename U = mType, typename std::enable_if<std::is_same<U, double>::value, int>::type = 0>
			void unscentedTransform(
				Eigen::Matrix<mType, _StateVars, _SigmaPts>& sigmaPts, 
				Eigen::Matrix<mType, _StateVars, _StateVars>& noise_covariance,
				Eigen::Matrix<mType, _StateVars, 1>& X,
				Eigen::Matrix<mType, _StateVars, _StateVars>& P,
				bool allowSpeedup = true)
			{
				/* Do everything in 32 bit for speed */
				if (enhanceSpeed && allowSpeedup)
				{
					/* Do some typecasting to get everything in the proper format */
					mResultOpt->Wm = mResult->Wm.template cast<float>();
					mResultOpt->Wc = mResult->Wc.template cast<float>();
					Eigen::Matrix<float, _StateVars, _SigmaPts> sigmaPtsf = sigmaPts.template cast<float>();
					Eigen::Matrix<float, _StateVars, _StateVars> noise_covariancef = noise_covariance.template cast<float>();
					Eigen::Matrix<float, _StateVars, 1> Xf = X.template cast<float>();
					Eigen::Matrix<float, _StateVars, _StateVars> Pf = P.template cast<float>();
		
					/* Calculate the new mean */
					for (int i = 0; i < _SigmaPts; i++)
						Xf += mResultOpt->Wm(0, i)*sigmaPtsf.col(i);

					/* Calculate the new covariance */
					Eigen::Matrix<float, _StateVars, 1> temp;
					Eigen::Matrix<float, 1, _StateVars> tempT;

					for (int i = 0; i < _SigmaPts; i++)
					{
						temp = sigmaPtsf.col(i) - Xf;
						tempT = temp.transpose();

						Pf += mResultOpt->Wc(0, i)*(temp)*tempT;
					}

					Pf += noise_covariancef;
					
					/* Cast results back to double */
					mResult->Xp = Xf.template cast<double>();
					mResult->Pp = Pf.template cast<double>();
				}
				else
				{
					/* Calculate the new mean */
					for (int i = 0; i < _SigmaPts; i++)
						X += mResult->Wm(0, i)*sigmaPts.col(i);

					/* Calculate the new covariance */
					Eigen::Matrix<mType, _StateVars, 1> temp;
					Eigen::Matrix<mType, 1, _StateVars> tempT;

					for (int i = 0; i < _SigmaPts; i++)
					{
						temp = sigmaPts.col(i) - X;
						tempT = temp.transpose();

						P += mResult->Wc(0, i)*(temp)*tempT;
					}

					P += noise_covariance;
				}
			}
		
		template<typename U = mType, typename std::enable_if<std::is_same<U, float>::value, int>::type = 0>
			void sigmaProjection(Eigen::Matrix<float, _StateVars, 1>& inputData)
			{
				for (int i = 0; i < _SigmaPts; i++)
					mResult->Ysig.col(i) = Input->A*mResult->Xsig.col(i) + Input->B*inputData;
			}
		
		template<typename U = mType, typename std::enable_if<std::is_same<U, double>::value, int>::type = 0>
			void sigmaProjection(Eigen::Matrix<double, _StateVars, 1>& inputData)
			{
				if (enhanceSpeed)
				{
					//Calculations done in forced 32 bit precision for speed
					mResultOpt->Xsig = mResult->Xsig.template cast<float>();
					Eigen::Matrix<float, _StateVars, 1> inputDataf = inputData.template cast<float>();
		
					for (int i = 0; i < _SigmaPts; i++)
						mResultOpt->Ysig.col(i) = mInputOpt->A*mResultOpt->Xsig.col(i) + mInputOpt->B*inputDataf;
		
					//Cast the results back to double for later calculations
					mResult->Ysig = mResultOpt->Ysig.template cast<double>();
				}
				else
				{
					for (int i = 0; i < _SigmaPts; i++)
						mResult->Ysig.col(i) = Input->A*mResult->Xsig.col(i) + Input->B*inputData;
				}
			}
	
		/*----------------------------
		 * Runtime Data Containers 
		 *---------------------------*/
		struct UKF_RunTimeMatrices
		{
			UKF_RunTimeMatrices()
			{
				Yk.setZero(_StateVars, 1);
				Xk.setZero(_StateVars, 1);
				Xp.setZero(_StateVars, 1);
				Xm.setZero(_StateVars, 1);
				Pk.setZero(_StateVars, _StateVars);
				Pp.setZero(_StateVars, _StateVars);
				Pm.setZero(_StateVars, _StateVars);
				Wc.setZero(1, _SigmaPts);
				Wm.setZero(1, _SigmaPts);
				K.setZero(_StateVars, _StateVars);
				Pxm.setZero(_StateVars, _StateVars);
				Pm_inv.setZero(_StateVars, _StateVars);
				Xsig.setZero(_StateVars, _SigmaPts);
				Ysig.setZero(_StateVars, _SigmaPts);
				Zsig.setZero(_StateVars, _SigmaPts);
			}

			mColumn Yk;     			//Residual between measured and predicted
			mColumn Xk, Xp, Xm;     	//[Current, Predicted, Measured] States
			mSquare Pk, Pp, Pm;     	//[Current, Predicted, Measured] Covariances
			mWeight Wc, Wm;        		//Mean & Covariance Sigma Weights
			mSquare Pm_inv;     		//Psuedo inverse of Pm
			mSquare Pxm;     			//Cross covariance between State & Measurement
			mSquare K;      			//Kalman Gain
			mSigma Xsig, Ysig, Zsig;    //Sigma Points
		};
		boost::shared_ptr<UKF_RunTimeMatrices> mResult;
		
		struct UKF_InputMatrices_OPT
		{
			UKF_InputMatrices_OPT()
			{
				A.setZero(_StateVars, _StateVars);
				B.setZero(_StateVars, _StateVars);
				Q.setZero(_StateVars, _StateVars);
				R.setZero(_StateVars, _StateVars);
				H.setZero(_StateVars, _StateVars);
			}
			
			Eigen::Matrix<float, _StateVars, _StateVars> A;  		// State Transition Matrix (Phi in some papers)
			Eigen::Matrix<float, _StateVars, _StateVars> B;  		// Input Transition Matrix
			Eigen::Matrix<float, _StateVars, _StateVars> Q;  		// Process Noise Matrix
			Eigen::Matrix<float, _StateVars, _StateVars> R;  		// Measurement Uncertainty/Covariance Matrix
			Eigen::Matrix<float, _StateVars, _StateVars> H;  		// Measurement Function
		};
		boost::shared_ptr<UKF_InputMatrices_OPT> mInputOpt;
		
		struct UKF_RunTimeMatrices_OPT
		{
			UKF_RunTimeMatrices_OPT()
			{
				Xk.setZero(_StateVars, 1);
				Xp.setZero(_StateVars, 1);
				Xm.setZero(_StateVars, 1);
				Pk.setZero(_StateVars, _StateVars);
				Pp.setZero(_StateVars, _StateVars);
				Pm.setZero(_StateVars, _StateVars);
				Wc.setZero(1, _SigmaPts);
				Wm.setZero(1, _SigmaPts);
				Xsig.setZero(_StateVars, _SigmaPts);
				Ysig.setZero(_StateVars, _SigmaPts);
				Zsig.setZero(_StateVars, _SigmaPts);
			}
			
			Eigen::Matrix<float, 1, _SigmaPts> Wc, Wm;              				//Mean & Covariance Sigma Weights
			Eigen::Matrix<float, _StateVars, _SigmaPts> Xsig, Ysig, Zsig;    		//Sigma Points
			Eigen::Matrix<float, _StateVars, 1> Xk, Xp, Xm;            			//[Current, Predicted, Measured] States
			Eigen::Matrix<float, _StateVars, _StateVars> Pk, Pp, Pm;            	//[Current, Predicted, Measured] Covariances
		};
		boost::shared_ptr<UKF_RunTimeMatrices_OPT> mResultOpt;
	};

/*------------------------------------
* Constructor/Destructor
*------------------------------------*/
template<typename mType, int _StateVars, int _SigmaPts>
	UnscentedKalmanFilter<mType, _StateVars, _SigmaPts>::UnscentedKalmanFilter(bool optimize_speed)
	{
		/* Initialize all the matrices used in the algorithm. Must specifically use the aligned allocator due to lovely
		 * problems described here:
		 * http://eigen.tuxfamily.org/dox/group__TopicUnalignedArrayAssert.html */
		Input = boost::allocate_shared<UKF_InputMatrices>(Eigen::aligned_allocator<UKF_InputMatrices>());
		mResult = boost::allocate_shared<UKF_RunTimeMatrices>(Eigen::aligned_allocator<UKF_RunTimeMatrices>());
		Merwe = boost::allocate_shared<UKF_MerweConstants<mType>>(Eigen::aligned_allocator<UKF_MerweConstants<mType>>());
		
		/* Figure out whether or not we are using double precision (64-bit). This will become useful
		 * when the user wants to run a faster version of 64 bit code. It can save about 33% execution cost.*/
		mType temp;
		std::string dType("double");
		std::string ukf_type = boost::typeindex::type_id<decltype(temp)>().pretty_name();
		
		if (ukf_type == dType)
		{
			doublePrecision = true;
			
			if (optimize_speed)
			{
				enhanceSpeed = true;
			
				/* Initialize the optimum version of the Input and Result structures */
				mInputOpt = boost::allocate_shared<UKF_InputMatrices_OPT>(Eigen::aligned_allocator<UKF_InputMatrices_OPT>());
				mResultOpt = boost::allocate_shared<UKF_RunTimeMatrices_OPT>(Eigen::aligned_allocator<UKF_RunTimeMatrices_OPT>());
			}
		}		
	}

template<typename mType, int _StateVars, int _SigmaPts>
	UnscentedKalmanFilter<mType, _StateVars, _SigmaPts>::~UnscentedKalmanFilter()
	{
	}

/*------------------------------------
* Public Functions
*------------------------------------*/
template<typename mType, int _StateVars, int _SigmaPts>
	void UnscentedKalmanFilter<mType, _StateVars, _SigmaPts>::initialize()
	{
		/* Initialize the Xk and Pk matrices. This is done in here rather
		 * than in the constructor to allow the user to setup all the requisite
		 * matrices properly. The enhanceSpeed option only runs if the user has
		 * selected 64-bit mode (double) and requests optimization through the constructor. */		
		if (enhanceSpeed)
		{
			//Initial state and covariance matrices
			//mResultOpt->Xk = Input->X.template cast<float>();
			//mResultOpt->Pk = Input->P.template cast<float>();
			//
			////Other major matrices
			//mInputOpt->A = Input->A.template cast<float>();
			//mInputOpt->B = Input->B.template cast<float>();
			//mInputOpt->Q = Input->A.template cast<float>();
			//mInputOpt->R = Input->B.template cast<float>();
			//mInputOpt->H = Input->A.template cast<float>();
		}
		else
		{
			mResult->Xk = Input->X;
			mResult->Pk = Input->P;
		}
			
	}

template<typename mType, int _StateVars, int _SigmaPts>
	Eigen::Matrix<mType, _StateVars, 1> UnscentedKalmanFilter<mType, _StateVars, _SigmaPts>::iterate(Eigen::Matrix<mType, _StateVars, 1>& measured_data) 
	{
		/*------------------------------
		* Predict Step
		*------------------------------*/
		// Generate the sigma points and weights
		merweSigmaPts();
		
		// Project the sigma points through the given model Ax+Bu
		sigmaProjection(measured_data);

		// Compute the mean and covariance of the prediction with the Unscented Transform
		unscentedTransform(mResult->Ysig, Input->Q, mResult->Xp, mResult->Pp);
		
		/*------------------------------
		* Update Step
		*------------------------------*/
		// Convert the sigma points into a measurement
		takeMeasurement();
		
		// Compute the mean and covariance of that measurement
		unscentedTransform(mResult->Zsig, Input->R, mResult->Xm, mResult->Pm, false);
		
		// Compute the residual between measured and predicted
		residual(measured_data);
		
		// Compute the cross covariance of the state and measurement
		xCov(mResult->Xp, mResult->Ysig, mResult->Xm, mResult->Zsig, mResult->Pxm);
		
		// Compute the Kalman Gain
		kalmanGain();
		
		//Compute the new state estimate and covariance
		update();
		
		return mResult->Xk;
	}

/*------------------------------------
* Private Functions
*------------------------------------*/
template<typename mType, int _StateVars, int _SigmaPts>
	bool UnscentedKalmanFilter<mType, _StateVars, _SigmaPts>::dim_assert(size_t row_act, size_t row_exp, size_t col_act, size_t col_exp)
	{
		if ((row_act != row_exp) || (col_act != col_exp))
		{
			std::string msg("Failed matrix dimension assert.");
			error_handler(msg);
		}
		return true;
	}

template<typename mType, int _StateVars, int _SigmaPts>
	void UnscentedKalmanFilter<mType, _StateVars, _SigmaPts>::error_handler(std::string errorMessage)
	{
		/* If for some reason you find yourself here, A-aron done messed up. Check the error message
		 * in a debugger and trace the call stack to find out what the problem is. */
		error_message = errorMessage;
		for (;;) {}
	}

template<typename mType, int _StateVars, int _SigmaPts>
	void UnscentedKalmanFilter<mType, _StateVars, _SigmaPts>::xCov(
		Eigen::Matrix<mType, _StateVars, 1>& meanH,
		Eigen::Matrix<mType, _StateVars, _SigmaPts>& sigmaH, 
		Eigen::Matrix<mType, _StateVars, 1>& meanJ,
		Eigen::Matrix<mType, _StateVars, _SigmaPts>& sigmaJ,
		Eigen::Matrix<mType, _StateVars, _StateVars>& xCov_out)
	{
		/* Create some temporary variables for the transpose calculation*/
		Eigen::Matrix<mType, _StateVars, 1> tempJ;
		Eigen::Matrix<mType, 1, _StateVars> tempJT;

		for (int i = 0; i < _SigmaPts; i++)
		{
			tempJ = sigmaJ.col(i) - meanJ;
			tempJT = tempJ.transpose();

			xCov_out += mResult->Wc(0, i)*(sigmaH.col(i) - meanH)*tempJT;
		}
	}

template<typename mType, int _StateVars, int _SigmaPts>
	void UnscentedKalmanFilter<mType, _StateVars, _SigmaPts>::merweSigmaPts()
	{
		/* First, find the Cholesky Decomposition of Pk, the state covariance matrix */
		Eigen::LDLT<mSquare> lltOfP(mResult->Pk*(_StateVars + Merwe->lambda));
		Merwe->cholesky = lltOfP.matrixL();
		
		/* Calculate the sigma points */
		mResult->Xsig.col(0) = mResult->Xk;
		for (int i = 1; i < _SigmaPts; i++)
		{
			if (i <= _StateVars)
				mResult->Xsig.col(i) = mResult->Xk + Merwe->cholesky.col(i - 1);

			if (i > _StateVars)
				mResult->Xsig.col(i) = mResult->Xk - Merwe->cholesky.col(i - 1 - _StateVars);
		}

		/* Calculate the weights */
		mResult->Wm(0, 0) = Merwe->lambda / (_StateVars + Merwe->lambda);
		mResult->Wc(0, 0) = mResult->Wm(0, 0) + 1.0 - Merwe->alpha*Merwe->alpha + Merwe->beta;

		for (int i = 1; i < _SigmaPts; i++)
		{
			mResult->Wm(0, i) = 1.0 / (2.0*(_StateVars + Merwe->lambda));
			mResult->Wc(0, i) = mResult->Wm(0, i);
		}
	}

template<typename mType, int _StateVars, int _SigmaPts>
	void UnscentedKalmanFilter<mType, _StateVars, _SigmaPts>::takeMeasurement()
	{
		mResult->Zsig = Input->H*mResult->Ysig;
	}

template<typename mType, int _StateVars, int _SigmaPts>
	void UnscentedKalmanFilter<mType, _StateVars, _SigmaPts>::residual(Eigen::Matrix<mType, _StateVars, 1>& inputData)
	{
		mResult->Yk = inputData - mResult->Xm;
	}

template<typename mType, int _StateVars, int _SigmaPts>
	void UnscentedKalmanFilter<mType, _StateVars, _SigmaPts>::kalmanGain()
	{
		mResult->Pm_inv = mResult->Pm.inverse();
		mResult->K = mResult->Pxm*mResult->Pm_inv;
	}

template<typename mType, int _StateVars, int _SigmaPts>
	void UnscentedKalmanFilter<mType, _StateVars, _SigmaPts>::update()
	{
		Eigen::Matrix<mType, _StateVars, _StateVars> kT = mResult->K.transpose();
		
		mResult->Xk = mResult->Xp + mResult->K*mResult->Yk;
		mResult->Pk = mResult->Pp - mResult->K*mResult->Pm*kT;
	}
#endif