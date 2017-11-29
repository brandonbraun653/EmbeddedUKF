#pragma once
#ifndef UKF_H_
#define UKF_H_
/*------------------------------------
* Includes
*------------------------------------*/
/* C++ Dependencies */
#include <stdlib.h>
#include <memory>

/* Eigen Dependencies */
#include <eigen/Eigen>
#include <eigen/Eigenvalues>
#include <eigen/StdVector>

/* Boost Dependencies */
#include <boost/smart_ptr.hpp>
#include <boost/shared_ptr.hpp>

/* HAL Dependencies */
#include <stm32f7xx_hal.h>
#include <stm32_hal_legacy.h>

/* Thor Dependencies*/
#include "thor.h"
#include "gdb.h"

/* UKF Dependencies */
#include "ukf_config.h"

/*------------------------------------
* Data Structures
*------------------------------------*/
struct UKF_InputMatrices_32
{
	UKF_InputMatrices_32()
	{
		A.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		X.setZero(NUM_STATE_VARS, 1);
		B.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		U.setZero(NUM_STATE_VARS, 1);
		P.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Q.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		R.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		H.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
	}

	mfSquare A;    // State Transition Matrix (Phi in some papers)
	mfColumn X;    // State Matrix (Also called "mean")
	mfSquare B;    // Input Transition Matrix
	mfColumn U;    // Input Matrix
	mfSquare P;    // State Uncertainty/Covariance Matrix
	mfSquare Q;    // Process Noise Matrix
	mfSquare R;    // Measurement Uncertainty/Covariance Matrix
	mfSquare H;    // Measurement Function

	const size_t N = NUM_STATE_VARS;
};

struct UKF_InputMatrices_64
{
	UKF_InputMatrices_64()
	{
		A.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		X.setZero(NUM_STATE_VARS, 1);
		B.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		U.setZero(NUM_STATE_VARS, 1);
		P.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Q.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		R.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		H.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
	}

	mdSquare A;     // State Transition Matrix (Phi in some papers)
	mdColumn X;     // State Matrix (Also called "mean")
	mdSquare B;     // Input Transition Matrix
	mdColumn U;     // Input Matrix
	mdSquare P;     // State Uncertainty/Covariance Matrix
	mdSquare Q;     // Process Noise Matrix
	mdSquare R;     // Measurement Uncertainty/Covariance Matrix
	mdSquare H;     // Measurement Function

	const size_t N = NUM_STATE_VARS;
};

struct UKF_InputMatrices_Opt
{
	UKF_InputMatrices_Opt()
	{
		A.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		X.setZero(NUM_STATE_VARS, 1);
		B.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		U.setZero(NUM_STATE_VARS, 1);
		P.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Q.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		R.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		H.setZero(NUM_STATE_VARS, NUM_STATE_VARS);

		A_single.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		B_single.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Q_single.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		H_single.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		R_single.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
	}

	mdSquare A;      // State Transition Matrix (Phi in some papers)
	mdColumn X;      // State Matrix (Also called "mean")
	mdSquare B;      // Input Transition Matrix
	mdColumn U;      // Input Matrix
	mdSquare P;      // State Uncertainty/Covariance Matrix
	mdSquare Q;      // Process Noise Matrix
	mdSquare R;      // Measurement Uncertainty/Covariance Matrix
	mdSquare H;      // Measurement Function

	const size_t N = NUM_STATE_VARS;

	/* SINGLE PRECISION VERSIONS */
	mfSquare A_single;
	mfSquare B_single;
	mfSquare Q_single;
	mfSquare H_single;
	mfSquare R_single;
};

struct UKF_RunTimeMatrices_32
{
	UKF_RunTimeMatrices_32()
	{
		Yk.setZero(NUM_STATE_VARS, 1);
		Xk.setZero(NUM_STATE_VARS, 1);
		Xp.setZero(NUM_STATE_VARS, 1);
		Xm.setZero(NUM_STATE_VARS, 1);
		Pk.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pp.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pm.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Wc.setZero(1, NUM_SIGMA_POINTS);
		Wm.setZero(1, NUM_SIGMA_POINTS);
		K.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pxm.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pm_inv.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Xsig.setZero(NUM_STATE_VARS, NUM_SIGMA_POINTS);
		Ysig.setZero(NUM_STATE_VARS, NUM_SIGMA_POINTS);
		Zsig.setZero(NUM_STATE_VARS, NUM_SIGMA_POINTS);
	}

	mfColumn Yk;  				//Residual between measured and predicted
	mfColumn Xk, Xp, Xm;  		//[Current, Predicted, Measured] States
	mfSquare Pk, Pp, Pm;  		//[Current, Predicted, Measured] Covariances
	mfWeight Wc, Wm;     			//Mean & Covariance Sigma Weights
	mfSquare Pm_inv;  			//Psuedo inverse of Pm
	mfSquare Pxm;  				//Cross covariance between State & Measurement
	mfSquare K;   				//Kalman Gain
	mfSigma Xsig, Ysig, Zsig;   	//Sigma Points

	const size_t sigPts = NUM_SIGMA_POINTS;
};

struct UKF_RunTimeMatrices_64
{
	UKF_RunTimeMatrices_64()
	{
		Yk.setZero(NUM_STATE_VARS, 1);
		Xk.setZero(NUM_STATE_VARS, 1);
		Xp.setZero(NUM_STATE_VARS, 1);
		Xm.setZero(NUM_STATE_VARS, 1);
		Pk.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pp.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pm.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Wc.setZero(1, NUM_SIGMA_POINTS);
		Wm.setZero(1, NUM_SIGMA_POINTS);
		K.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pxm.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pm_inv.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Xsig.setZero(NUM_STATE_VARS, NUM_SIGMA_POINTS);
		Ysig.setZero(NUM_STATE_VARS, NUM_SIGMA_POINTS);
		Zsig.setZero(NUM_STATE_VARS, NUM_SIGMA_POINTS);
	}

	mdColumn Yk;   				//Residual between measured and predicted
	mdColumn Xk, Xp, Xm;   		//[Current, Predicted, Measured] States
	mdSquare Pk, Pp, Pm;   		//[Current, Predicted, Measured] Covariances
	mdWeight Wc, Wm;      		//Mean & Covariance Sigma Weights
	mdSquare Pm_inv;   			//Psuedo inverse of Pm
	mdSquare Pxm;   				//Cross covariance between State & Measurement
	mdSquare K;    				//Kalman Gain
	mdSigma Xsig, Ysig, Zsig;    	//Sigma Points

	const size_t sigPts = NUM_SIGMA_POINTS;
};

struct UKF_RunTimeMatrices_Opt
{
	UKF_RunTimeMatrices_Opt()
	{
		Yk.setZero(NUM_STATE_VARS, 1);
		Xk_double.setZero(NUM_STATE_VARS, 1);
		Xp_double.setZero(NUM_STATE_VARS, 1);
		Xm_double.setZero(NUM_STATE_VARS, 1);
		Pk_double.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pp_double.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pm_double.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Wc_double.setZero(1, NUM_SIGMA_POINTS);
		Wm_double.setZero(1, NUM_SIGMA_POINTS);
		K_double.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pxm_double.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pm_inv_double.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Xsig_double.setZero(NUM_STATE_VARS, NUM_SIGMA_POINTS);
		Ysig_double.setZero(NUM_STATE_VARS, NUM_SIGMA_POINTS);
		Zsig_double.setZero(NUM_STATE_VARS, NUM_SIGMA_POINTS);

		Pk_single.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pp_single.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pm_single.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Xk_single.setZero(NUM_STATE_VARS, 1);
		Xp_single.setZero(NUM_STATE_VARS, 1);
		Xm_single.setZero(NUM_STATE_VARS, 1);
		Wc_single.setZero(1, NUM_SIGMA_POINTS);
		Wm_single.setZero(1, NUM_SIGMA_POINTS);
		K_single.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pxm_single.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Pm_inv_single.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
		Xsig_single.setZero(NUM_STATE_VARS, NUM_SIGMA_POINTS);
		Ysig_single.setZero(NUM_STATE_VARS, NUM_SIGMA_POINTS);
		Zsig_single.setZero(NUM_STATE_VARS, NUM_SIGMA_POINTS);
	}

	/* DOUBLE VERSIONS */
	mdColumn Yk;    				//Residual between measured and predicted
	mdColumn Xk_double, Xp_double, Xm_double;    		//[Current, Predicted, Measured] States
	mdSquare Pk_double, Pp_double, Pm_double;    		//[Current, Predicted, Measured] Covariances
	mdSquare Pm_inv_double;    			//Psuedo inverse of Pm
	mdSquare Pxm_double;    			//Cross covariance between State & Measurement
	mdSquare K_double;     				//Kalman Gain
	mdWeight Wc_double, Wm_double;        				//Mean & Covariance Sigma Weights
	mdSigma Xsig_double, Ysig_double, Zsig_double; 		//Sigma Points

	/* FLOAT VERSIONS */
	mfSigma Xsig_single, Ysig_single, Zsig_single;
	mfWeight Wc_single, Wm_single;
	mfColumn Xk_single, Xp_single, Xm_single;
	mfSquare Pk_single, Pp_single, Pm_single;
	mfSquare Pm_inv_single;
	mfSquare Pxm_single;
	mfSquare K_single;

	const size_t sigPts = NUM_SIGMA_POINTS;
};

struct MerweConstants_32
{
	MerweConstants_32()
	{
		cholesky.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
	}

	float alpha = 0.0;
	float beta = 0.0;
	float kappa = 0.0;
	float lambda = 0.0;
	mfSquare cholesky;
};

struct MerweConstants_64
{
	MerweConstants_64()
	{
		cholesky.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
	}

	double alpha = 0.0;
	double beta = 0.0;
	double kappa = 0.0;
	double lambda = 0.0;
	mdSquare cholesky;
};

struct MerweConstants_Opt
{
	MerweConstants_Opt()
	{
		cholesky.setZero(NUM_STATE_VARS, NUM_STATE_VARS);
	}

	double alpha = 0.0;
	double beta = 0.0;
	double kappa = 0.0;
	double lambda = 0.0;
	mdSquare cholesky;
};

class UnscentedKalmanFilter_32
{
public:
	void assignErrorLed(boost::shared_ptr<GPIOClass> led);
	void initialize(UKF_InputMatrices_32& input_matrix, MerweConstants_32& input_merwe, boost::shared_ptr<GPIOClass> errLedPin = NULL);
	mfColumn iterate(mfColumn &measured_data);

	UnscentedKalmanFilter_32();
	~UnscentedKalmanFilter_32();

private:
	/* Debugging Variables & Tools */
	bool errorLed_Assigned;
	std::string error_message;
	boost::shared_ptr<GPIOClass> ledPin;

	// Holds all the various matrices needed for the UKF alg.
	MerweConstants_32 *merwe;
	UKF_InputMatrices_32 *mConfig;
	UKF_RunTimeMatrices_32 mResult;

	/* 32 bit Evaluation Functions */
	void calc_UT32(mfSigma sigmaPts, mfSquare noise_covariance, mfWeight wc, mfWeight wm, mfColumn &X, mfSquare &P);
	void calc_XCov32(mfColumn meanX, mfSigma sigmaX, mfColumn meanM, mfSigma sigmaM, mfWeight wc, mfSquare &cross_covariance);
	void calc_MerweSigma32(mfColumn m_X, mfSquare m_P, mfWeight &m_wc, mfWeight &m_wm, mfSigma &m_xsig);

	bool dim_assert(size_t row_act, size_t row_exp, size_t col_act, size_t col_exp);
	void error_handler(std::string errorMessage);
};

class UnscentedKalmanFilter_64
{
public:
	void assignErrorLed(boost::shared_ptr<GPIOClass> led);
	void initialize(UKF_InputMatrices_64& input_matrix, MerweConstants_64& input_merwe, boost::shared_ptr<GPIOClass> errLedPin = NULL);
	mdColumn iterate(mdColumn &measured_data);

	UnscentedKalmanFilter_64();
	~UnscentedKalmanFilter_64();

private:
	/* Debugging Variables & Tools */
	bool errorLed_Assigned;
	std::string error_message;
	boost::shared_ptr<GPIOClass> ledPin;

	// Holds all the various matrices needed for the UKF alg.
	MerweConstants_64 *merwe;
	UKF_InputMatrices_64 *mConfig;
	UKF_RunTimeMatrices_64 mResult;

	/* 64 bit Evaluation Functions */
	void calc_UT64(mdSigma sigmaPts, mdSquare noise_covariance, mdWeight wc, mdWeight wm, mdColumn &X, mdSquare &P);
	void calc_XCov64(mdColumn meanX, mdSigma sigmaX, mdColumn meanM, mdSigma sigmaM, mdWeight wc, mdSquare &cross_covariance);
	void calc_MerweSigma64(mdColumn m_X, mdSquare m_P, mdWeight &m_wc, mdWeight &m_wm, mdSigma &m_xsig);

	bool dim_assert(size_t row_act, size_t row_exp, size_t col_act, size_t col_exp);
	void error_handler(std::string errorMessage);
};

class UnscentedKalmanFilter_Opt
{
public:
	void assignErrorLed(boost::shared_ptr<GPIOClass> led);
	void initialize(UKF_InputMatrices_Opt& input_matrix, MerweConstants_Opt& input_merwe, boost::shared_ptr<GPIOClass> errLedPin = NULL);
	mdColumn iterate(mdColumn &measured_data);

	UnscentedKalmanFilter_Opt();
	~UnscentedKalmanFilter_Opt();

private:
	/* Debugging Variables & Tools */
	bool errorLed_Assigned;
	std::string error_message;
	boost::shared_ptr<GPIOClass> ledPin;

	// Holds all the various matrices needed for the UKF alg.
	MerweConstants_Opt *merwe;
	UKF_InputMatrices_Opt *mConfig;
	UKF_RunTimeMatrices_Opt mResult;

	/* Evaluation Functions */
	void calc_XCovOpt(mdColumn meanX, mdSigma sigmaX, mdColumn meanM, mdSigma sigmaM, mdWeight wc, mdSquare &cross_covariance);
	void calc_UTOpt(mdSigma sigmaPts, mdSquare noise_covariance, mdWeight wc, mdWeight wm, mdColumn &X, mdSquare &P);
	void calc_MerweSigmaOpt(mdColumn m_X, mdSquare m_P, mdWeight &m_wc, mdWeight &m_wm, mdSigma &m_xsig);

	bool dim_assert(size_t row_act, size_t row_exp, size_t col_act, size_t col_exp);
	void error_handler(std::string errorMessage);

	void calc_UTOpt_MIXEDTEST(mfSigma sigmaPts, mfSquare noise_covariance, mfWeight wc, mfWeight wm, mfColumn &X, mfSquare &P);
};
#endif /* UKF_H_ */