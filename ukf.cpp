#include "ukf.hpp"

//////////////////////////////////////////////////////////
/*CLASS: UnscentedKalmanFilter_32 */
//////////////////////////////////////////////////////////
/*------------------------------------
* Constructor/Destructor
*------------------------------------*/
UnscentedKalmanFilter_32::UnscentedKalmanFilter_32()
{
	errorLed_Assigned = false;
}

UnscentedKalmanFilter_32::~UnscentedKalmanFilter_32()
{
}

/*------------------------------------
* Public Functions
*------------------------------------*/
void UnscentedKalmanFilter_32::initialize(UKF_InputMatrices_32& input_matrix, MerweConstants_32& input_merwe, boost::shared_ptr<GPIOClass> errLedPin)
{
	if (errLedPin != NULL)
		assignErrorLed(errLedPin);

	dim_assert(input_matrix.A.rows(), NUM_STATE_VARS, input_matrix.A.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.X.rows(), NUM_STATE_VARS, input_matrix.X.cols(), 1);
	dim_assert(input_matrix.B.rows(), NUM_STATE_VARS, input_matrix.B.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.U.rows(), NUM_STATE_VARS, input_matrix.U.cols(), 1);
	dim_assert(input_matrix.P.rows(), NUM_STATE_VARS, input_matrix.P.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.Q.rows(), NUM_STATE_VARS, input_matrix.Q.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.R.rows(), NUM_STATE_VARS, input_matrix.R.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.H.rows(), NUM_STATE_VARS, input_matrix.H.cols(), NUM_STATE_VARS);

	mConfig = &input_matrix;
	merwe = &input_merwe;

	/* Initialize the Xk and Pk matrices */
	mResult.Xk = mConfig->X;
	mResult.Pk = mConfig->P;
}

void UnscentedKalmanFilter_32::assignErrorLed(boost::shared_ptr<GPIOClass> led)
{
	ledPin = led;
	ledPin->mode(OUTPUT_PP);
	ledPin->write(LOW);

	errorLed_Assigned = true;
}

mfColumn UnscentedKalmanFilter_32::iterate(mfColumn &measured_data)
{
	/*------------------------------
	* Predict Step
	*------------------------------*/
	// Generate the sigma points and weights
	calc_MerweSigma32(mResult.Xk, mResult.Pk, mResult.Wc, mResult.Wm, mResult.Xsig);

	// Project the sigma points through the given model Ax+Bu+Wn
	for(int i = 0 ; i < NUM_SIGMA_POINTS ; i++)
		mResult.Ysig.col(i) = mConfig->A*mResult.Xsig.col(i) + mConfig->B*measured_data;

	// Compute the mean and covariance of the prediction with the Unscented Transform
	calc_UT32(mResult.Ysig, mConfig->Q, mResult.Wc, mResult.Wm, mResult.Xp, mResult.Pp);

	/*------------------------------
	* Update Step
	*------------------------------*/
	// Convert the sigma points into a measurement
	mResult.Zsig = mConfig->H*mResult.Ysig;

	// Compute the mean and covariance of that measurement
	calc_UT32(mResult.Zsig, mConfig->R, mResult.Wc, mResult.Wm, mResult.Xm, mResult.Pm);

	// Compute the residual between measured and predicted
	mResult.Yk = measured_data - mResult.Xm;

	// Compute the cross covariance of the state and measurement
	calc_XCov32(mResult.Xp, mResult.Ysig, mResult.Xm, mResult.Zsig, mResult.Wc, mResult.Pxm);

	// Compute the Kalman Gain
	mResult.Pm_inv = mResult.Pm.completeOrthogonalDecomposition().pseudoInverse();
	mResult.K = mResult.Pxm*mResult.Pm_inv;

	//Compute the new state estimate and covariance
	mfSquare kT = mResult.K.transpose();

	mResult.Xk = mResult.Xp + mResult.K*mResult.Yk;
	mResult.Pk = mResult.Pp - mResult.K*mResult.Pm*kT;

	return mResult.Xk;
}
/*------------------------------------
* Private Functions
*------------------------------------*/
bool UnscentedKalmanFilter_32::dim_assert(size_t row_act, size_t row_exp, size_t col_act, size_t col_exp)
{
	if ((row_act != row_exp) || (col_act != col_exp))
	{
		std::string msg("Failed matrix dimension assert.");
		error_handler(msg);
	}
	return true;
}

void UnscentedKalmanFilter_32::error_handler(std::string errorMessage)
{
	error_message = errorMessage;
	for (;;)
	{
		if (errorLed_Assigned)
		{
			ledPin->write(HIGH);
			HAL_Delay(250);
			ledPin->write(LOW);
			HAL_Delay(250);
		}
	}
}

void UnscentedKalmanFilter_32::calc_XCov32(mfColumn meanX, mfSigma sigmaX, mfColumn meanM, mfSigma sigmaM, mfWeight wc, mfSquare &cross_covariance)
{
	Eigen::Matrix<float, NUM_STATE_VARS, 1> tempM;
	Eigen::Matrix<float, 1, NUM_STATE_VARS> tempMT;

	for (int i = 0; i < NUM_SIGMA_POINTS; i++)
	{
		tempM = sigmaM.col(i) - meanM;
		tempMT = tempM.transpose();

		cross_covariance += wc(0, i)*(sigmaX.col(i) - meanX)*tempMT;
	}
}

void UnscentedKalmanFilter_32::calc_UT32(mfSigma sigmaPts, mfSquare noise_covariance, mfWeight wc, mfWeight wm, mfColumn &X, mfSquare &P)
{
	/* Calculate the new mean */
	for (int i = 0; i < NUM_SIGMA_POINTS; i++)
		X += wm(0, i)*sigmaPts.col(i);

	/* Calculate the new covariance */
	Eigen::Matrix<float, NUM_STATE_VARS, 1> temp;
	Eigen::Matrix<float, 1, NUM_STATE_VARS> tempT;

	for (int i = 0; i < NUM_SIGMA_POINTS; i++)
	{
		temp = sigmaPts.col(i) - X;
		tempT = temp.transpose();

		P += wc(0, i)*(temp)*tempT;
	}

	P += noise_covariance;
}

void UnscentedKalmanFilter_32::calc_MerweSigma32(mfColumn m_X, mfSquare m_P, mfWeight &m_wc, mfWeight &m_wm, mfSigma &m_xsig)
{
	/* First, find the Cholesky Decomposition of m_P, the state covariance matrix */
	Eigen::LDLT<mfSquare> lltOfP(m_P*(NUM_STATE_VARS + merwe->lambda));
	merwe->cholesky = lltOfP.matrixL();

	/* Calculate the sigma points */
	m_xsig.col(0) = m_X;
	for (int i = 1; i < NUM_SIGMA_POINTS; i++)
	{
		if (i <= NUM_STATE_VARS)
			m_xsig.col(i) = m_X + merwe->cholesky.col(i - 1);

		if (i > NUM_STATE_VARS)
			m_xsig.col(i) = m_X - merwe->cholesky.col(i - 1 - NUM_STATE_VARS);
	}

	/* Calculate the weights */
	m_wm(0, 0) = merwe->lambda / (NUM_STATE_VARS + merwe->lambda);
	m_wc(0, 0) = m_wm(0, 0) + 1.0 - merwe->alpha*merwe->alpha + merwe->beta;

	for (int i = 1; i < NUM_SIGMA_POINTS; i++)
	{
		m_wm(0, i) = 1.0 / (2.0*(NUM_STATE_VARS + merwe->lambda));
		m_wc(0, i) = m_wm(0, i);
	}
}

//////////////////////////////////////////////////////////
/*CLASS: UnscentedKalmanFilter_64 */
//////////////////////////////////////////////////////////
/*------------------------------------
* Constructor/Destructor
*------------------------------------*/
UnscentedKalmanFilter_64::UnscentedKalmanFilter_64()
{
	errorLed_Assigned = false;
}

UnscentedKalmanFilter_64::~UnscentedKalmanFilter_64()
{
}

/*------------------------------------
* Public Functions
*------------------------------------*/
void UnscentedKalmanFilter_64::initialize(UKF_InputMatrices_64& input_matrix, MerweConstants_64& input_merwe, boost::shared_ptr<GPIOClass> errLedPin)
{
	if (errLedPin != NULL)
		assignErrorLed(errLedPin);

	dim_assert(input_matrix.A.rows(), NUM_STATE_VARS, input_matrix.A.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.X.rows(), NUM_STATE_VARS, input_matrix.X.cols(), 1);
	dim_assert(input_matrix.B.rows(), NUM_STATE_VARS, input_matrix.B.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.U.rows(), NUM_STATE_VARS, input_matrix.U.cols(), 1);
	dim_assert(input_matrix.P.rows(), NUM_STATE_VARS, input_matrix.P.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.Q.rows(), NUM_STATE_VARS, input_matrix.Q.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.R.rows(), NUM_STATE_VARS, input_matrix.R.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.H.rows(), NUM_STATE_VARS, input_matrix.H.cols(), NUM_STATE_VARS);

	mConfig = &input_matrix;
	merwe = &input_merwe;

	/* Initialize the Xk and Pk matrices */
	mResult.Xk = mConfig->X;
	mResult.Pk = mConfig->P;
}

void UnscentedKalmanFilter_64::assignErrorLed(boost::shared_ptr<GPIOClass> led)
{
	ledPin = led;
	ledPin->mode(OUTPUT_PP);
	ledPin->write(LOW);

	errorLed_Assigned = true;
}

mdColumn UnscentedKalmanFilter_64::iterate(mdColumn &measured_data)
{
	/*------------------------------
	* Predict Step
	*------------------------------*/
	// Generate the sigma points and weights
	calc_MerweSigma64(mResult.Xk, mResult.Pk, mResult.Wc, mResult.Wm, mResult.Xsig);

	// Project the sigma points through the given model Ax+Bu+Wn
	for(int i = 0 ; i < NUM_SIGMA_POINTS ; i++)
		mResult.Ysig.col(i) = mConfig->A*mResult.Xsig.col(i) + mConfig->B*measured_data;

	// Compute the mean and covariance of the prediction with the Unscented Transform
	calc_UT64(mResult.Ysig, mConfig->Q, mResult.Wc, mResult.Wm, mResult.Xp, mResult.Pp);

	/*------------------------------
	* Update Step
	*------------------------------*/
	// Convert the sigma points into a measurement
	mResult.Zsig = mConfig->H*mResult.Ysig;

	// Compute the mean and covariance of that measurement
	calc_UT64(mResult.Zsig, mConfig->R, mResult.Wc, mResult.Wm, mResult.Xm, mResult.Pm);

	// Compute the residual between measured and predicted
	mResult.Yk = measured_data - mResult.Xm;

	// Compute the cross covariance of the state and measurement
	calc_XCov64(mResult.Xp, mResult.Ysig, mResult.Xm, mResult.Zsig, mResult.Wc, mResult.Pxm);

	// Compute the Kalman Gain
	mResult.Pm_inv = mResult.Pm.inverse();
	mResult.K = mResult.Pxm*mResult.Pm_inv;

	//Compute the new state estimate and covariance
	mdSquare kT = mResult.K.transpose();

	mResult.Xk = mResult.Xp + mResult.K*mResult.Yk;
	mResult.Pk = mResult.Pp - mResult.K*mResult.Pm*kT;

	return mResult.Xk;
}
/*------------------------------------
* Private Functions
*------------------------------------*/
bool UnscentedKalmanFilter_64::dim_assert(size_t row_act, size_t row_exp, size_t col_act, size_t col_exp)
{
	if ((row_act != row_exp) || (col_act != col_exp))
	{
		std::string msg("Failed matrix dimension assert.");
		error_handler(msg);
	}
	return true;
}

void UnscentedKalmanFilter_64::error_handler(std::string errorMessage)
{
	error_message = errorMessage;
	for (;;)
	{
		if (errorLed_Assigned)
		{
			ledPin->write(HIGH);
			HAL_Delay(250);
			ledPin->write(LOW);
			HAL_Delay(250);
		}
	}
}

void UnscentedKalmanFilter_64::calc_XCov64(mdColumn meanX, mdSigma sigmaX, mdColumn meanM, mdSigma sigmaM, mdWeight wc, mdSquare &cross_covariance)
{
	Eigen::Matrix<double, NUM_STATE_VARS, 1> tempM;
	Eigen::Matrix<double, 1, NUM_STATE_VARS> tempMT;

	for (int i = 0; i < NUM_SIGMA_POINTS; i++)
	{
		tempM = sigmaM.col(i) - meanM;
		tempMT = tempM.transpose();

		cross_covariance += wc(0, i)*(sigmaX.col(i) - meanX)*tempMT;
	}
}

void UnscentedKalmanFilter_64::calc_UT64(mdSigma sigmaPts, mdSquare noise_covariance, mdWeight wc, mdWeight wm, mdColumn &X, mdSquare &P)
{
	/* Calculate the new mean */
	for (int i = 0; i < NUM_SIGMA_POINTS; i++)
		X += wm(0, i)*sigmaPts.col(i);

	/* Calculate the new covariance */
	Eigen::Matrix<double, NUM_STATE_VARS, 1> temp;
	Eigen::Matrix<double, 1, NUM_STATE_VARS> tempT;

	for (int i = 0; i < NUM_SIGMA_POINTS; i++)
	{
		temp = sigmaPts.col(i) - X;
		tempT = temp.transpose();

		P += wc(0, i)*(temp)*tempT;
	}

	P += noise_covariance;
}

void UnscentedKalmanFilter_64::calc_MerweSigma64(mdColumn m_X, mdSquare m_P, mdWeight &m_wc, mdWeight &m_wm, mdSigma &m_xsig)
{
	/* First, find the Cholesky Decomposition of m_P, the state covariance matrix */
	Eigen::LDLT<mdSquare> lltOfP(m_P*(NUM_STATE_VARS + merwe->lambda));
	merwe->cholesky = lltOfP.matrixL();

	/* Calculate the sigma points */
	m_xsig.col(0) = m_X;
	for (int i = 1; i < NUM_SIGMA_POINTS; i++)
	{
		if (i <= NUM_STATE_VARS)
			m_xsig.col(i) = m_X + merwe->cholesky.col(i - 1);

		if (i > NUM_STATE_VARS)
			m_xsig.col(i) = m_X - merwe->cholesky.col(i - 1 - NUM_STATE_VARS);
	}

	/* Calculate the weights */
	//TODO! THESE ARE CONSTANTS. MOVE OUT TO CLASS VARS.
	m_wm(0, 0) = merwe->lambda / (NUM_STATE_VARS + merwe->lambda);
	m_wc(0, 0) = m_wm(0, 0) + 1.0 - merwe->alpha*merwe->alpha + merwe->beta;

	for (int i = 1; i < NUM_SIGMA_POINTS; i++)
	{
		m_wm(0, i) = 1.0 / (2.0*(NUM_STATE_VARS + merwe->lambda));
		m_wc(0, i) = m_wm(0, i);
	}
}

//////////////////////////////////////////////////////////
/*CLASS: UnscentedKalmanFilter_Opt */
//////////////////////////////////////////////////////////
/*------------------------------------
* Constructor/Destructor
*------------------------------------*/
UnscentedKalmanFilter_Opt::UnscentedKalmanFilter_Opt()
{
	errorLed_Assigned = false;
}

UnscentedKalmanFilter_Opt::~UnscentedKalmanFilter_Opt()
{
}

/*------------------------------------
* Public Functions
*------------------------------------*/
void UnscentedKalmanFilter_Opt::initialize(UKF_InputMatrices_Opt& input_matrix, MerweConstants_Opt& input_merwe, boost::shared_ptr<GPIOClass> errLedPin)
{
	if (errLedPin != NULL)
		assignErrorLed(errLedPin);

	dim_assert(input_matrix.A.rows(), NUM_STATE_VARS, input_matrix.A.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.X.rows(), NUM_STATE_VARS, input_matrix.X.cols(), 1);
	dim_assert(input_matrix.B.rows(), NUM_STATE_VARS, input_matrix.B.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.U.rows(), NUM_STATE_VARS, input_matrix.U.cols(), 1);
	dim_assert(input_matrix.P.rows(), NUM_STATE_VARS, input_matrix.P.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.Q.rows(), NUM_STATE_VARS, input_matrix.Q.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.R.rows(), NUM_STATE_VARS, input_matrix.R.cols(), NUM_STATE_VARS);
	dim_assert(input_matrix.H.rows(), NUM_STATE_VARS, input_matrix.H.cols(), NUM_STATE_VARS);

	mConfig = &input_matrix;
	merwe = &input_merwe;

	/* Initialize the Xk and Pk matrices */
	mResult.Xk_double = mConfig->X;
	mResult.Pk_double = mConfig->P;
}

void UnscentedKalmanFilter_Opt::assignErrorLed(boost::shared_ptr<GPIOClass> led)
{
	ledPin = led;
	ledPin->mode(OUTPUT_PP);
	ledPin->write(LOW);

	errorLed_Assigned = true;
}

mdColumn UnscentedKalmanFilter_Opt::iterate(mdColumn &measured_data)
{
	//////////////////////////////////////////////////////////////////////////
	// Items marked with a **** above it still need to be optimized
	// Items marked with a ---- above it are pretty decently optimized
	// Items marked with a ???? above it are optimized but untested
	//////////////////////////////////////////////////////////////////////////

	/*------------------------------
	* Predict Step
	*------------------------------*/
	// Generate the sigma points and weights
	// ****
	// Double: 888uS
	// Single: Not worth it/likely unstable
	calc_MerweSigmaOpt(mResult.Xk_double, mResult.Pk_double, mResult.Wc_double, mResult.Wm_double, mResult.Xsig_double);

	//Typecasting for later
	mResult.Xsig_single = mResult.Xsig_double.cast<float>();
	mResult.Wm_single = mResult.Wm_double.cast<float>();
	mResult.Wc_single = mResult.Wc_double.cast<float>();
	mfColumn measured_data_single = measured_data.cast<float>();

	// Project the sigma points through the given model
	// ----
	// Double: 3474 uS
	// Single:  282 uS (including cast at the end)
	for(int i = 0 ; i < NUM_SIGMA_POINTS ; i++)
		mResult.Ysig_single.col(i) = mConfig->A_single*mResult.Xsig_single.col(i) + mConfig->B_single*measured_data_single;
	
	mResult.Ysig_double = mResult.Ysig_single.cast<double>();

	// Compute the mean and covariance of the prediction with the Unscented Transform
	// ****
	// Double: 2812 uS
	// Single:  197 uS (including two casts at end)
	calc_UTOpt_MIXEDTEST(mResult.Ysig_single, mConfig->Q_single, mResult.Wc_single, mResult.Wm_single, mResult.Xp_single, mResult.Pp_single);

	mResult.Xp_double = mResult.Xp_single.cast<double>();
	mResult.Pp_double = mResult.Pp_single.cast<double>();

	//============================================================================
	//EVERYTHING BELOW BLOWS UP IF SET IN 32BIT MODE 
	/*------------------------------
	* Update Step
	*------------------------------*/
	// Convert the sigma points into a measurement
	// ----
	mResult.Zsig_double = mConfig->H*mResult.Ysig_double;

	// Compute the mean and covariance of that measurement
	// ****
	// Double: ?
	// Single: Currently only stable under double precision
	calc_UTOpt(mResult.Zsig_double, mConfig->R, mResult.Wc_double, mResult.Wm_double, mResult.Xm_double, mResult.Pm_double);

	// Compute the residual between measured and predicted
	mResult.Yk = measured_data - mResult.Xm_double;

	// Compute the cross covariance of the state and measurement
	// ****
	// Double: ?
	// Single: Likely Unstable
	calc_XCovOpt(mResult.Xp_double, mResult.Ysig_double, mResult.Xm_double, mResult.Zsig_double, mResult.Wc_double, mResult.Pxm_double);

	// Compute the Kalman Gain
	// ****
	// Double: 3621 uS
	// Single: Unstable
	mResult.Pm_inv_double = mResult.Pm_double.inverse();
	mResult.K_double = mResult.Pxm_double*mResult.Pm_inv_double;

	//Compute the new state estimate and covariance
	// **** 22 uS
	mdSquare kT = mResult.K_double.transpose();

	// **** 2680 uS
	mResult.Xk_double = mResult.Xp_double + mResult.K_double*mResult.Yk;
	mResult.Pk_double = mResult.Pp_double - mResult.K_double*mResult.Pm_double*kT;

	return mResult.Xk_double;
}
/*------------------------------------
* Private Functions
*------------------------------------*/
bool UnscentedKalmanFilter_Opt::dim_assert(size_t row_act, size_t row_exp, size_t col_act, size_t col_exp)
{
	if ((row_act != row_exp) || (col_act != col_exp))
	{
		std::string msg("Failed matrix dimension assert.");
		error_handler(msg);
	}
	return true;
}

void UnscentedKalmanFilter_Opt::error_handler(std::string errorMessage)
{
	error_message = errorMessage;
	for (;;)
	{
		if (errorLed_Assigned)
		{
			ledPin->write(HIGH);
			HAL_Delay(250);
			ledPin->write(LOW);
			HAL_Delay(250);
		}

		//Maybe do a serial print here of the message?
	}
}

void UnscentedKalmanFilter_Opt::calc_XCovOpt(mdColumn meanX, mdSigma sigmaX, mdColumn meanM, mdSigma sigmaM, mdWeight wc, mdSquare &cross_covariance)
{
	Eigen::Matrix<double, NUM_STATE_VARS, 1> tempM;
	Eigen::Matrix<double, 1, NUM_STATE_VARS> tempMT;

	for (int i = 0; i < NUM_SIGMA_POINTS; i++)
	{
		tempM = sigmaM.col(i) - meanM;
		tempMT = tempM.transpose();

		cross_covariance += wc(0, i)*(sigmaX.col(i) - meanX)*tempMT;
	}
}

void UnscentedKalmanFilter_Opt::calc_UTOpt(mdSigma sigmaPts, mdSquare noise_covariance, mdWeight wc, mdWeight wm, mdColumn &X, mdSquare &P)
{
	/* Calculate the new mean */
	for (int i = 0; i < NUM_SIGMA_POINTS; i++)
		X += wm(0, i)*sigmaPts.col(i);

	/* Calculate the new covariance */
	Eigen::Matrix<double, NUM_STATE_VARS, 1> temp;
	Eigen::Matrix<double, 1, NUM_STATE_VARS> tempT;

	for (int i = 0; i < NUM_SIGMA_POINTS; i++)
	{
		temp = sigmaPts.col(i) - X;
		tempT = temp.transpose();

		P += wc(0, i)*(temp)*tempT;
	}

	P += noise_covariance;
}

void UnscentedKalmanFilter_Opt::calc_UTOpt_MIXEDTEST(mfSigma sigmaPts, mfSquare noise_covariance, mfWeight wc, mfWeight wm, mfColumn &X, mfSquare &P)
{
	/* Calculate the new mean */
	for (int i = 0; i < NUM_SIGMA_POINTS; i++)
		X += wm(0, i)*sigmaPts.col(i);

	/* Calculate the new covariance */
	Eigen::Matrix<float, NUM_STATE_VARS, 1> temp;
	Eigen::Matrix<float, 1, NUM_STATE_VARS> tempT;

	for (int i = 0; i < NUM_SIGMA_POINTS; i++)
	{
		temp = sigmaPts.col(i) - X;
		tempT = temp.transpose();

		P += wc(0, i)*(temp)*tempT;
	}

	P += noise_covariance;
}

void UnscentedKalmanFilter_Opt::calc_MerweSigmaOpt(mdColumn m_X, mdSquare m_P, mdWeight &m_wc, mdWeight &m_wm, mdSigma &m_xsig)
{
	/* First, find the Cholesky Decomposition of m_P, the state covariance matrix */
	Eigen::LDLT<mdSquare> lltOfP(m_P*(NUM_STATE_VARS + merwe->lambda));
	merwe->cholesky = lltOfP.matrixL();

	/* Calculate the sigma points */
	m_xsig.col(0) = m_X;
	for (int i = 1; i < NUM_SIGMA_POINTS; i++)
	{
		if (i <= NUM_STATE_VARS)
			m_xsig.col(i) = m_X + merwe->cholesky.col(i - 1);

		if (i > NUM_STATE_VARS)
			m_xsig.col(i) = m_X - merwe->cholesky.col(i - 1 - NUM_STATE_VARS);
	}

	/* Calculate the weights */
	m_wm(0, 0) = merwe->lambda / (NUM_STATE_VARS + merwe->lambda);
	m_wc(0, 0) = m_wm(0, 0) + 1.0 - merwe->alpha*merwe->alpha + merwe->beta;

	for (int i = 1; i < NUM_SIGMA_POINTS; i++)
	{
		m_wm(0, i) = 1.0 / (2.0*(NUM_STATE_VARS + merwe->lambda));
		m_wc(0, i) = m_wm(0, i);
	}

	//For the most part this is pretty fast compared to the rest of the alg: 888uS
}