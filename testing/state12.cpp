#include "state12.h"

#define DOF 3
#define NUM_STATE_VARS 12
typedef double state12Type;

void runState12()
{
	/* All ok Status LED */
	GPIOClass ledPin(GPIOB, PIN_7, HIGH_SPD, NOALTERNATE);
	ledPin.mode(OUTPUT_PP);
	ledPin.write(LOW);
	
	/*------------------------------------
	* Setup Parameters
	*------------------------------------*/
	state12Type dt = 0.02;     //Sampling rate for the accelerometer in sec

	/*------------------------------------
	* Setup the UKF
	*------------------------------------*/
	UnscentedKalmanFilter<state12Type, NUM_STATE_VARS> UKF(true);
	
	/* Merwe Sigma Points Constants Setup */
	UKF.Merwe->alpha = 0.5;
	UKF.Merwe->beta = 2.0;
	UKF.Merwe->kappa = 1.0;
	UKF.Merwe->lambda = (UKF.Merwe->alpha * UKF.Merwe->alpha)*(NUM_STATE_VARS + UKF.Merwe->kappa) - NUM_STATE_VARS;

	/* Matrix Initialization */
	// State Matrix: X == Already initialized to ZERO
	// Input Transition: B == Already initialized to ZERO

	// State Uncertainty/Covariance Matrix: P ----------------------------
	state12Type iPX = 0.25, iPV = 0.25, iPA = 1.25;

	Eigen::Matrix<state12Type, 3, 3> pBlock;
	pBlock << iPX*iPX, 0.0, 0.0,
		      0.0, iPV*iPV, 0.0,
		      0.0, 0.0, iPA*iPA;

	UKF.Input->P.block<3, 3>(0, 0) = pBlock;
	UKF.Input->P.block<3, 3>(3, 3) = pBlock;
	UKF.Input->P.block<3, 3>(6, 6) = pBlock;
	UKF.Input->P.block<3, 3>(9, 9) = pBlock;

	// State Transition Matrix: A ----------------------------------------
	Eigen::Matrix<state12Type, 3, 3> physBlock;
	physBlock << 1.0, dt, 0.5*dt*dt,
		         0.0, 1.0, dt,
		         0.0, 0.0, 1.0;

	UKF.Input->A.block<3, 3>(0, 0) = physBlock;
	UKF.Input->A.block<3, 3>(3, 3) = physBlock;
	UKF.Input->A.block<3, 3>(6, 6) = physBlock;
	UKF.Input->A.block<3, 3>(9, 9) = physBlock;
	
	// Process Noise: Q --------------------------------------------------
	Eigen::Matrix<state12Type, 3, 3> noiseBlock;
	noiseBlock << 5.00e-7, 1.25e-5, 1.66e-4,
				  1.25e-5, 3.33e-4, 5.00e-3,
				  1.66e-4, 5.00e-3, 1.00e-6;
	
	/* I think the noise here needs to be modeled just a bit better.... */
	UKF.Input->Q.block<3, 3>(0, 0) = noiseBlock;
	UKF.Input->Q.block<3, 3>(3, 3) = noiseBlock;
	UKF.Input->Q.block<3, 3>(6, 6) = noiseBlock;
	UKF.Input->Q.block<3, 3>(9, 9) = noiseBlock;
	
	// Measurement Selection: H ------------------------------------------
	UKF.Input->H.setIdentity(UKF.Input->H.rows(), UKF.Input->H.cols());

	// Measurement Uncertainty/Covariance: R -----------------------------
	state12Type iRX = 1.25, iRV = 1.25, iRA = 0.25;

	Eigen::Matrix<state12Type, 3, 3> rBlock;
	rBlock << iRX*iRX, 0.0, 0.0,
		      0.0, iRV*iRV, 0.0,
		      0.0, 0.0, iRA*iRA;

	UKF.Input->R.block<3, 3>(0, 0) = rBlock;
	UKF.Input->R.block<3, 3>(3, 3) = rBlock;
	UKF.Input->R.block<3, 3>(6, 6) = rBlock;
	UKF.Input->R.block<3, 3>(9, 9) = rBlock;

	/* Start up the UKF Algorithm */
	UKF.initialize();

	/*------------------------------------
	* Setup the Accel/Gyro/Mag
	*------------------------------------*/
	LSM9DS0 sensor;
	sensor.setCS_PinPort(ACCEL, PIN_7, GPIOF);
	sensor.setCS_PinPort(GYRO, PIN_6, GPIOF);
	bool liveData = sensor.initialize(true);

	/*------------------------------------
	* Setup the Runtime Data Buffers
	*------------------------------------*/
//	volatile state12Type rawAccelX, rawAccelY, rawAccelZ, predictedAccelX, predictedAccelY, predictedAccelZ;
//	volatile state12Type rawVelX, rawVelY, rawVelZ, predictedVelX, predictedVelY, predictedVelZ;
//	volatile state12Type rawPosX, rawPosY, rawPosZ, predictedPosX, predictedPosY, predictedPosZ;
	Eigen::Matrix<state12Type, NUM_STATE_VARS, 1> mData, xOutData;
	Eigen::Matrix<state12Type, DOF, 1> ak, vk, xk, aklast, vklast, xklast;
	ak.setZero(DOF, 1);
	vk.setZero(DOF, 1);
	xk.setZero(DOF, 1);
	aklast.setZero(DOF, 1);
	vklast.setZero(DOF, 1);
	xklast.setZero(DOF, 1);
	mData.setZero(NUM_STATE_VARS, 1);  		//Measured data
	xOutData.setZero(NUM_STATE_VARS, 1);  	//Filtered output from UKF

	/*------------------------------------
	* Setup Serial
	*------------------------------------*/
	uart7.begin();
	uart7.setTxModeDMA();

	std::string outputMsg;

	/*------------------------------------
	* Run
	*------------------------------------*/
	for (;;)
	{
		if (true)
		{
			sensor.read(ACCEL);    //Really only using this for now

			ak(0, 0) = sensor.accel_data.x;
			ak(1, 0) = sensor.accel_data.y;
			ak(2, 0) = sensor.accel_data.z;
		}
		else
		{
			/* I wonder how I could do a rng on here. That would be awesome for creating fake noise */
			ak(0, 0) = 0.0;
			ak(1, 0) = 0.0;
			ak(2, 0) = 0.5;
		}
		
		/* Integrate Accelerometer data into velocity and position measurement */
		vk = vklast + 0.5*(aklast + ak)*dt; vklast = vk;
		xk = xklast + 0.5*(vklast + vk)*dt; xklast = xk;

		/* Format the data into something the UKF expects */
		mData << xk(0, 0), vk(0, 0), ak(0, 0),      // Pos, vel, accel in X
			     xk(1, 0), vk(1, 0), ak(1, 0),      // Pos, vel, accel in Y
				 xk(2, 0), vk(2, 0), ak(2, 0),      // Pos, vel, accel in Z
				 xk(2, 0), vk(2, 0), ak(2, 0);		// Purely just to take up space.

		/* Iterate once */
		xOutData = UKF.iterate(mData);
	}
}