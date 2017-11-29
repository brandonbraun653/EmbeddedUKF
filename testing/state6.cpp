#include "state6.h"

#define DOF 3
#define NUM_STATE_VARS 6
typedef float state6Type;

void runState6()
{
	/* All ok Status LED */
	GPIOClass ledPin(GPIOB, PIN_7, HIGH_SPD, NOALTERNATE);
	ledPin.mode(OUTPUT_PP);
	ledPin.write(LOW);
	
	/*------------------------------------
	* Setup Parameters
	*------------------------------------*/
	state6Type dt = 0.02;     //Sampling rate for the accelerometer in sec

	/*------------------------------------
	* Setup the UKF
	*------------------------------------*/
	UnscentedKalmanFilter<state6Type, NUM_STATE_VARS> UKF;
	
	/* Merwe Sigma Points Constants Setup */
	UKF.Merwe->alpha = 0.5;
	UKF.Merwe->beta = 2.0;
	UKF.Merwe->kappa = 1.0;
	UKF.Merwe->lambda = (UKF.Merwe->alpha * UKF.Merwe->alpha)*(NUM_STATE_VARS + UKF.Merwe->kappa) - NUM_STATE_VARS;

	/* Matrix Initialization */
	// State Matrix: X == Already initialized to ZERO
	// Input Transition: B == Already initialized to ZERO

	// State Uncertainty/Covariance Matrix: P ----------------------------
	state6Type iPX = 0.25, iPV = 0.25, iPA = 1.25;

	Eigen::Matrix<state6Type, 3, 3> pBlock;
	pBlock << iPX*iPX, 0.0, 0.0,
		      0.0, iPV*iPV, 0.0,
		      0.0, 0.0, iPA*iPA;

	UKF.Input->P.block<3, 3>(0, 0) = pBlock;
	UKF.Input->P.block<3, 3>(3, 3) = pBlock;

	// State Transition Matrix: A ----------------------------------------
	Eigen::Matrix<state6Type, 3, 3> physBlock;
	physBlock << 1.0, dt, 0.5*dt*dt,
		         0.0, 1.0, dt,
		         0.0, 0.0, 1.0;

	UKF.Input->A.block<3, 3>(0, 0) = physBlock;
	UKF.Input->A.block<3, 3>(3, 3) = physBlock;

	// Process Noise: Q --------------------------------------------------
	Eigen::Matrix<state6Type, 3, 3> noiseBlock;
	noiseBlock << 5.00e-7, 1.25e-5, 1.66e-4,
				  1.25e-5, 3.33e-4, 5.00e-3,
				  1.66e-4, 5.00e-3, 1.00e-6;
	
	/* I think the noise here needs to be modeled just a bit better.... */
	UKF.Input->Q.block<3, 3>(0, 0) = noiseBlock;
	UKF.Input->Q.block<3, 3>(3, 3) = noiseBlock;

	// Measurement Selection: H ------------------------------------------
	UKF.Input->H.setIdentity(UKF.Input->H.rows(), UKF.Input->H.cols());

	// Measurement Uncertainty/Covariance: R -----------------------------
	state6Type iRX = 1.25, iRV = 1.25, iRA = 0.25;

	Eigen::Matrix<state6Type, 3, 3> rBlock;
	rBlock << iRX*iRX, 0.0, 0.0,
		      0.0, iRV*iRV, 0.0,
		      0.0, 0.0, iRA*iRA;

	UKF.Input->R.block<3, 3>(0, 0) = rBlock;
	UKF.Input->R.block<3, 3>(3, 3) = rBlock;

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
	Eigen::Matrix<state6Type, NUM_STATE_VARS, 1> mData, xOutData;
	Eigen::Matrix<state6Type, DOF, 1> ak, vk, xk, aklast, vklast, xklast;
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
		ledPin.write(HIGH);
		//HAL_Delay((uint32_t)(dt * 1000.0));
		ledPin.write(LOW);

		if (true)
		{
			sensor.read(ACCEL);     //Really only using this for now
			//sensor.read(GYRO); // Placeholder for time reasons
			//sensor.read(MAG);  // Placeholder for time reasons

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
		mData << xk(0, 0), vk(0, 0), ak(0, 0),        // Pos, vel, accel in X
			     xk(1, 0), vk(1, 0), ak(1, 0);       // Pos, vel, accel in Y

		/* Iterate once */
		xOutData = UKF.iterate(mData);
	}
}