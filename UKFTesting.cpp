/* Language Dependencies */
#include <stdlib.h>
#include <stdint.h>
#include <string>

/* Eigen Dependencies */
#include <eigen/Eigen>
#include <eigen/StdVector>

/* Boost Dependencies */
#include <boost/smart_ptr.hpp>

/* HAL Dependencies */
#include <stm32f7xx_hal.h>
#include <stm32_hal_legacy.h>

/* Thor Dependencies*/
#include "thor.h"
#include "uart.h"

/* Project Dependencies */
#include "ukf.hpp"
#include "ukf_v2.hpp"
#include "stm32f7_lsm9ds0.h"

/* Visual GDB Dependencies */
#include "SysprogsProfiler.h"
#include "gdb.h"

#define DOF 3
//#define MODE_32
#define MODE_64
//#define MODE_OPT


/* Define the data type used in the UKF. Could be int, float, or double */
#ifdef MODE_32
typedef float mDataType;
#endif

#ifdef MODE_64
typedef double mDataType;
#endif

#ifdef MODE_OPT
typedef double mDataType;
#endif



volatile mDataType dummyAccelX, dummyAccelY, dummyAccelZ;

int main(void)
{
	HAL_Init();
	SystemClockConfig();
	InitializeSamplingProfiler();
	InitializeInstrumentingProfiler();
	
	/* All ok Status LED */
	GPIOClass ledPin(GPIOB, PIN_7, HIGH_SPD, NOALTERNATE);
	ledPin.mode(OUTPUT_PP);
	ledPin.write(LOW);

	/* Error LED */
	boost::shared_ptr<GPIOClass> oops_led = boost::make_shared<GPIOClass>(GPIOB, PIN_14, HIGH_SPD, NOALTERNATE);
	
	UnscentedKalmanFilter<float, 3> ukfTest;
	ukfTest.initialize();
	
	Eigen::Matrix<float, 3, 1> testM;
	
	ukfTest.iterate(testM);
	
	/*------------------------------------
	* Setup Parameters
	*------------------------------------*/
	mDataType dt = 0.05;  //Sampling rate for the accelerometer in sec

	/*------------------------------------
	* Setup the UKF
	*------------------------------------*/
#ifdef MODE_32
	UKF_InputMatrices_32 ukf_matrix;
	MerweConstants_32 merwe;
#endif

#ifdef MODE_64
	UKF_InputMatrices_64 ukf_matrix;
	MerweConstants_64 merwe;
#endif

#ifdef MODE_OPT
	UKF_InputMatrices_Opt ukf_matrix;
	MerweConstants_Opt merwe;
#endif 

#if defined(MODE_32) || defined(MODE_64)
	
	/* Merwe Sigma Points Constants Setup */
	merwe.alpha = 0.5;
	merwe.beta = 2.0;
	merwe.kappa = 1.0;
	merwe.lambda = (merwe.alpha * merwe.alpha)*(NUM_STATE_VARS + merwe.kappa) - NUM_STATE_VARS;

	/* Matrix Initialization */
	// State Matrix: X == Already initialized to ZERO
	// Input Transition: B == Already initialized to ZERO
	// Input Matrix: U == Already initialized to ZERO

	// State Uncertainty/Covariance Matrix: P ----------------------------
	mDataType iPX = 10.0, iPV = 5.0, iPA = 0.25;

	Eigen::Matrix<mDataType, 3, 3> pBlock;
	pBlock << iPX*iPX, 0.0, 0.0,
		      0.0, iPV*iPV, 0.0,
		      0.0, 0.0, iPA*iPA;

	ukf_matrix.P.block<3, 3>(0, 0) = pBlock;
	ukf_matrix.P.block<3, 3>(3, 3) = pBlock;
	ukf_matrix.P.block<3, 3>(6, 6) = pBlock;

	// State Transition Matrix: A ----------------------------------------
	Eigen::Matrix<mDataType, 3, 3> physBlock;
	physBlock << 1.0, dt, 0.5*dt*dt,
		         0.0, 1.0, dt,
		         0.0, 0.0, 1.0;

	ukf_matrix.A.block<3, 3>(0, 0) = physBlock;
	ukf_matrix.A.block<3, 3>(3, 3) = physBlock;
	ukf_matrix.A.block<3, 3>(6, 6) = physBlock;
	
	

	// Process Noise: Q --------------------------------------------------
	Eigen::Matrix<mDataType, 3, 3> noiseBlock;
	noiseBlock << 5.00e-7, 1.25e-5, 1.66e-4,
				  1.25e-5, 3.33e-4, 5.00e-3,
				  1.66e-4, 5.00e-3, 1.00e-3;

	ukf_matrix.Q.block<3, 3>(0, 0) = noiseBlock;
	ukf_matrix.Q.block<3, 3>(3, 3) = noiseBlock;
	ukf_matrix.Q.block<3, 3>(6, 6) = noiseBlock;
	
	ukf_matrix.Q = ukf_matrix.Q * 1.0;

	// Measurement Selection: H ------------------------------------------
	ukf_matrix.H.setIdentity(ukf_matrix.H.rows(), ukf_matrix.H.cols());

	// Measurement Uncertainty/Covariance: R -----------------------------
	mDataType iRX = 15.0, iRV = 10.0, iRA = 0.25;

	Eigen::Matrix<mDataType, 3, 3> rBlock;
	rBlock << iRX*iRX, 0.0, 0.0,
		      0.0, iRV*iRV, 0.0,
		      0.0, 0.0, iRA*iRA;

	ukf_matrix.R.block<3, 3>(0, 0) = rBlock;
	ukf_matrix.R.block<3, 3>(3, 3) = rBlock;
	ukf_matrix.R.block<3, 3>(6, 6) = rBlock;
#endif

#ifdef MODE_OPT
	/* Merwe Sigma Points Constants Setup */
	merwe.alpha = 0.5;
	merwe.beta = 2.0;
	merwe.kappa = 1.0;
	merwe.lambda = (merwe.alpha * merwe.alpha)*(NUM_STATE_VARS + merwe.kappa) - NUM_STATE_VARS;

	/* Matrix Initialization */
	// State Matrix: X == Already initialized to ZERO
	// Input Transition: B == Already initialized to ZERO
	// Input Matrix: U == Already initialized to ZERO

	// State Uncertainty/Covariance Matrix: P ----------------------------
	double iPX = 10.0, iPV = 5.0, iPA = 0.25;

	Eigen::Matrix<double, 3, 3> pBlock;
	pBlock << iPX*iPX, 0.0, 0.0,
		      0.0, iPV*iPV, 0.0,
		      0.0, 0.0, iPA*iPA;

	ukf_matrix.P.block<3, 3>(0, 0) = pBlock;
	ukf_matrix.P.block<3, 3>(3, 3) = pBlock;
	ukf_matrix.P.block<3, 3>(6, 6) = pBlock;

	// State Transition Matrix: A ----------------------------------------
	Eigen::Matrix<double, 3, 3> physBlock;
	physBlock << 1.0, dt, 0.5*dt*dt,
		         0.0, 1.0, dt,
		         0.0, 0.0, 1.0;

	ukf_matrix.A.block<3, 3>(0, 0) = physBlock;
	ukf_matrix.A.block<3, 3>(3, 3) = physBlock;
	ukf_matrix.A.block<3, 3>(6, 6) = physBlock;
	
	ukf_matrix.A_single = ukf_matrix.A.cast<float>();

	// Process Noise: Q --------------------------------------------------
	Eigen::Matrix<double, 3, 3> noiseBlock;
	noiseBlock << 5.00e-7, 1.25e-5, 1.66e-4,
				  1.25e-5, 3.33e-4, 5.00e-3,
				  1.66e-4, 5.00e-3, 1.00e-3;

	ukf_matrix.Q.block<3, 3>(0, 0) = noiseBlock;
	ukf_matrix.Q.block<3, 3>(3, 3) = noiseBlock;
	ukf_matrix.Q.block<3, 3>(6, 6) = noiseBlock;
	
	ukf_matrix.Q_single = ukf_matrix.Q.cast<float>();
	
	// Measurement Selection: H ------------------------------------------
	ukf_matrix.H.setIdentity(ukf_matrix.H.rows(), ukf_matrix.H.cols());
	ukf_matrix.H_single = ukf_matrix.H.cast<float>();

	// Measurement Uncertainty/Covariance: R -----------------------------
	double iRX = 15.0, iRV = 10.0, iRA = 0.25;

	Eigen::Matrix<double, 3, 3> rBlock;
	rBlock << iRX*iRX, 0.0, 0.0,
		      0.0, iRV*iRV, 0.0,
		      0.0, 0.0, iRA*iRA;

	ukf_matrix.R.block<3, 3>(0, 0) = rBlock;
	ukf_matrix.R.block<3, 3>(3, 3) = rBlock;
	ukf_matrix.R.block<3, 3>(6, 6) = rBlock;
	
	ukf_matrix.R_single = ukf_matrix.R.cast<float>();
#endif 

	/* Start up the UKF Algorithm */
#ifdef MODE_32
	boost::shared_ptr<UnscentedKalmanFilter_32> UKF = boost::make_shared<UnscentedKalmanFilter_32>();
#endif 
	
#ifdef MODE_64
	boost::shared_ptr<UnscentedKalmanFilter_64> UKF = boost::make_shared<UnscentedKalmanFilter_64>();
#endif
	
#ifdef MODE_OPT
	boost::shared_ptr<UnscentedKalmanFilter_Opt> UKF = boost::make_shared<UnscentedKalmanFilter_Opt>();
#endif
	
	UKF->initialize(ukf_matrix, merwe, oops_led);


	/*------------------------------------
	* Setup the Accel/Gyro/Mag
	*------------------------------------*/
	LSM9DS0 sensor;
	bool liveData = false;
	sensor.setCS_PinPort(ACCEL, PIN_7, GPIOF);
	sensor.setCS_PinPort(GYRO, PIN_6, GPIOF);
	if (sensor.initialize(true))
		liveData = true;

	/*------------------------------------
	* Setup the Runtime Data Buffers
	*------------------------------------*/
#ifdef MODE_32
	mfColumn mData, xOutData;
#endif 
	
#ifdef MODE_64
	mdColumn mData, xOutData;
#endif 
	
#ifdef MODE_OPT
	mdColumn mData, xOutData;	
#endif

	
	Eigen::Matrix<mDataType, DOF, 1> ak, vk, xk, aklast, vklast, xklast;
	ak.setZero(DOF, 1);
	vk.setZero(DOF, 1);
	xk.setZero(DOF, 1);
	aklast.setZero(DOF, 1);
	vklast.setZero(DOF, 1);
	xklast.setZero(DOF, 1);
	mData.setZero(NUM_STATE_VARS, 1);  //Measured data
	xOutData.setZero(NUM_STATE_VARS, 1);  //Filtered output from UKF

	/*------------------------------------
	* Setup Serial
	*------------------------------------*/
	uart7.begin();
	uart7.setTxModeDMA();

	std::string outputMsg;

	for (;;)
	{
		ledPin.write(HIGH);
		HAL_Delay((uint32_t)(dt*1000.0));
		ledPin.write(LOW);

		if (true)
		{
			/* Read out a full set of data */
			sensor.read(ACCEL); //Really only using this for now
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
		mData << xk(0, 0), vk(0, 0), ak(0, 0),   // Pos, vel, accel in X
			     xk(1, 0), vk(1, 0), ak(1, 0),   // Pos, vel, accel in Y
				 xk(2, 0), vk(2, 0), ak(2, 0);   // Pos, vel, accel in Z

		/* Iterate once */
		xOutData = UKF->iterate(mData);
		dummyAccelX = xOutData(2, 0);
		dummyAccelY = xOutData(5, 0);
		dummyAccelZ = xOutData(8, 0);
		
		//HAL_Delay(100);
		//prettyPrint<mColumnf>(xOutData, "Matrix: UKF Data\n");
		
		/* Write the output data to serial */

		/*
		outputMsg = "Accel X: " + convert2String(sensor.accel_data.x) + "\t" +\
					"Accel Y: " + convert2String(sensor.accel_data.y) + "\t" +\
					"Accel Z: " + convert2String(sensor.accel_data.z) + "\n";
		*/
		
		outputMsg = convert2String(xOutData(8,0)) + "\n";
		uart7.write(outputMsg);
	}
}

