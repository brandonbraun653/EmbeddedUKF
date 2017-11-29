#pragma once
#ifndef IMU_H_
#define IMU_H_
/* HAL Dependencies */
#include <stm32f7xx_hal.h>
#include <stm32_hal_legacy.h>

/* C++ Dependencies */
#include <stdlib.h>

/* Must be included after HAL libs. Compiler error results otherwise
 * due to #defines for hard FPU usage not being defined yet. This is 
 * device specific, but does work on an STM32F7 target.*/
#include <arm_math.h>

/* Eigen Dependencies */
#include <eigen/Eigen>

/* Boost Dependencies */
#include <boost/smart_ptr.hpp>
#include <boost/shared_ptr.hpp>

/* Other dependencies */
#include "ukf_v2.hpp"
#include "stm32f7_lsm9ds0.h"

#define STATE_SIZE 9
#define SPEED_UP true

class IMU
{
public:
	void initialize();
	
	void getPitchRollYaw(double& pitch, double& roll, double& yaw);
	
	IMU();
	~IMU();
	
private:
	LSM9DS0 sensor;
	boost::shared_ptr<UnscentedKalmanFilter<double, STATE_SIZE>> UKF; 
	
	/* TEMP VARS */
	double alpha = 0.5;
	double fXg = 0.0, fYg = 0.0, fZg = 0.0;
};

#endif