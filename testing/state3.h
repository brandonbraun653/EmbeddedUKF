#pragma once
#ifndef STATE3_H_
#define STATE3_H_
/* Notes:
 * For the higher state versions of this, just do multiples of the basic
 * Newton's equations. Once you get to the 12th state version, recycle the
 * first 3 states into the 10-12 states. These tests are only meant to 
 * demonstrate the computational complexity of the problem. 
 * */

/* C/C++ Dependencies */
#include <stdlib.h>
#include <stdint.h>
#include <string>

/* Eigen Dependencies */
#include <eigen/Eigen>
#include <eigen/StdVector>

/* HAL Dependencies */
#include <stm32f7xx_hal.h>
#include <stm32_hal_legacy.h>

/* Thor Dependencies*/
#include "thor.h"
#include "uart.h"

/* Unscented Kalman Filter Stuff */
#include "ukf_v2.hpp"
#include "stm32f7_lsm9ds0.h"

extern void runState3();
#endif