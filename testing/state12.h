#pragma once
#ifndef STATE12_H_
#define STATE12_H_
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

extern void runState12();
#endif