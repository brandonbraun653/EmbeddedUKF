/* Language Dependencies */
#include <stdlib.h>
#include <stdint.h>

/* HAL Dependencies */
#include <stm32f7xx_hal.h>
#include <stm32_hal_legacy.h>

/* Thor Dependencies*/
#include "thor.h"

/* Visual GDB Dependencies */
#include "SysprogsProfiler.h"
#include "gdb.h"

/* Local Dependencies (Testing) */
#include "imu.h"
#include "state3.h"
#include "state6.h"
#include "state9.h"
#include "state12.h"

//#define TEST_3_STATE
//#define TEST_6_STATE
//#define TEST_9_STATE
#define TEST_12_STATE

int main(void)
{
	HAL_Init();
	ThorSystemClockConfig();
	InitializeSamplingProfiler();
	InitializeInstrumentingProfiler();
	
	/* Each of these tests have their own loop */
#ifdef TEST_3_STATE
	runState3();
#endif
	
#ifdef TEST_6_STATE
	runState6();
#endif
	
#ifdef TEST_9_STATE
	runState9();
#endif
	
#ifdef TEST_12_STATE
	runState12();
#endif
	
}

