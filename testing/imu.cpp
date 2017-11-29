#include "imu.h"


/*------------------------------------
* Constructor/Destructor
*------------------------------------*/
IMU::IMU()
{
	UKF = boost::make_shared<UnscentedKalmanFilter<double, STATE_SIZE>>(SPEED_UP);	
}

IMU::~IMU()
{
	
}

/*------------------------------------
* Public Functions
*------------------------------------*/
void IMU::initialize()
{
	/* UKF Init Things */
	UKF->initialize();
	
	/* Sensor Init Things */
	sensor.initialize();
	sensor.setCS_PinPort(ACCEL, PIN_7, GPIOF);
	sensor.setCS_PinPort(GYRO, PIN_6, GPIOF);
	
}

void IMU::getPitchRollYaw(double& pitch, double& roll, double& yaw)
{
	/* Quick and dirty algorithm taken from:
	 * https://theccontinuum.com/2012/09/24/arduino-imu-pitch-roll-from-accelerometer/ */
	sensor.read(ACCEL);
	
	double Xg = sensor.accel_data.x;
	double Yg = sensor.accel_data.y;
	double Zg = sensor.accel_data.z;
	
	//Low pass filter
	fXg = Xg * alpha + (fXg * (1.0 - alpha));
	fYg = Yg * alpha + (fYg * (1.0 - alpha));
	fZg = Zg * alpha + (fZg * (1.0 - alpha));
	
	//Roll & Pitch Equations
	roll = (atan2(-fYg, fZg)*180.0) / PI;
	pitch = (atan2(fXg, (sqrt(fYg*fYg + fZg*fZg)))*180.0) / PI;
	yaw = -1.0;
}

/*------------------------------------
* Private Functions
*------------------------------------*/


