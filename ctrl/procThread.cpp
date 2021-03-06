#include "procThread.h"
#include "QtGui/qimage.h"
#include "stdio.h"
#include <iostream>
#include <ctime>

ProcThread::ProcThread(QObject *parent) :
		QThread(parent) {
	rawLeft = NULL;
	rawRight = NULL;
	rgbLeft = NULL;
	rgbRight = NULL;
	isInitialized = false;
	isRunning = false;
}
ProcThread::~ProcThread() {
	uninit();
}

void ProcThread::uninit() {
	leftGrabber.stopCapture();
	rightGrabber.stopCapture();
	leftGrabber.closeDev();
	rightGrabber.closeDev();

	if (rawLeft != NULL) {
		delete rawLeft;
		delete rawRight;
		delete rgbLeft;
		delete rgbRight;
		rawLeft = NULL;
		rawRight = NULL;
		rgbLeft = NULL;
		rgbRight = NULL;
	}

	isInitialized = false;

}

bool ProcThread::init(string calibFile, string leftDev, string rightDev,
		int reqWidth, int reqHeight) {

	if(isInitialized)
		uninit();

	if (!calibrationSettings.open(calibFile, cv::FileStorage::READ))
		return false;

	calibrationSettings["CM1"] >> calibData.CM1;
	calibrationSettings["CM2"] >> calibData.CM2;
	calibrationSettings["D1"] >> calibData.D1;
	calibrationSettings["D2"] >> calibData.D2;
	calibrationSettings["R"] >> calibData.R;
	calibrationSettings["T"] >> calibData.T;
	calibrationSettings["E"] >> calibData.E;
	calibrationSettings["F"] >> calibData.F;
	calibrationSettings["R1"] >> calibData.R1;
	calibrationSettings["R2"] >> calibData.R2;
	calibrationSettings["P1"] >> calibData.P1;
	calibrationSettings["P2"] >> calibData.P2;
	calibrationSettings["Q"] >> calibData.Q;

	bool initialized = leftGrabber.openAndInitDev(leftDev, reqWidth, reqHeight)
			&& rightGrabber.openAndInitDev(rightDev, reqWidth, reqHeight)
			&& leftGrabber.startCapture() && rightGrabber.startCapture();

	if (!initialized)
		printf("Can't open Device\n");

	if (!initialized || leftGrabber.getWidth() != rightGrabber.getWidth()
			|| leftGrabber.getHeight() != rightGrabber.getHeight()
			|| rightGrabber.getHeight() == 0 || leftGrabber.getHeight() == 0) {
		printf("Invalid capture sizes.\n");
		leftGrabber.stopCapture();
		rightGrabber.stopCapture();
		leftGrabber.closeDev();
		rightGrabber.closeDev();
	} else {
		imgSize = leftGrabber.getWidth() * leftGrabber.getHeight();

		rawLeft = new unsigned char[imgSize * 2];
		rawRight = new unsigned char[imgSize * 2];
		rgbLeft = new unsigned char[imgSize * 3];
		rgbRight = new unsigned char[imgSize * 3];
		initialized = matcher.initSystem(leftGrabber.getWidth(),
						leftGrabber.getHeight(),
						cCudaStereoMatcher::OP_INPUT_COLOR | cCudaStereoMatcher::OP_COLOR_MATCH);
		if (!initialized) {
			printf("Init of Stereomatcher failed.\n");
			return false;
		}

		cv::initUndistortRectifyMap(calibData.CM1, calibData.D1, calibData.R1,
				calibData.P1,
				cv::Size(leftGrabber.getWidth(), leftGrabber.getHeight()),
				CV_32FC1, undistData.map1x, undistData.map1y);
		cv::initUndistortRectifyMap(calibData.CM2, calibData.D2, calibData.R2,
				calibData.P2,
				cv::Size(leftGrabber.getWidth(), leftGrabber.getHeight()),
				CV_32FC1, undistData.map2x, undistData.map2y);
		isInitialized = true;
	}
	return initialized;
}
void ProcThread::updateSettings(int maxDisp, int consistTresh, int kernelSize) {
	matcher.updateSettings(kernelSize, maxDisp, consistTresh);
}

void ProcThread::run() {
	isRunning = true;
	if (rawLeft == NULL)
		return;

	cv::Mat tmpMatRGBLeft(leftGrabber.getHeight(),
			leftGrabber.getWidth(), CV_8UC3, rgbLeft);
	cv::Mat tmpMatRGBRight(leftGrabber.getHeight(),
			leftGrabber.getWidth(), CV_8UC3, rgbRight);

	QImage imgLeft, imgRight;
	imgLeft = QImage(rgbLeft, leftGrabber.getWidth(),
			leftGrabber.getHeight(), QImage::Format_RGB888);
	imgRight = QImage(rgbRight, leftGrabber.getWidth(),
			leftGrabber.getHeight(), QImage::Format_RGB888);

	QImage imgLeftDisp, imgRightDisp, leftDispRaw;
	imgLeftDisp = QImage(matcher.host_dispColorLeft,
			leftGrabber.getWidth(), leftGrabber.getHeight(),
			QImage::Format_RGB888);
	imgRightDisp = QImage(matcher.host_dispColorRight,
			leftGrabber.getWidth(), leftGrabber.getHeight(),
			QImage::Format_RGB888);
	leftDispRaw = QImage((unsigned char*) matcher.host_dispRawLeft,
			leftGrabber.getWidth(), leftGrabber.getHeight(),
			QImage::Format_ARGB32);

	while (isRunning) {
		if (leftGrabber.grabFrame(rawLeft)
				&& rightGrabber.grabFrame(rawRight)) {
			clock_t startTime = clock();
			cColorConverter::cudaYUY2ToRGB(rawLeft, rgbLeft, imgSize);
			cColorConverter::cudaYUY2ToRGB(rawRight, rgbRight, imgSize);

			cv::remap(tmpMatRGBLeft, tmpMatRGBLeft, undistData.map1x,
					undistData.map1y, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
					cv::Scalar());
			cv::remap(tmpMatRGBRight, tmpMatRGBRight, undistData.map2x,
					undistData.map2y, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
					cv::Scalar());

			if (matcher.processStereo(rgbLeft,rgbRight)) {

				emit finishedFrame(imgLeft.copy(), imgRight.copy(),
						imgLeftDisp.copy(),imgRightDisp.copy(),leftDispRaw.copy());
			}else{
				printf("error\n");
			}

			printf("%f ms\n", (clock()-startTime)/10000.0);
		}
	}
}

void ProcThread::enableAutoExp(bool enable) {
	leftGrabber.setAutoexposureEnabled(enable);
	rightGrabber.setAutoexposureEnabled(enable);
}
void ProcThread::setManualExp(int v) {
	leftGrabber.setExposureTime(v);
	rightGrabber.setExposureTime(v);
}
