#ifndef PROCTHREAD_H_
#define PROCTHREAD_H_
#include "QtCore/qthread.h"
#include "QtCore/qmutex.h"

#include "opencv2/opencv.hpp"

#include "../cuda/ColorConverter.cuh"
#include "../cuda/StereoMatch.cuh"
#include "../grabber/grabber.h"

class ProcThread: public QThread {
  Q_OBJECT

public:
	ProcThread(QObject *parent = 0);
	~ProcThread();
	bool init(string calibFile, string leftDev, string rightDev, int reqWidth = -1, int reqHeight = -1);

	void stop() {
		isRunning = false;
	}
	void uninit();

	void updateSettings(int maxDisp,int consistTresh, int kernelSize);
	void enableAutoExp(bool enable);
	void setManualExp(int v);

	typedef struct {
		// Calibration Data
		cv::Mat CM1, CM2;
		cv::Mat D1, D2;
		cv::Mat R, T, E, F;
		cv::Mat R1, R2, P1, P2, Q;
	} calibrationData;

	typedef struct {
		cv::Mat map1x, map1y, map2x, map2y;
	} undistortData;

signals:
	void finishedFrame(QImage leftRGB, QImage rightRGB, QImage leftDispColor, QImage rightDispColor, QImage leftDispRaw);

protected:
	void run();

	bool isRunning, isInitialized;
	cGrabber leftGrabber, rightGrabber;
	int imgSize;

	cv::FileStorage calibrationSettings;
	calibrationData calibData;

	undistortData undistData;

	cCudaStereoMatcher matcher;
	unsigned char* rawLeft;
	unsigned char* rawRight;
	unsigned char* rgbLeft;
	unsigned char* rgbRight;

};
#endif
