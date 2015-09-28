#ifndef GRABBER_H_
#define GRABBER_H_

#include <string>
#include <vector>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <iostream>

using namespace std;

class cGrabber {
public:
	cGrabber();
	~cGrabber();

	typedef struct CaptureDeviceInfo{
		string devName;
	} CaptureDeviceInfo;
	static vector<CaptureDeviceInfo> enumDevices();

	bool openAndInitDev(string devStr, int reqWidth = -1, int reqHeight = -1);
	void closeDev();

	bool startCapture();
	bool stopCapture();

	int getWidth();
	int getHeight();

	bool grabFrame(unsigned char* buff);

	void printCaps();

	bool setAutoexposureEnabled(bool enabled);
	bool setExposureTime(int time_us);

private:
	struct buffer {
	        void   *start;
	        size_t  length;
	};

	struct buffer* buffers;
	unsigned int   n_buffers;

	string devStr;
	bool isCaptureing;
	bool isInitialized;

	int devFd;

	struct v4l2_pix_format format;


};


#endif
