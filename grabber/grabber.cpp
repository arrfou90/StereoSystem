#include "grabber.h"
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sstream>

#define CLEAR(x) memset(&(x), 0, sizeof(x))

static int xioctl(int devFd, int request, void *arg) {
	int r;
	do
		r = ioctl(devFd, request, arg);
	while (-1 == r && EINTR == errno);
	return r;
}

cGrabber::cGrabber() {
	devStr = "";
	isCaptureing = false;
	isInitialized = false;
	devFd = -1;
	buffers = NULL;
	n_buffers = 0;
}
cGrabber::~cGrabber() {

}
vector<cGrabber::CaptureDeviceInfo> cGrabber::enumDevices() {
	vector<cGrabber::CaptureDeviceInfo> devInfo;
	int Nmax = 64;
	int Ncurr = 0;
	int fd;
	string devNameTmp = "/dev/video";
	stringstream devname;
	devname << devNameTmp << Ncurr;

	struct v4l2_capability caps = { };

	while(Ncurr <= Nmax && (fd = open(devname.str().c_str(), O_RDWR)) != -1){


		if (xioctl(fd, VIDIOC_QUERYCAP, &caps) != -1) {
			CaptureDeviceInfo info;
			info.devName = devname.str();

			devInfo.push_back(info);
		}

		Ncurr++;
		devname.str(string());
		devname << devNameTmp << Ncurr;
		close(fd);
	}
	cout << "Can't open dev: " << devname.str() << endl;


	return devInfo;
}

bool cGrabber::openAndInitDev(string devStr, int reqWidth, int reqHeight) {
	if (devFd != -1)
		closeDev();
	cout << "Capture from " << devStr << endl;

	devFd = open(devStr.c_str(), O_RDWR);
	if (devFd == -1) {
		// couldn't find capture device
		perror("Opening Video device");
		return false;
	}
	this->devStr = devStr;

	struct v4l2_capability cap;
	struct v4l2_cropcap cropcap;
	struct v4l2_crop crop;
	struct v4l2_format fmt;
	unsigned int min;

	if (-1 == xioctl(devFd, VIDIOC_QUERYCAP, &cap)) {
		if (EINVAL == errno) {
			fprintf(stderr, "%s is no V4L2 device\n", devStr.c_str());
			return false;
		} else {
			return false;
		}
	}

	if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
		fprintf(stderr, "%s is no video capture device\n", devStr.c_str());
		return false;
	}

	if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
		fprintf(stderr, "%s does not support streaming i/o\n", devStr.c_str());
		return false;
	}

	/* Select video input, video standard and tune here. */
	CLEAR(cropcap);

	cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if (0 == xioctl(devFd, VIDIOC_CROPCAP, &cropcap)) {
		crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		crop.c = cropcap.defrect; /* reset to default */

		if (-1 == xioctl(devFd, VIDIOC_S_CROP, &crop)) {
			switch (errno) {
			case EINVAL:
				/* Cropping not supported. */
				break;
			default:
				/* Errors ignored. */
				break;
			}
		}
	} else {
		/* Errors ignored. */
	}

	CLEAR(fmt);
	bool force_format = reqHeight > 0 && reqWidth > 0;
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (force_format) {
		fmt.fmt.pix.width = reqWidth;
		fmt.fmt.pix.height = reqHeight;
		fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
		fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

		if (-1 == xioctl(devFd, VIDIOC_S_FMT, &fmt))
			return false;

		/* Note VIDIOC_S_FMT may change width and height. */
	}


	/* Preserve original settings as set by v4l2-ctl for example */
	if (-1 == xioctl(devFd, VIDIOC_G_FMT, &fmt))
		return false;

	format = fmt.fmt.pix;

	struct v4l2_requestbuffers req = { 0 };
	req.count = 2;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;

	if (-1 == xioctl(devFd, VIDIOC_REQBUFS, &req)) {
		perror("Requesting Buffer");
		return false;
	}

	if (req.count < 2) {
		fprintf(stderr, "Insufficient buffer memory on %s\n", devStr.c_str());
		return false;
	}

	buffers = (struct buffer*)calloc(req.count, sizeof(*buffers));

	if (!buffers) {
		fprintf(stderr, "Out of memory\n");
		return false;
	}

    for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
		struct v4l2_buffer buf = { 0 };

        CLEAR(buf);

        buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory      = V4L2_MEMORY_MMAP;
        buf.index       = n_buffers;

		if (-1 == xioctl(devFd, VIDIOC_QUERYBUF, &buf)) {
			perror("Querying Buffer");
			return false;
		}
		buffers[n_buffers].length = buf.length;
		buffers[n_buffers].start = (unsigned char*) mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
				MAP_SHARED, devFd, buf.m.offset);

        if (MAP_FAILED == buffers[n_buffers].start){
			perror("mmap");
			return false;
        }
	}
    isInitialized = true;
	return true;
}
void cGrabber::closeDev() {
    for (int i = 0; i < n_buffers; ++i)
    	munmap(buffers[i].start, buffers[i].length);

	close(devFd);
	devFd = -1;
	isInitialized = false;
}

bool cGrabber::startCapture() {
	isCaptureing = false;
	for (int i = 0; i < n_buffers; ++i) {
		struct v4l2_buffer buf;

		CLEAR(buf);
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;

		if (-1 == xioctl(devFd, VIDIOC_QBUF, &buf)){
			perror("VIDIOC_QBUF");
			return false;
		}
	}

    enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == xioctl(devFd, VIDIOC_STREAMON, &type)) {
		perror("Start Capture");
		return false;
	}

	isCaptureing = true;
	return true;
}

bool cGrabber::stopCapture() {

	enum v4l2_buf_type type;
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == xioctl(devFd, VIDIOC_STREAMOFF, &type)){
		perror("VIDIOC_STREAMOFF");
		return false;
	}
	isCaptureing = false;
	return true;

}

int cGrabber::getWidth() {
	return format.width;
}
int cGrabber::getHeight() {
	return format.height;
}

bool cGrabber::grabFrame(unsigned char* buff) {
	if(!isInitialized && ! isCaptureing)
		return false;

	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(devFd, &fds);
	struct timeval tv = { 0 };
	tv.tv_sec = 0;
	tv.tv_usec = 10000;
	int r = select(devFd + 1, &fds, NULL, NULL, &tv);
	if (-1 == r) {
		perror("Waiting for Frame");
		return false;
	}
	if(0 == r){
		return false;
	}
    struct v4l2_buffer buf = {0};
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;

	if (-1 == xioctl(devFd, VIDIOC_DQBUF, &buf)) {
		perror("Retrieving Frame");
		return false;
	}

	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	if (-1 == xioctl(devFd, VIDIOC_QBUF, &buf)) {
		perror("Query Buffer");
		return false;
	}

	memcpy(buff,buffers[buf.index].start,buffers[buf.index].length);
	return true;
}
void cGrabber::printCaps()
{
	struct v4l2_capability caps = { };
	if (-1 == xioctl(devFd, VIDIOC_QUERYCAP, &caps)) {
		perror("Querying Capabilities");
		return;
	}

	printf("Driver Caps:\n"
			"  Driver: \"%s\"\n"
			"  Card: \"%s\"\n"
			"  Bus: \"%s\"\n"
			"  Version: %d.%d\n"
			"  Capabilities: %08x\n", caps.driver, caps.card, caps.bus_info,
			(caps.version >> 16) && 0xff, (caps.version >> 24) && 0xff,
			caps.capabilities);

	struct v4l2_cropcap cropcap = { 0 };
	cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == xioctl(devFd, VIDIOC_CROPCAP, &cropcap)) {
		perror("Querying Cropping Capabilities");
		return;
	}

	printf("Camera Cropping:\n"
			"  Bounds: %dx%d+%d+%d\n"
			"  Default: %dx%d+%d+%d\n"
			"  Aspect: %d/%d\n", cropcap.bounds.width, cropcap.bounds.height,
			cropcap.bounds.left, cropcap.bounds.top, cropcap.defrect.width,
			cropcap.defrect.height, cropcap.defrect.left, cropcap.defrect.top,
			cropcap.pixelaspect.numerator, cropcap.pixelaspect.denominator);

	int support_grbg10 = 0;

	struct v4l2_fmtdesc fmtdesc = { 0 };
	fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	char fourcc[5] = { 0 };
	char c, e;
	printf("  FMT : CE Desc\n--------------------\n");
	while (0 == xioctl(devFd, VIDIOC_ENUM_FMT, &fmtdesc)) {
		strncpy(fourcc, (char *) &fmtdesc.pixelformat, 4);
		if (fmtdesc.pixelformat == V4L2_PIX_FMT_SGRBG10)
			support_grbg10 = 1;
		c = fmtdesc.flags & 1 ? 'C' : ' ';
		e = fmtdesc.flags & 2 ? 'E' : ' ';
		printf("  %s: %c%c %s\n", fourcc, c, e, fmtdesc.description);
		fmtdesc.index++;
	}
	/*
	 if (!support_grbg10)
	 {
	 printf("Doesn't support GRBG10.\n");
	 return 1;
	 }*/

	struct v4l2_format fmt = { 0 };
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == xioctl(devFd, VIDIOC_G_FMT, &fmt)) {
		perror("Getting Pixel Format");
		return;
	}

	strncpy(fourcc, (char *) &fmt.fmt.pix.pixelformat, 4);
	printf("Selected Camera Mode:\n"
			"  Width: %d\n"
			"  Height: %d\n"
			"  PixFmt: %s\n"
			"  Field: %d\n", fmt.fmt.pix.width, fmt.fmt.pix.height, fourcc,
			fmt.fmt.pix.field);
	return;
}

bool cGrabber::setAutoexposureEnabled(bool enabled)
{
	v4l2_control ctrl;
	ctrl.id = V4L2_CID_EXPOSURE_AUTO;
	if(enabled)
		ctrl.value = V4L2_EXPOSURE_APERTURE_PRIORITY;
	else
		ctrl.value = V4L2_EXPOSURE_MANUAL;

	return xioctl(devFd,VIDIOC_S_CTRL, &ctrl) == 0;

}
bool cGrabber::setExposureTime(int time_100us)
{
	v4l2_control ctrl;
	ctrl.id = V4L2_CID_EXPOSURE_ABSOLUTE;
	ctrl.value = time_100us;

	return xioctl(devFd,VIDIOC_S_CTRL, &ctrl) == 0;
}
