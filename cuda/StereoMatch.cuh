#ifndef _CUDA_STEREO_MATCH_H_
#define _CUDA_STEREO_MATCH_H_

#include "cuda.h"
#include "cuda_runtime.h"

class cCudaStereoMatcher {
public:
	cCudaStereoMatcher();
	~cCudaStereoMatcher();

	typedef enum {
		INPUT_GRAYSCALE = 0x01, INPUT_COLOR = 0x02, SUBPIXEL = 0x04, COLOR_MATCH = 0x08
	} tOperationMode;

	bool initSystem(int width, int height, tOperationMode modes);
	void deinitSystem();
	void updateSettings(int kernelSize, int maxDisp, int consistencyTreshold);

	bool processStereo(unsigned char* host_leftImg,
			unsigned char* host_rightImg);

	// Direct access
	unsigned char* host_grayLeft;
	unsigned char* host_grayRight;
	unsigned char* host_colorLeft;
	unsigned char* host_colorRight;
	unsigned char* host_dispColorLeft;
	unsigned char* host_dispColorRight;
	float* host_dispRawLeft;
	float* host_dispRawRight;

private:
	tOperationMode modes;
	bool isInitialized;
	int imgSize, width, height, kernelSize, maxDisp, consistencyTreshold;

	int blockSize, blockCnt;
	bool grayscaleBufferUsed;

	unsigned char* dev_tmpBuff;
	unsigned char* dev_tmpBuff2;
	unsigned char* dev_grayLeft;
	unsigned char* dev_grayRight;
	unsigned char* dev_dispColorLeft;
	unsigned char* dev_dispColorRight;
	float* dev_dispRawLeft;
	float* dev_dispRawRight;
};

#endif
