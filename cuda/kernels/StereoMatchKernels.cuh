
#ifndef _CUDA_STEREO_MATCH_KERNELS_
#define _CUDA_STEREO_MATCH_KERNELS_

__global__ void kernelStereoMatchL2R(unsigned char* dev_leftImg,
		unsigned char* dev_rightImg, float* dev_disparity, int width,
		int height, int kernelSize, int maxDisp, bool colorInput, bool subPixel);
__global__ void kernelStereoMatchR2L(unsigned char* dev_leftImg,
		unsigned char* dev_rightImg, float* dev_disparity, int width,
		int height, int kernelSize, int maxDisp, bool colorInput, bool subPixel);
__global__ void kernelLRConsistencyCheck(float* dev_dispLeft,
		float* dev_dispRight, int width, int height, int kernelSize,
		int consistencyTreshold, bool colorInput);

#endif
