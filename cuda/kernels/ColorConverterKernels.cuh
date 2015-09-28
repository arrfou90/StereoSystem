#ifndef _COLOR_CONVERTER_KERNELS_H_
#define _COLOR_CONVERTER_KERNELS_H_

#define CLIP(X)	X > 255? 255: X<0? 0 : (int) X

__global__ void kernelCalcHist(unsigned char* data, unsigned int* hist,
		unsigned int size);
__global__ void kernelRGBToGray(unsigned char* dev_rgbData,
		unsigned char* dev_grayData, unsigned int imgSize);
__global__ void kernelYUY2ToRGB(unsigned char* dev_yuy2Data,
		unsigned char* dev_brgData, unsigned int imgSize);
__global__ void kernelYUY2ToGray(unsigned char* dev_yuy2Data,
		unsigned char* dev_grayData, unsigned int imgSize);

__global__ void kernelGrayToPseudoColor(unsigned char* dev_inputImg,
		unsigned char* dev_pseudoColor, int imgSize, int maxGray, int minH,
		int maxH);
__global__ void kernelGrayToPseudoColor(float* dev_inputImg,
		unsigned char* dev_pseudoColor, int imgSize, int maxGray, int minH,
		int maxH, bool scaledFloat = false);

#endif
