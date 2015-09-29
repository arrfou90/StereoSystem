#include "cuda.h"
#include "ColorConverter.cuh"
#include "kernels/ColorConverterKernels.cuh"

void cColorConverter::cudaCalcHist_dev(unsigned char* dev_data, unsigned int* dev_hist,
		unsigned int size) {

	int blockSize = 1024;
	int n_blocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

	kernelCalcHist<<<n_blocks, blockSize>>>(dev_data, dev_hist, size);
}
bool cColorConverter::cudaCalcHist(unsigned char* data, unsigned int* hist, unsigned int size) {
	if (size > 1024 * 1024) {
		return false;
	}

	unsigned char* dev_data;
	unsigned int* dev_hist;
	unsigned int histSize = 256;
	unsigned int histBuffSize = histSize * sizeof(int);

	cudaMalloc((void**) &dev_data, size);
	cudaMalloc((void**) &dev_hist, histBuffSize);

	cudaMemset(dev_hist, 0, histBuffSize);
	cudaMemcpy(dev_data, data, size, cudaMemcpyHostToDevice);

	cudaCalcHist_dev(dev_data, dev_hist, size);

	cudaMemcpy(hist, dev_hist, histBuffSize, cudaMemcpyDeviceToHost);
	cudaFree(dev_data);
	cudaFree(dev_hist);

	return true;
}

void cColorConverter::cudaRGBToGray_dev(unsigned char* dev_rgbData, unsigned char* dev_grayData,
		unsigned int imgSize) {
	int blockSize = 1024;
	int nBlocks = imgSize / blockSize + (imgSize % blockSize == 0 ? 0 : 1);

	kernelRGBToGray<<<nBlocks, blockSize>>>(dev_rgbData, dev_grayData, imgSize);
}
bool cColorConverter::cudaRGBToGray(unsigned char* host_rgbData, unsigned char* host_grayData,
		unsigned int imgSize) {
	if (imgSize > 1024 * 1024)
		return false;

	unsigned char* dev_rgbData;
	unsigned char* dev_grayData;

	cudaMalloc((void**) &dev_rgbData, imgSize * 3);
	cudaMalloc((void**) &dev_grayData, imgSize);

	cudaMemcpy(dev_rgbData, host_rgbData, imgSize * 3, cudaMemcpyHostToDevice);

	cudaRGBToGray_dev(dev_rgbData, dev_grayData, imgSize);

	cudaMemcpy(host_grayData, dev_grayData, imgSize, cudaMemcpyDeviceToHost);
	cudaFree(dev_rgbData);
	cudaFree(dev_grayData);

	return true;
}

void cColorConverter::cudaYUY2ToRGB_dev(unsigned char* dev_yuy2Data, unsigned char* dev_rgbData,
		unsigned int imgSize) {

	int blockSize = 1024;
	int nBlocks = (imgSize / 2) / blockSize
			+ ((imgSize / 2) % blockSize == 0 ? 0 : 1);

	kernelYUY2ToRGB<<<nBlocks, blockSize>>>(dev_yuy2Data, dev_rgbData, imgSize);
}
bool cColorConverter::cudaYUY2ToRGB(unsigned char* host_yuy2Data, unsigned char* host_rgbData,
		unsigned int imgSize) {

	unsigned char* dev_rgbData;
	unsigned char* dev_yuy2Data;

	cudaMalloc((void**) &dev_rgbData, imgSize * 3);
	cudaMalloc((void**) &dev_yuy2Data, imgSize * 2);

	cudaMemcpy(dev_yuy2Data, host_yuy2Data, imgSize * 2,
			cudaMemcpyHostToDevice);

	cudaYUY2ToRGB_dev(dev_yuy2Data, dev_rgbData, imgSize);

	cudaMemcpy(host_rgbData, dev_rgbData, imgSize * 3, cudaMemcpyDeviceToHost);

	cudaFree(dev_rgbData);
	cudaFree(dev_yuy2Data);
	return true;
}

void cColorConverter::cudaYUY2ToGray_dev(unsigned char* dev_yuy2Data,
		unsigned char* dev_grayData, unsigned int imgSize) {

	int blockSize = 1024;
	int nBlocks = imgSize / blockSize + (imgSize % blockSize == 0 ? 0 : 1);

	kernelYUY2ToGray<<<nBlocks, blockSize>>>(dev_yuy2Data, dev_grayData,
			imgSize);
}
bool cColorConverter::cudaYUY2ToGray(unsigned char* host_yuy2Data, unsigned char* host_grayData,
		unsigned int imgSize) {
	unsigned char* dev_grayData;
	unsigned char* dev_yuy2Data;

	cudaMalloc((void**) &dev_grayData, imgSize);
	cudaMalloc((void**) &dev_yuy2Data, imgSize * 2);

	cudaMemcpy(dev_yuy2Data, host_yuy2Data, imgSize * 2,
			cudaMemcpyHostToDevice);

	cudaYUY2ToGray_dev(dev_yuy2Data, dev_grayData, imgSize);

	cudaMemcpy(host_grayData, dev_grayData, imgSize, cudaMemcpyDeviceToHost);

	cudaFree(dev_grayData);
	cudaFree(dev_yuy2Data);
	return true;
}
void cColorConverter::cudaGrayToPseudoColor_dev(unsigned char* dev_grayImg,
		unsigned char* dev_pseudoColor, int imgSize, int maxGray, int minH,
		int maxH) {
	int blockSize = 1024;
	int nBlocks = imgSize / blockSize + (imgSize % blockSize == 0 ? 0 : 1);

	kernelGrayToPseudoColor<<<nBlocks, blockSize>>>(dev_grayImg,
			dev_pseudoColor, imgSize, maxGray, minH, maxH);
}
void cColorConverter::cudaGrayToPseudoColor(unsigned char* host_grayImg,
		unsigned char* host_pseudoColor, int imgSize, int maxGray, int minH,
		int maxH) {
	unsigned char* dev_grayImg;
	unsigned char* dev_pseudoColor;

	cudaMalloc((void**) &dev_grayImg, imgSize);
	cudaMalloc((void**) &dev_pseudoColor, imgSize * 3);

	cudaMemcpy(dev_grayImg, host_grayImg, imgSize, cudaMemcpyHostToDevice);

	cudaGrayToPseudoColor_dev(dev_grayImg, dev_pseudoColor, imgSize, maxGray,
			minH, maxH);

	cudaMemcpy(host_pseudoColor, dev_pseudoColor, imgSize * 3,
			cudaMemcpyDeviceToHost);

	cudaFree(dev_grayImg);
	cudaFree(dev_pseudoColor);
}
