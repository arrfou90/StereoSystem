#include <cuda.h>
#include <cuda_runtime.h>

#include "ColorConverterKernels.cuh"
#include "../errorCheck.cuh"

__global__ void kernelCalcHist(unsigned char* data, unsigned int* hist,
		unsigned int size) {
	// Shared memory für lokales Histogramm im Aktuellen Block
	__shared__ unsigned int temp[256];

	// Thread i im Block setzt tmp[i] auf 0
	if (threadIdx.x < 256)
		temp[threadIdx.x] = 0;

	__syncthreads();	// Hier ist dann alles auf 0 gesetzt

	// Get index
	//  i =    x        + y * width;
	// Der Pixel, der durch den Thread verarbeitet wird.
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// Im Shared memory histogram berechnen
	if (i < size)
		atomicAdd(&temp[data[i]], 1);

	// Warten bis alle Threads ihr Lokales histogramm berechnet haben.
	__syncthreads();

	// die ersten 256 Threads jedes Blocks addieren die lokalen Ergebnisse
	if (threadIdx.x < 256)
		atomicAdd(&(hist[threadIdx.x]), temp[threadIdx.x]);
}
void cudaCalcHist_dev(unsigned char* dev_data, unsigned int* dev_hist,
		unsigned int size) {

	int blockSize = 1024;
	int n_blocks = size / blockSize + (size % blockSize == 0 ? 0 : 1);

	kernelCalcHist<<<n_blocks, blockSize>>>(dev_data, dev_hist, size);
	CudaCheckError();
}
bool cudaCalcHist(unsigned char* data, unsigned int* hist, unsigned int size) {
	if (size > 1024 * 1024) {
		return false;
	}

	unsigned char* dev_data;
	unsigned int* dev_hist;
	unsigned int histSize = 256;
	unsigned int histBuffSize = histSize * sizeof(int);

	CudaSafeCall(cudaMalloc((void**) &dev_data, size));
	CudaSafeCall(cudaMalloc((void**) &dev_hist, histBuffSize));

	CudaSafeCall(cudaMemset(dev_hist, 0, histBuffSize));
	CudaSafeCall(cudaMemcpy(dev_data, data, size, cudaMemcpyHostToDevice));

	cudaCalcHist_dev(dev_data, dev_hist, size);

	CudaSafeCall(cudaMemcpy(hist, dev_hist, histBuffSize, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaFree(dev_data));
	CudaSafeCall(cudaFree(dev_hist));

	return true;
}

__global__ void kernelRGBToGray(unsigned char* dev_rgbData,
		unsigned char* dev_grayData, unsigned int imgSize) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < imgSize)
		dev_grayData[i] = dev_rgbData[3 * i] * 0.144
				+ dev_rgbData[3 * i + 1] * 0.587
				+ dev_rgbData[3 * i + 2] * 0.299;
}
void cudaRGBToGray_dev(unsigned char* dev_rgbData, unsigned char* dev_grayData,
		unsigned int imgSize) {
	int blockSize = 1024;
	int nBlocks = imgSize / blockSize + (imgSize % blockSize == 0 ? 0 : 1);

	kernelRGBToGray<<<nBlocks, blockSize>>>(dev_rgbData, dev_grayData, imgSize);
	CudaCheckError();
}
bool cudaRGBToGray(unsigned char* host_rgbData, unsigned char* host_grayData,
		unsigned int imgSize) {
	if (imgSize > 1024 * 1024)
		return false;

	unsigned char* dev_rgbData;
	unsigned char* dev_grayData;

	CudaSafeCall(cudaMalloc((void**) &dev_rgbData, imgSize * 3));
	CudaSafeCall(cudaMalloc((void**) &dev_grayData, imgSize));

	CudaSafeCall(cudaMemcpy(dev_rgbData, host_rgbData, imgSize * 3, cudaMemcpyHostToDevice));

	cudaRGBToGray_dev(dev_rgbData, dev_grayData, imgSize);

	CudaSafeCall(cudaMemcpy(host_grayData, dev_grayData, imgSize, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaFree(dev_rgbData));
	CudaSafeCall(cudaFree(dev_grayData));

	return true;
}

__global__ void kernelYUY2ToRGB(unsigned char* dev_yuy2Data,
		unsigned char* dev_rgbData, unsigned int imgSize) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	i *= 2;

	if (i < imgSize - 1) {
		unsigned char* yuy2Buff = dev_yuy2Data + i * 2;
		unsigned char* rgbBuff = dev_rgbData + i * 3;
		int u, v, y1, y2;

		y1 = *yuy2Buff++;
		u = *yuy2Buff++;
		y2 = *yuy2Buff++;
		v = *yuy2Buff++;

		// Integer operation of ITU-R standard for YCbCr is (from Wikipedia)
		// https://en.wikipedia.org/wiki/YUV#Y.27UV422_to_RGB888_conversion
		u = u - 128;
		v = v - 128;

		*rgbBuff++ = CLIP(y1 + 45 * v / 32);
		*rgbBuff++ = CLIP(y1 - (11 * u + 23 * v) / 32);
		*rgbBuff++ = CLIP(y1 + 113 * u / 64);

		*rgbBuff++ = CLIP(y2 + 45 * v / 32);
		*rgbBuff++ = CLIP(y2 - (11 * u + 23 * v) / 32);
		*rgbBuff++ = CLIP(y2 + 113 * u / 64);

//		v = v - 128;
//		u = u - 128;
//
//		*rgbBuff++ = y1 + u + (u >> 1) + (u >> 2) + (u >> 6);
//		*rgbBuff++ = y1 -((u >> 2) + (u >> 4) + (u >> 5)) - ((v >> 1) + (v >> 3) + (v >> 4) + (v >> 5));
//		*rgbBuff++ = y1 + v + (v >> 2) + (v >> 3) + (v >> 5);
//
//		*rgbBuff++ = y2 + u + (u >> 1) + (u >> 2) + (u >> 6);
//		*rgbBuff++ = y2 -((u >> 2) + (u >> 4) + (u >> 5)) - ((v >> 1) + (v >> 3) + (v >> 4) + (v >> 5));
//		*rgbBuff++ = y2 + v + (v >> 2) + (v >> 3) + (v >> 5);
	}
}

void cudaYUY2ToRGB_dev(unsigned char* dev_yuy2Data, unsigned char* dev_rgbData,
		unsigned int imgSize) {

	int blockSize = 1024;
	int nBlocks = (imgSize / 2) / blockSize
			+ ((imgSize / 2) % blockSize == 0 ? 0 : 1);

	kernelYUY2ToRGB<<<nBlocks, blockSize>>>(dev_yuy2Data, dev_rgbData, imgSize);
}
bool cudaYUY2ToRGB(unsigned char* host_yuy2Data, unsigned char* host_rgbData,
		unsigned int imgSize) {

	unsigned char* dev_rgbData;
	unsigned char* dev_yuy2Data;

	CudaSafeCall(cudaMalloc((void**) &dev_rgbData, imgSize * 3));
	CudaSafeCall(cudaMalloc((void**) &dev_yuy2Data, imgSize * 2));

	CudaSafeCall(cudaMemcpy(dev_yuy2Data, host_yuy2Data, imgSize * 2,
			cudaMemcpyHostToDevice));

	cudaYUY2ToRGB_dev(dev_yuy2Data, dev_rgbData, imgSize);

	CudaSafeCall(cudaMemcpy(host_rgbData, dev_rgbData, imgSize * 3, cudaMemcpyDeviceToHost));

	CudaSafeCall(cudaFree(dev_rgbData));
	CudaSafeCall(cudaFree(dev_yuy2Data));
	return true;
}

__global__ void kernelYUY2ToGray(unsigned char* dev_yuy2Data,
		unsigned char* dev_grayData, unsigned int imgSize) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	dev_grayData[i] = dev_yuy2Data[i * 2];
}
void cudaYUY2ToGray_dev(unsigned char* dev_yuy2Data,
		unsigned char* dev_grayData, unsigned int imgSize) {

	int blockSize = 1024;
	int nBlocks = imgSize / blockSize + (imgSize % blockSize == 0 ? 0 : 1);

	kernelYUY2ToGray<<<nBlocks, blockSize>>>(dev_yuy2Data, dev_grayData,
			imgSize);
	CudaCheckError();
}
bool cudaYUY2ToGray(unsigned char* host_yuy2Data, unsigned char* host_grayData,
		unsigned int imgSize) {
	unsigned char* dev_grayData;
	unsigned char* dev_yuy2Data;

	CudaSafeCall(cudaMalloc((void**) &dev_grayData, imgSize));
	CudaSafeCall(cudaMalloc((void**) &dev_yuy2Data, imgSize * 2));

	CudaSafeCall(cudaMemcpy(dev_yuy2Data, host_yuy2Data, imgSize * 2,
			cudaMemcpyHostToDevice));

	cudaYUY2ToGray_dev(dev_yuy2Data, dev_grayData, imgSize);

	CudaSafeCall(cudaMemcpy(host_grayData, dev_grayData, imgSize, cudaMemcpyDeviceToHost));

	CudaSafeCall(cudaFree(dev_grayData));
	CudaSafeCall(cudaFree(dev_yuy2Data));
	return true;
}
__global__ void kernelGrayToPseudoColor(unsigned char* dev_inputImg,
		unsigned char* dev_pseudoColor, int imgSize, int maxGray, int minH,
		int maxH) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < imgSize) {
		int h, s, v, r, b, g;
		// h: 0 - 359 (0 = rot, 120 = grün)
		// s: 0 - 100
		// v: 0 - 100
		s = 100;
		v = 100;
		int grayVal = dev_inputImg[idx];
		if (grayVal > maxGray)
			grayVal = maxGray;

		if (grayVal > 0) {

			h = (100 * grayVal / maxGray) * (maxH - minH) / 100 + minH;

			// HSV to RGB
			//Winkel im Farbkeis 0 - 360 in 1 Grad Schritten
			//h = (englisch hue) Farbwert
			//1 Grad Schrittweite, 4.25 Steigung pro Schritt bei 60 Grad
			if (h < 61) {
				r = 255;
				b = 0;
				g = 4.25 * h;
			} else if (h < 121) {
				g = 255;
				b = 0;
				r = 255 - (4.25 * (h - 60));
			} else if (h < 181) {
				r = 0;
				g = 255;
				b = 4.25 * (h - 120);
			} else if (h < 241) {
				r = 0;
				b = 255;
				g = 255 - (4.25 * (h - 180));
			} else if (h < 301) {
				g = 0;
				b = 255;
				r = 4.25 * (h - 240);
			} else if (h < 360) {
				r = 255;
				g = 0;
				b = 255 - (4.25 * (h - 300));
			}
			//Berechnung der Farbsättigung
			//s = (englisch saturation) Farbsättigung
			int diff;
			s = 100 - s; //Kehrwert berechnen
			diff = ((255 - r) * s) / 100;
			r = r + diff;
			diff = ((255 - g) * s) / 100;
			g = g + diff;
			diff = ((255 - b) * s) / 100;
			b = b + diff;

			//Berechnung der Dunkelstufe
			//v = (englisch value) Wert Dunkelstufe einfacher Dreisatz 0..100%
			r = (r * v) / 100;
			g = (g * v) / 100;
			b = (b * v) / 100;

			dev_pseudoColor[idx * 3] = r;
			dev_pseudoColor[idx * 3 + 1] = g;
			dev_pseudoColor[idx * 3 + 2] = b;
		} else {
			dev_pseudoColor[idx * 3] = 0;
			dev_pseudoColor[idx * 3 + 1] = 0;
			dev_pseudoColor[idx * 3 + 2] = 0;
		}
	}
}
__global__ void kernelGrayToPseudoColor(float* dev_inputImg,
		unsigned char* dev_pseudoColor, int imgSize, int maxGray, int minH,
		int maxH, bool scaledFloat) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < imgSize) {
		int h, s, v, r, b, g;
		// h: 0 - 359 (0 = rot, 120 = grün)
		// s: 0 - 100
		// v: 0 - 100
		s = 100;
		v = 100;
		int grayVal = dev_inputImg[idx];

		if (scaledFloat)
			grayVal *= 255;

		if (grayVal > maxGray)
			grayVal = maxGray;

		if (grayVal > 0) {

			h = (100 * grayVal / maxGray) * (maxH - minH) / 100 + minH;

			// HSV to RGB
			//Winkel im Farbkeis 0 - 360 in 1 Grad Schritten
			//h = (englisch hue) Farbwert
			//1 Grad Schrittweite, 4.25 Steigung pro Schritt bei 60 Grad
			if (h < 61) {
				r = 255;
				b = 0;
				g = 4.25 * h;
			} else if (h < 121) {
				g = 255;
				b = 0;
				r = 255 - (4.25 * (h - 60));
			} else if (h < 181) {
				r = 0;
				g = 255;
				b = 4.25 * (h - 120);
			} else if (h < 241) {
				r = 0;
				b = 255;
				g = 255 - (4.25 * (h - 180));
			} else if (h < 301) {
				g = 0;
				b = 255;
				r = 4.25 * (h - 240);
			} else if (h < 360) {
				r = 255;
				g = 0;
				b = 255 - (4.25 * (h - 300));
			}
			//Berechnung der Farbsättigung
			//s = (englisch saturation) Farbsättigung
			int diff;
			s = 100 - s; //Kehrwert berechnen
			diff = ((255 - r) * s) / 100;
			r = r + diff;
			diff = ((255 - g) * s) / 100;
			g = g + diff;
			diff = ((255 - b) * s) / 100;
			b = b + diff;

			//Berechnung der Dunkelstufe
			//v = (englisch value) Wert Dunkelstufe einfacher Dreisatz 0..100%
			r = (r * v) / 100;
			g = (g * v) / 100;
			b = (b * v) / 100;

			dev_pseudoColor[idx * 3] = r;
			dev_pseudoColor[idx * 3 + 1] = g;
			dev_pseudoColor[idx * 3 + 2] = b;
		} else {
			dev_pseudoColor[idx * 3] = 0;
			dev_pseudoColor[idx * 3 + 1] = 0;
			dev_pseudoColor[idx * 3 + 2] = 0;
		}
	}
}
void cudaGrayToPseudoColor_dev(unsigned char* dev_grayImg,
		unsigned char* dev_pseudoColor, int imgSize, int maxGray, int minH,
		int maxH) {
	int blockSize = 1024;
	int nBlocks = imgSize / blockSize + (imgSize % blockSize == 0 ? 0 : 1);

	kernelGrayToPseudoColor<<<nBlocks, blockSize>>>(dev_grayImg,
			dev_pseudoColor, imgSize, maxGray, minH, maxH);
	CudaCheckError();
}
void cudaGrayToPseudoColor(unsigned char* host_grayImg,
		unsigned char* host_pseudoColor, int imgSize, int maxGray, int minH,
		int maxH) {
	unsigned char* dev_grayImg;
	unsigned char* dev_pseudoColor;

	CudaSafeCall(cudaMalloc((void**) &dev_grayImg, imgSize));
	CudaSafeCall(cudaMalloc((void**) &dev_pseudoColor, imgSize * 3));

	CudaSafeCall(cudaMemcpy(dev_grayImg, host_grayImg, imgSize, cudaMemcpyHostToDevice));

	cudaGrayToPseudoColor_dev(dev_grayImg, dev_pseudoColor, imgSize, maxGray,
			minH, maxH);

	CudaSafeCall(cudaMemcpy(host_pseudoColor, dev_pseudoColor, imgSize * 3,
			cudaMemcpyDeviceToHost));

	CudaSafeCall(cudaFree(dev_grayImg));
	CudaSafeCall(cudaFree(dev_pseudoColor));
}
