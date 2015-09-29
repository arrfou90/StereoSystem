#include "StereoMatch.cuh"
#include "kernels/StereoMatchKernels.cuh"
#include "kernels/ColorConverterKernels.cuh"
#include "errorCheck.cuh"

cCudaStereoMatcher::cCudaStereoMatcher() {
	isInitialized = false;
	grayscaleBufferUsed = false;
	imgSize = 0;
	width = 0;
	height = 0;
	kernelSize = 0;
	maxDisp = 0;
	consistencyTreshold = 0;
	modes = OP_INPUT_GRAYSCALE;

	host_grayLeft = NULL;
	host_grayRight = NULL;
	host_colorLeft = NULL;
	host_colorRight = NULL;
	host_dispColorLeft = NULL;
	host_dispColorRight = NULL;
	host_dispRawLeft = NULL;
	host_dispRawRight = NULL;

	dev_colorLeft = NULL;
	dev_colorRight = NULL;
	dev_grayLeft = NULL;
	dev_grayRight = NULL;
	dev_dispColorLeft = NULL;
	dev_dispColorRight = NULL;
	dev_dispRawLeft = NULL;
	dev_dispRawRight = NULL;
}
cCudaStereoMatcher::~cCudaStereoMatcher() {
	deinitSystem();
}
bool cCudaStereoMatcher::initSystem(int width, int height, int modes) {

	if(isInitialized)
		deinitSystem();

	printf("width: %d, height: %d, mode: %d\n",width,height,modes);

	if((modes & OP_INPUT_COLOR) && (modes & OP_INPUT_GRAYSCALE)){
		printf("Invalid operation mode. \n");
		return false;
	}

	if(width <= 0 || height <= 0 || modes < 1){
		printf("Invalid mode or size. \n");
		return false;
	}

	this->width = width;
	this->height = height;
	this->imgSize = width*height;
	this->modes = modes;

	blockSize = 1024;
	blockCnt = imgSize / blockSize + (imgSize % blockSize == 0 ? 0 : 1);
	printf("BlockCnt: %d, BlockSize: %d\n",blockCnt,blockSize);

	// Farbbild als Eingabe aber kein color match -> Grayscale erzeugen
	if((modes & OP_INPUT_COLOR) && !(modes & OP_COLOR_MATCH)){
		grayscaleBufferUsed = true;
	}
	else{
		grayscaleBufferUsed = false;
	}

	host_grayLeft = new unsigned char[imgSize];
	host_grayRight = new unsigned char[imgSize];
	host_colorLeft = new unsigned char[imgSize * 3];
	host_colorRight = new unsigned char[imgSize * 3];
	host_dispColorLeft = new unsigned char[imgSize * 3];
	host_dispColorRight = new unsigned char[imgSize * 3];
	host_dispRawLeft = new float[imgSize];
	host_dispRawRight = new float[imgSize];

	CudaSafeCall(cudaMalloc((void**)&dev_grayLeft,imgSize));
	CudaSafeCall(cudaMalloc((void**)&dev_grayRight,imgSize));
	CudaSafeCall(cudaMalloc((void**)&dev_colorLeft,imgSize * 3));
	CudaSafeCall(cudaMalloc((void**)&dev_colorRight,imgSize * 3));
	CudaSafeCall(cudaMalloc((void**)&dev_dispColorLeft,imgSize * 3));
	CudaSafeCall(cudaMalloc((void**)&dev_dispColorRight,imgSize * 3));
	CudaSafeCall(cudaMalloc((void**)&dev_dispRawLeft,imgSize * sizeof(float)));
	CudaSafeCall(cudaMalloc((void**)&dev_dispRawRight,imgSize * sizeof(float)));
	isInitialized = true;
	return true;
}
void cCudaStereoMatcher::deinitSystem() {
	if (isInitialized) {
		isInitialized = false;

		delete host_grayLeft;
		delete host_grayRight;
		delete host_colorLeft;
		delete host_colorRight;
		delete host_dispColorLeft;
		delete host_dispColorRight;
		delete host_dispRawLeft;
		delete host_dispRawRight;

		CudaSafeCall(cudaFree((void**) &dev_colorLeft));
		CudaSafeCall(cudaFree((void**) &dev_colorRight));
		CudaSafeCall(cudaFree((void**) &dev_grayLeft));
		CudaSafeCall(cudaFree((void**) &dev_grayRight));
		CudaSafeCall(cudaFree((void**) &dev_dispColorLeft));
		CudaSafeCall(cudaFree((void**) &dev_dispColorRight));
		CudaSafeCall(cudaFree((void**) &dev_dispRawLeft));
		CudaSafeCall(cudaFree((void**) &dev_dispRawRight));
	}
}
void cCudaStereoMatcher::updateSettings(int kernelSize, int maxDisp, int consistencyTreshold) {
	this->kernelSize = kernelSize;
	this->maxDisp = maxDisp;
	this->consistencyTreshold = consistencyTreshold;
}
bool cCudaStereoMatcher::processStereo(unsigned char* host_leftImg, unsigned char* host_rightImg) {

	if(!isInitialized)
		return false;

	bool colorInput = modes & OP_INPUT_COLOR;
	bool colorMatch = modes & OP_COLOR_MATCH;
	bool subpixel = modes & OP_SUBPIXEL;

	CudaSafeCall(cudaMemset(dev_dispColorLeft,0,imgSize*3));
	CudaSafeCall(cudaMemset(dev_dispColorRight,0,imgSize*3));
	CudaSafeCall(cudaMemset(dev_dispRawLeft,0,imgSize*sizeof(float)));
	CudaSafeCall(cudaMemset(dev_dispRawRight,0,imgSize*sizeof(float)));

	if(colorInput){
		CudaSafeCall(cudaMemcpy(dev_colorLeft,host_leftImg,imgSize*3,cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy(dev_colorRight,host_rightImg,imgSize*3,cudaMemcpyHostToDevice));
		if(colorMatch && subpixel){

		}
		else if(colorMatch){
			kernelStereoMatchL2R<<<blockCnt, blockSize>>>(dev_colorLeft,
					dev_colorRight, dev_dispRawLeft, width, height, kernelSize,
					maxDisp, colorMatch, subpixel);
			CudaCheckError();
			kernelStereoMatchR2L<<<blockCnt, blockSize>>>(dev_colorLeft,
					dev_colorRight, dev_dispRawRight, width, height, kernelSize,
					maxDisp, colorMatch, subpixel);
			CudaCheckError();

			if (consistencyTreshold >= 0) {
				kernelLRConsistencyCheck<<<blockCnt, blockSize>>>(
						dev_dispRawLeft, dev_dispRawRight, width, height,
						kernelSize, consistencyTreshold, colorMatch);
				CudaCheckError();
			}
		}
		else if (subpixel){

		}else {
			kernelRGBToGray<<<blockCnt, blockSize>>>(dev_colorLeft,dev_grayLeft, imgSize);
			CudaCheckError();
			kernelRGBToGray<<<blockCnt, blockSize>>>(dev_colorRight,dev_grayRight, imgSize);
			CudaCheckError();

			kernelStereoMatchL2R<<<blockCnt, blockSize>>>(dev_grayLeft,
					dev_grayRight, dev_dispRawLeft, width, height, kernelSize,
					maxDisp, colorMatch, subpixel);
			CudaCheckError();
			kernelStereoMatchR2L<<<blockCnt, blockSize>>>(dev_grayLeft,
					dev_grayRight, dev_dispRawRight, width, height, kernelSize,
					maxDisp, colorMatch, subpixel);
			CudaCheckError();

			if (consistencyTreshold >= 0) {
				kernelLRConsistencyCheck<<<blockCnt, blockSize>>>(
						dev_dispRawLeft, dev_dispRawRight, width, height,
						kernelSize, consistencyTreshold, colorMatch);
				CudaCheckError();
			}
		}
	}
	// Grayscale
	// TODO Testen
	else{

		if (subpixel) {

		} else {
			CudaSafeCall(
					cudaMemcpy(dev_grayLeft, host_leftImg, imgSize,
							cudaMemcpyHostToDevice));
			CudaSafeCall(
					cudaMemcpy(dev_grayRight, host_rightImg, imgSize,
							cudaMemcpyHostToDevice));

			kernelStereoMatchL2R<<<blockCnt, blockSize>>>(dev_grayLeft,
					dev_grayRight, dev_dispRawLeft, width, height, kernelSize,
					maxDisp, colorMatch, subpixel);
			CudaCheckError();
			kernelStereoMatchR2L<<<blockCnt, blockSize>>>(dev_grayLeft,
					dev_grayRight, dev_dispRawRight, width, height, kernelSize,
					maxDisp, colorMatch, subpixel);
			CudaCheckError();

			if (consistencyTreshold >= 0) {
				kernelLRConsistencyCheck<<<blockCnt, blockSize>>>(
						dev_dispRawLeft, dev_dispRawRight, width, height,
						kernelSize, consistencyTreshold, colorMatch);
				CudaCheckError();
			}
		}
	}

	kernelGrayToPseudoColor<<<blockCnt,blockSize>>>(dev_dispRawLeft,dev_dispColorLeft,imgSize,maxDisp,0,120);
	CudaCheckError();
	kernelGrayToPseudoColor<<<blockCnt,blockSize>>>(dev_dispRawRight,dev_dispColorRight,imgSize,maxDisp,0,120);
	CudaCheckError();

	// Receive data
	CudaSafeCall(cudaMemcpy(host_dispRawLeft,dev_dispRawLeft,imgSize * sizeof(float),cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(host_dispRawRight,dev_dispRawRight,imgSize * sizeof(float),cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(host_dispColorLeft,dev_dispColorLeft,imgSize*3,cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(host_dispColorRight,dev_dispColorRight,imgSize*3,cudaMemcpyDeviceToHost));

	return true;
}
