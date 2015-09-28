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
	modes = INPUT_GRAYSCALE;

	host_grayLeft = NULL;
	host_grayRight = NULL;
	host_colorLeft = NULL;
	host_colorRight = NULL;
	host_dispColorLeft = NULL;
	host_dispColorRight = NULL;
	host_dispRawLeft = NULL;
	host_dispRawRight = NULL;

	dev_tmpBuff = NULL;
	dev_tmpBuff2 = NULL;
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
bool cCudaStereoMatcher::initSystem(int width, int height, tOperationMode modes) {

	if(isInitialized)
		deinitSystem();

	printf("width: %d, height: %d, mode: %d\n",width,height,modes);

	if((modes & INPUT_COLOR) && (modes & INPUT_GRAYSCALE)){
		printf("Invalid operation mode. \n");
		return false;
	}

	if(width <= 0 || height <= 0 || modes == 0){
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
	if((modes & INPUT_COLOR) && !(modes & COLOR_MATCH)){
		grayscaleBufferUsed = true;
		host_grayLeft = new unsigned char[imgSize];
		host_grayRight = new unsigned char[imgSize];
		cudaMalloc((void**)&dev_grayLeft,imgSize);
		cudaMalloc((void**)&dev_grayRight,imgSize);
	}
	else{
		grayscaleBufferUsed = false;
	}

	host_dispColorLeft = new unsigned char[imgSize * 3];
	host_dispColorRight = new unsigned char[imgSize * 3];
	host_dispRawLeft = new float[imgSize];
	host_dispRawRight = new float[imgSize];

	CudaSafeCall(cudaMalloc((void**)&dev_tmpBuff,imgSize * 3));
	CudaSafeCall(cudaMalloc((void**)&dev_tmpBuff2,imgSize * 3));
	CudaSafeCall(cudaMalloc((void**)&dev_dispColorLeft,imgSize * 3));
	CudaSafeCall(cudaMalloc((void**)&dev_dispColorRight,imgSize * 3));
	CudaSafeCall(cudaMalloc((void**)&dev_dispRawLeft,imgSize));
	CudaSafeCall(cudaMalloc((void**)&dev_dispRawRight,imgSize));
	isInitialized = true;
	return true;
}
void cCudaStereoMatcher::deinitSystem() {
	if (isInitialized) {
		isInitialized = false;
		if (grayscaleBufferUsed) {
			delete host_grayLeft;
			delete host_grayRight;
		}
		delete host_colorLeft;
		delete host_colorRight;
		delete host_dispColorLeft;
		delete host_dispColorRight;
		delete host_dispRawLeft;
		delete host_dispRawRight;

		CudaSafeCall(cudaFree((void**) &dev_tmpBuff));
		CudaSafeCall(cudaFree((void**) &dev_tmpBuff2));
		if(grayscaleBufferUsed){
			CudaSafeCall(cudaFree((void**) &dev_grayLeft));
			CudaSafeCall(cudaFree((void**) &dev_grayRight));
		}
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

	bool colorInput = modes & INPUT_COLOR;
	bool colorMatch = modes & COLOR_MATCH;
	bool subpixel = modes & SUBPIXEL;

	if(colorInput){
		if(colorMatch && subpixel){

		}
		else if(colorMatch){

		}
		else if (subpixel){

		}else {
			CudaSafeCall(cudaMemcpy(dev_tmpBuff,host_leftImg,imgSize*3,cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpy(dev_tmpBuff2,host_rightImg,imgSize*3,cudaMemcpyHostToDevice));
			kernelRGBToGray<<<blockCnt,blockSize>>>(dev_tmpBuff,dev_grayLeft,imgSize);
			CudaCheckError();
			kernelRGBToGray<<<blockCnt, blockSize>>>(dev_tmpBuff2,
					dev_grayRight, imgSize);
			CudaCheckError();

			kernelStereoMatchL2R<<<blockCnt, blockSize>>>(dev_grayLeft,
					dev_grayRight, dev_dispRawLeft, width, height, kernelSize,
					maxDisp);
			CudaCheckError();
			kernelStereoMatchR2L<<<blockCnt, blockSize>>>(dev_grayLeft,
					dev_grayRight, dev_dispRawRight, width, height, kernelSize,
					maxDisp);
			CudaCheckError();

			if (consistencyTreshold >= 0) {
				kernelLRConsistencyCheck<<<blockCnt, blockSize>>>(
						dev_dispRawLeft, dev_dispRawRight, width, height,
						kernelSize, consistencyTreshold);
				CudaCheckError();
			}
		}
	}
	// Grayscale
	else{

		if (subpixel){

		} else {
			CudaSafeCall(cudaMemcpy(dev_grayLeft, host_leftImg, imgSize, cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpy(dev_grayRight, host_rightImg, imgSize, cudaMemcpyHostToDevice));

			kernelStereoMatchL2R<<<blockCnt,blockSize>>>(dev_grayLeft,dev_grayRight,dev_dispRawLeft,width,height,kernelSize,maxDisp);
			CudaCheckError();
			kernelStereoMatchR2L<<<blockCnt,blockSize>>>(dev_grayLeft,dev_grayRight,dev_dispRawRight,width,height,kernelSize,maxDisp);
			CudaCheckError();

			if (consistencyTreshold >= 0) {
				kernelLRConsistencyCheck<<<blockCnt,blockSize>>>(dev_dispRawLeft, dev_dispRawRight,
						width, height, kernelSize, consistencyTreshold);
				CudaCheckError();
			}
		}
	}

	kernelGrayToPseudoColor<<<blockCnt,blockSize>>>(dev_dispRawLeft,dev_dispColorLeft,imgSize,maxDisp,0,120);
	CudaCheckError();
	kernelGrayToPseudoColor<<<blockCnt,blockSize>>>(dev_dispRawRight,dev_dispColorRight,imgSize,maxDisp,0,120);
	CudaCheckError();

	// Receive data
	CudaSafeCall(cudaMemcpy(host_dispRawLeft,dev_dispRawLeft,imgSize,cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(host_dispRawRight,dev_dispRawRight,imgSize,cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(host_dispColorLeft,dev_dispColorLeft,imgSize*3,cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(host_dispColorRight,dev_dispColorRight,imgSize*3,cudaMemcpyDeviceToHost));

	return true;
}
