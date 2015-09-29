#ifndef _COLOR_CONVERTER_H_
#define _COLOR_CONVERTER_H_

class cColorConverter {
public:

	static bool cudaCalcHist(unsigned char* data, unsigned int* hist,
			unsigned int size);

	static bool cudaRGBToGray(unsigned char* host_rgbData,
			unsigned char* host_grayData, unsigned int imgSize);
	static bool cudaYUY2ToRGB(unsigned char* host_yuy2Data,
			unsigned char* host_grayData, unsigned int imgSize);
	static bool cudaYUY2ToGray(unsigned char* host_yuy2Data,
			unsigned char* host_grayData, unsigned int imgSize);
	static void cudaGrayToPseudoColor(unsigned char* host_grayImg,
			unsigned char* host_pseudoColor, int imgSize, int maxGray, int minH,
			int maxH);

private:
	static void cudaCalcHist_dev(unsigned char* dev_data,
			unsigned int* dev_hist, unsigned int size);
	static void cudaRGBToGray_dev(unsigned char* dev_rgbData,
			unsigned char* dev_grayData, unsigned int imgSize);
	static void cudaYUY2ToRGB_dev(unsigned char* dev_yuy2Data,
			unsigned char* dev_rgbData, unsigned int imgSize);
	static void cudaYUY2ToGray_dev(unsigned char* dev_yuy2Data,
			unsigned char* dev_grayData, unsigned int imgSize);

	static void cudaGrayToPseudoColor_dev(unsigned char* dev_grayImg,
			unsigned char* dev_pseudoColor, int imgSize, int maxGray, int minH,
			int maxH);
};

#endif
