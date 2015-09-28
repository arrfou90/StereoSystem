#include "ctrl.h"
#include "gui/gui.h"

Ctrl::Ctrl() {
	gui = new Gui(this);

	gui->setCamDevList(cGrabber::enumDevices());

	processingThread = new ProcThread();

	cv::FileStorage calibrationSettings;

	if(!calibrationSettings.open("mystereocalib_0.17.yml", cv::FileStorage::READ))
		return;

	cv::Mat CM1;
	cv::Mat T;
	calibrationSettings["CM1"] >> CM1;
	calibrationSettings["T"] >> T;

	std::cout << CM1 << std::endl;
	std::cout << CM1.at<double>(0, 0) << " " << CM1.at<double>(1, 1) << " "
			<< CM1.at<double>(0, 2) << " " << CM1.at<double>(1, 2) << " "
			<< std::endl;
	std::cout << "b:" << cv::norm(T) << std::endl;

	gui->setCameraData(CM1.at<double>(0, 0), CM1.at<double>(1, 1),
			CM1.at<double>(0, 2), CM1.at<double>(1, 2), cv::norm(T));

	connect(processingThread,
			SIGNAL(finishedFrame(QImage , QImage, QImage , QImage , QImage )),
			this,
			SLOT(receiveFrame(QImage , QImage, QImage , QImage , QImage )),
			Qt::QueuedConnection);

//	QImage l, r;
//	l.load("tsukubaleft.jpg");
//	r.load("tsukubaright.jpg");
//
//	if (l.format() == QImage::Format_RGB888) {
//		QImage lG(l.width(), l.height(), QImage::Format_Grayscale8);
//		QImage rG(l.width(), l.height(), QImage::Format_Grayscale8);
//		cudaRGBToGray(l.bits(), lG.bits(), l.width() * l.height());
//		cudaRGBToGray(r.bits(), rG.bits(), r.width() * r.height());
//	} else if (l.format() == QImage::Format_Grayscale8) {
//		QImage ldisp(l.width(), l.height(), QImage::Format_Grayscale8);
//		QImage rdisp(l.width(), l.height(), QImage::Format_Grayscale8);
//		gui->displayRGB(l, r);
//		cudaStereoMatch(l.bits(), r.bits(), ldisp.bits(), rdisp.bits(),
//				l.width(), l.height(), 7, 30, 5);
//		gui->displayDisp(ldisp, rdisp);
//	}
}
Ctrl::~Ctrl() {
	processingThread->stop();
	delete processingThread;
}

void Ctrl::updateStereoSettings(int maxDisp, int consistTresh, int kernelSize) {
	processingThread->updateSettings(maxDisp, consistTresh, kernelSize);
}
bool Ctrl::onStartCapture(bool capture) {
	if (capture) {
		if (processingThread->init("mystereocalib_0.17.yml",
				gui->getLeftCamDev(), gui->getRightCamDev())) {

			processingThread->start();
			return true;
		} else {
			return false;
		}
	} else {
		processingThread->stop();
		processingThread->wait();
		processingThread->uninit();
	}
	return true;
}
void Ctrl::receiveFrame(QImage leftRGB, QImage rightRGB, QImage leftDisp,
		QImage rightDisp, QImage leftDispRaw) {
	gui->displayFrame(leftRGB, rightRGB, leftDisp, rightDisp, leftDispRaw);
}

void Ctrl::enableAutoExp(bool enable) {
	processingThread->enableAutoExp(enable);
}
void Ctrl::setManualExp(int v) {
	processingThread->setManualExp(v);
}
void Ctrl::onClickRefreshDevList() {
	gui->setCamDevList(cGrabber::enumDevices());
}
