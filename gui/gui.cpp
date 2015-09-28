#include "gui.h"
#include "../ctrl/ctrl.h"
#include <QtWidgets/qmessagebox.h>
#include <QtCore/qsettings.h>
#include <QtWidgets/qinputdialog.h>

Gui::Gui(Ctrl* ctrl, QWidget *parent) :
		QWidget(parent) {
	this->ctrl = ctrl;
	form.setupUi(this);

	myGlWidget = new MyGlWidget(form.tab_2);
	form.gridLayout_3->addWidget(myGlWidget);

	connect(form.b_startCapture, SIGNAL(clicked()), this,
			SLOT(onClickCapture()));
	connect(form.sbConsist, SIGNAL(valueChanged(int)), this,
			SLOT(onSettingsChanged(int)));
	connect(form.sbKernel, SIGNAL(valueChanged(int)), this,
			SLOT(onSettingsChanged(int)));
	connect(form.sbMaxDisp, SIGNAL(valueChanged(int)), this,
			SLOT(onSettingsChanged(int)));
	connect(form.hs_expCtrlManual, SIGNAL(valueChanged(int)), this,
			SLOT(onChangeManualExp(int)));
	connect(form.cb_autoExposure, SIGNAL(clicked()), this,
			SLOT(onClickAutoExp()));

	connect(form.b_refreshDev, SIGNAL(clicked()), this,
			SLOT(onClickRefreshDevList()));
	connect(form.b_otherCamLeft, SIGNAL(clicked()), this,
			SLOT(onClickOtherLeftCam()));
	connect(form.b_otherCamRight, SIGNAL(clicked()), this,
			SLOT(onClickOtherRightCam()));

	setVisible(true);

}
Gui::~Gui() {

}
void Gui::setCamDevList(vector<cGrabber::CaptureDeviceInfo> devs) {
	form.cb_LeftCam->clear();
	form.cb_RightCam->clear();

	QSettings settings("config.ini", QSettings::IniFormat);
	QString txt1 = settings.value("camLeft").toString();
	QString txt2 = settings.value("camRight").toString();
	int camLeftIdx = -1;
	int camRightIdx = -1;

	for (int i = 0; i < devs.size(); i++) {
		form.cb_LeftCam->addItem(devs.at(i).devName.c_str());
		form.cb_RightCam->addItem(devs.at(i).devName.c_str());

		if(devs.at(i).devName.compare(txt1.toStdString()) == 0)
			camLeftIdx = i;
		if(devs.at(i).devName.compare(txt2.toStdString()) == 0)
			camRightIdx = i;
	}

	if (txt1.compare("") != 0 && camLeftIdx == -1) {
		form.cb_LeftCam->addItem(txt1);
		form.cb_LeftCam->setCurrentText(txt1);
	}else if (camLeftIdx != -1)
	{
		form.cb_LeftCam->setCurrentIndex(camLeftIdx);
	}
	if (txt2.compare("") != 0 && camRightIdx == -1) {
		form.cb_RightCam->addItem(txt2);
		form.cb_RightCam->setCurrentText(txt2);
	}else if (camRightIdx != -1)
	{
		form.cb_RightCam->setCurrentIndex(camRightIdx);
	}
}

void Gui::displayFrame(QImage &left, QImage &right, QImage &leftDispColor,
		QImage &rightDispColor, QImage& leftDispRaw) {
	form.l_rawLeft->setPixmap(QPixmap::fromImage(left));
	form.l_rawRight->setPixmap(QPixmap::fromImage(right));
	form.l_dispLeft->setPixmap(QPixmap::fromImage(leftDispColor));
	form.l_dispRight->setPixmap(QPixmap::fromImage(rightDispColor));

	myGlWidget->setData(left, leftDispRaw);
}

void Gui::onClickCapture() {

	bool startCapture = form.b_startCapture->text().compare("Capture") == 0;
	if (ctrl->onStartCapture(startCapture)) {

		QSettings settings("config.ini", QSettings::IniFormat);
		settings.setValue("camLeft",getLeftCamDev().c_str());
		settings.setValue("camRight",getRightCamDev().c_str());
		if (startCapture) {
			onSettingsChanged(0);
			form.b_startCapture->setText("Stop");
		} else {
			form.b_startCapture->setText("Capture");
		}
	} else {
//		QMessageBox messageBox;
//		messageBox.critical(0, "Error", "Can't open CamDevices !");
	}
}
void Gui::onSettingsChanged(int i) {
	ctrl->updateStereoSettings(form.sbMaxDisp->value(), form.sbConsist->value(),
			form.sbKernel->value());
}
void Gui::onClickAutoExp() {
	ctrl->enableAutoExp(form.cb_autoExposure->isChecked());
}
void Gui::onChangeManualExp(int v) {
	ctrl->setManualExp(form.hs_expCtrlManual->value());
}

void Gui::onClickRefreshDevList() {
	ctrl->onClickRefreshDevList();
}
void Gui::onClickOtherLeftCam() {
	bool ok;
	QString text = QInputDialog::getText(this,"LeftCam",
			"LeftCam-DeviceName:", QLineEdit::Normal, "/dev", &ok);
	if (ok && !text.isEmpty()){
		form.cb_LeftCam->addItem(text);
		form.cb_LeftCam->setCurrentText(text);
	}
}
void Gui::onClickOtherRightCam() {
	bool ok;
	QString text = QInputDialog::getText(this,"RightCam",
			"RightCam-DeviceName:", QLineEdit::Normal, "/dev", &ok);
	if (ok && !text.isEmpty()){
		form.cb_RightCam->addItem(text);
		form.cb_RightCam->setCurrentText(text);
	}
}
