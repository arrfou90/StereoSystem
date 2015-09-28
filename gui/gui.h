#ifndef GUI_H_
#define GUI_H_

#include "ui_mainform.h"
#include <QtWidgets/qwidget.h>
#include <vector>
#include "widgets/myGlWidget.h"

#include "../grabber/grabber.h"
using namespace std;

class Ctrl;

class Gui: public QWidget {

Q_OBJECT

public:
	Gui(Ctrl* ctrl, QWidget *parent = 0);
	~Gui();

	void setCamDevList(vector<cGrabber::CaptureDeviceInfo> devs);
	void displayFrame(QImage &left, QImage &right, QImage &leftDispColor, QImage &rightDispColor, QImage& leftDispRaw);

	string getLeftCamDev()
	{
		return form.cb_LeftCam->currentText().toStdString();
	}
	string getRightCamDev()
	{
		return form.cb_RightCam->currentText().toStdString();
	}
	void setCameraData(float fX,float fY, float cX, float cY, float b){
		myGlWidget->setCameraData(fX,fY,cX,cY, b);
	}


public slots:
    void onClickCapture();
    void onSettingsChanged(int i);
    void onClickAutoExp();
    void onChangeManualExp(int v);
    void onClickRefreshDevList();
    void onClickOtherLeftCam();
    void onClickOtherRightCam();

private:
	Ui::Form form;
	Ctrl* ctrl;
	MyGlWidget* myGlWidget;
};
#endif
