#ifndef CTRL_H_
#define CTRL_H_

#include "grabber/grabber.h"
#include "QtCore/qtimer.h"
#include "QtGui/qimage.h"
#include "procThread.h"

class Gui;

class Ctrl : QObject{
	Q_OBJECT
public:
	Ctrl();
	~Ctrl();

	bool onStartCapture(bool capture);


	public slots:
	void receiveFrame(QImage leftRGB, QImage rightRGB, QImage leftDisp, QImage rightDisp, QImage leftDispRaw);

	void updateStereoSettings(int maxDisp,int consistTresh, int kernelSize);
	void enableAutoExp(bool enable);
	void setManualExp(int v);
    void onClickRefreshDevList();

private:
	Gui* gui;
	ProcThread* processingThread;
};
#endif
