#include <QtWidgets/qapplication.h>
#include <QtCore/qobject.h>
#include "ctrl/ctrl.h"

//#include "kernels.h"
#include <iostream>

int main(int argc, char **argv) {
	QApplication app(argc, argv);
	new Ctrl();
	return app.exec();
}
