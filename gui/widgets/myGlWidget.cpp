#include "myGlWidget.h"
#include "GL/freeglut.h"

MyGlWidget::MyGlWidget(QWidget *parent)
    : QGLWidget(parent)
{
    setFormat(QGLFormat(QGL::DoubleBuffer | QGL::DepthBuffer));

    rotationX = -21.0;
    rotationY = -57.0;
    rotationZ = 0.0;
    translateX = 0;
}
void MyGlWidget::initializeGL()
{
    qglClearColor(Qt::black);
    glShadeModel(GL_FLAT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);


}void MyGlWidget::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
//    GLfloat x = GLfloat(width) / height;
//    glFrustum(-x, +x, -1.0, +1.0, 0.1, 100.0);
//    glOrtho(0,width, height,0,1,-1);

//    float aspect = width / height;
//    glViewport(0, 0, width, height);
//    glOrtho(-50.0 * aspect, 50.0 * aspect, -50.0, 50.0, 1.0, -1.0);
	gluPerspective( /* field of view in degree */40.0,
	/* aspect ratio */1.0,
	/* Z near */1.0, /* Z far */200.0);

    glMatrixMode(GL_MODELVIEW);
}void MyGlWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    draw();
}void MyGlWidget::draw()
{
    static const GLfloat P1[3] = { 0.0, -1.0, +2.0 };
    static const GLfloat P2[3] = { +1.73205081, -1.0, -1.0 };
    static const GLfloat P3[3] = { -1.73205081, -1.0, -1.0 };
    static const GLfloat P4[3] = { 0.0, +2.0, 0.0 };

    static const GLfloat * const coords[4][3] = {
        { P1, P2, P3 }, { P1, P3, P4 }, { P1, P4, P2 }, { P2, P4, P3 }
    };

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    gluLookAt(0.0, 0.0, 100.0,  /* eye is at (0,0,5) */
      0.0, 0.0, 0.0,      /* center is at (0,0,0) */
      0.0, 1.0, 0.);      /* up is in positive Y direction */

    glRotatef(rotationX, 1.0, 0.0, 0.0);
    glRotatef(rotationY, 0.0, 1.0, 0.0);
    glRotatef(rotationZ, 0.0, 0.0, 1.0);
    glTranslatef(translateX, 0.0, 0.0);
	glScalef(0.1,0.1,1);

//    for (int i = 0; i < 4; ++i) {
//        glLoadName(i);
//        glBegin(GL_TRIANGLES);
//        qglColor(faceColors[i]);
//        for (int j = 0; j < 3; ++j) {
//            glVertex3f(coords[i][j][0], coords[i][j][1],
//                       coords[i][j][2]);
//        }
//        glEnd();
//    }

	if (!img.isNull() && !disp.isNull())
	{
		if (disp.format() == QImage::Format_Grayscale8) {
			glBegin(GL_POINTS);
			if (img.format() == QImage::Format_RGB888) {
				unsigned char* imgData = img.bits();
				unsigned char* dispData = disp.bits();
				for (int y = 0; y < img.height(); y++) {
					for (int x = 0; x < img.width(); x++) {

						unsigned char r,g,b;
						r= *imgData++;
						g= *imgData++;
						b= *imgData++;
						glColor3ub(r, g, b);

						float disp = *dispData++;

						float z;

						if (disp > 0) {
							z = fX/disp;
							float tmpx = (x - cX) * z / fX;
							float tmpy = (y - cY) * z / fY;

							glVertex3f(x - img.width() / 2,
									y - img.height() / 2, z);
						}
					}
				}
			}
			else if (img.format() == QImage::Format_Grayscale8) {
				unsigned char* imgData = img.bits();
				unsigned char* dispData = disp.bits();
				for (int y = 0; y < img.height(); y++) {
					for (int x = 0; x < img.width(); x++) {

						unsigned char r, g, b;
						r = *imgData;
						g = *imgData;
						b = *imgData++;
						glColor3ub(r, g, b);

						float disp = *dispData++;

						float z;

						if (disp > 0) {
							z = fX / disp;
							float tmpx = (x - cX) * z / fX;
							float tmpy = (y - cY) * z / fY;

							glVertex3f(x - img.width() / 2,
									y - img.height() / 2, z);
						}
					}
				}
			}
			glEnd();
		}else{
			std::cout << "kein Grayscale" << std::endl;
		}
	}
}

void MyGlWidget::mousePressEvent(QMouseEvent *event)
{
		lastPos = event->pos();
}

void MyGlWidget::mouseMoveEvent(QMouseEvent *event)
{
    GLfloat dx = GLfloat(event->x() - lastPos.x()) / width();
	GLfloat dy = GLfloat(event->y() - lastPos.y()) / height();
	if ((event->buttons() & Qt::LeftButton) &&  (event->buttons() & Qt::RightButton)) {
		translateX += 50*dx;
		updateGL();
	} else if (event->buttons() & Qt::LeftButton) {
		rotationX += 180 * dy;
		rotationY += 180 * dx;
		updateGL();
	} else if (event->buttons() & Qt::RightButton) {
		rotationX += 180 * dy;
		rotationZ += 180 * dx;
		updateGL();
	}
    lastPos = event->pos();
}
