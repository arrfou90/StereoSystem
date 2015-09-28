#ifndef MYGLWIDGET_H_
#define MYGLWIDGET_H_

#include <QtOpenGL/qgl.h>
#include <QtGui/qevent.h>
#include "iostream"

class MyGlWidget : public QGLWidget
{
    Q_OBJECT

public:
    MyGlWidget(QWidget *parent = 0);

    void setData(QImage img, QImage disp){
    	this->img = img;
    	this->disp = disp;
    	repaint();
    }
    void setCameraData(float fX,float fY, float cX, float cY, float b){
    	this->fX = fX;
    	this->fY = fY;
    	this->cX = cX;
    	this->cY = cY;
    	this->b = b;

    }

protected:
    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
//    void mouseDoubleClickEvent(QMouseEvent *event);

private:
    void draw();
//    int faceAtPosition(const QPoint &pos);

    float fX;
    float fY;
    float cX;
    float cY;
    float b;
    QImage img;

    QImage disp;
    GLfloat rotationX;
    GLfloat rotationY;
    GLfloat rotationZ;
    GLfloat translateX;
    QPoint lastPos;
};
#endif
