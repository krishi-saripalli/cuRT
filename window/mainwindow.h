#pragma once

#include <QMainWindow>
#include <QImage>
#include <QLabel>
#include "utils/rgba.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(int width, int height, QWidget *parent = nullptr)
        : QMainWindow(parent), m_image(width, height, QImage::Format_RGBX8888)
    {
        m_image.fill(Qt::black);
        m_imageLabel = new QLabel(this);
        m_imageLabel->setMinimumSize(width, height);
        setCentralWidget(m_imageLabel);
        updateDisplay();
    }

    void updatePixel(int x, int y, const RGBA &color)
    {
        if (x >= 0 && x < m_image.width() && y >= 0 && y < m_image.height())
        {
            m_image.setPixelColor(x, y, QColor(color.r, color.g, color.b, color.a));
        }
    }

    void updateDisplay()
    {
        m_imageLabel->setPixmap(QPixmap::fromImage(m_image));
    }

private:
    QImage m_image;
    QLabel *m_imageLabel;
};