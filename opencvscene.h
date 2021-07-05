#ifndef OPENCVSCENE_H
#define OPENCVSCENE_H

#include <QWidget>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>

class opencvScene:public QWidget
{
    Q_OBJECT
public:
    explicit opencvScene();
    void on_TakeAPhotoBtn_clicked();   //第一层，按键后执行的各类任务
    void disposePic(void);      //第二层，对原始图片进行处理
    void detectAndDisplay(QString source, QString target);  //第三层，对原始图片具体处理
    void MakecsvFile(void);     //生成储存灰度图片绝对路径的csv文件
    void TrainingModel(void);   //训练并记录每个人的人脸特征
    void read_csv(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels);//读取csv文件
    void on_action_FaceRecognition_triggered(void);
private:
    cv::Mat frame;
    cv::VideoCapture *capture;

signals:
    void opencvBckBtn(void);

};

#endif // OPENCVSCENE_H
