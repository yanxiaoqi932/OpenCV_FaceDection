#include "opencvscene.h"
#include "mypushbutton.h"
#include <QDialog>
#include <QMessageBox>
#include <QLabel>
#include <QFont>
#include <QTimer>
#include <QDir>
#include <QDebug>
using namespace cv;

extern QString IDstring;
QString ID_opencv;  //人脸识别函数与widget界面共享的数据，用于传递识别出的用户的ID
bool IS_Land_OK = 0;  //人脸识别函数与widget界面共享的数据，用于判断是否识别成功
opencvScene::opencvScene()
{
    qDebug()<<IDstring;
    this->setFixedSize(1600,900);
    //创建开始拍摄的按钮
    MyPushButton *TakeAPhotoBtn = new MyPushButton;
    TakeAPhotoBtn->setParent(this);
    TakeAPhotoBtn->setText("开始拍摄");
    TakeAPhotoBtn->setFixedSize(120,55);
    TakeAPhotoBtn->move(this->width()*0.55,this->height()*0.3);

    //创建拍摄提示
    QFont font2;
    font2.setFamily("华文新魏");
    font2.setPointSize(15);
    QLabel *picEnd = new QLabel;
    picEnd->setParent(this);
    picEnd->setFont(font2);
    picEnd->move(this->width()*0.35,this->height()*0.5);
    picEnd->setText("请正对摄像头，点击拍摄按钮，进行人脸采集");

    //创建注册完成按钮
    MyPushButton *TakePhotoEndBtn = new MyPushButton;
    TakePhotoEndBtn->setParent(this);
    TakePhotoEndBtn->setText("退出");
    TakePhotoEndBtn->setFixedSize(120,55);
    TakePhotoEndBtn->move(this->width()*0.55,this->height()*0.7);

    //点击开始拍摄按钮
    connect(TakeAPhotoBtn,&MyPushButton::clicked,[=](){
        on_TakeAPhotoBtn_clicked();    //开始拍摄十组原始照片储存在addpicture文件夹中
        disposePic();     //将原始照片处理转换成灰度图片存储在allpicture/%idstring文件夹中
        MakecsvFile();    //生成储存灰度图片绝对路径的csv文件，放在allpicture文件夹中
        TrainingModel();  //训练每个人的人脸图片，并得到特征生成xml文件
        picEnd->clear();
        picEnd->setText("拍摄完成");
        TakePhotoEndBtn->setText("完成注册");
    });

    //点击完成注册按钮
    connect(TakePhotoEndBtn,&MyPushButton::clicked,[=](){
        QTimer::singleShot(300,this,[=](){
            emit this->opencvBckBtn();
        });

    });

}

void opencvScene::on_TakeAPhotoBtn_clicked()
{
    capture = new cv::VideoCapture(0);  //用于打开摄像头API
    int i = 0;

    //拍摄十组人脸照片并储存
    while (i!=10) {
        *capture >>frame;
        imshow("frame",frame);  //"frame":新建的窗口名称;frame:显示的图像
        //原始拍摄图片的存储路径
        std::string filename = cv::format("AllData\\AddPicture\\%d.jpg",i+1);
        cv::waitKey(1000);
        imwrite(filename,frame);
        imshow("photo",frame);
        cv::waitKey(500);
        cv::destroyWindow("photo");
        i++;
    }
    //关闭摄像头以及图像采集界面
    capture->release();
    cv::destroyWindow("frame");

}

//对原始人脸图片进行处理
void opencvScene::disposePic()
{
    //创建targetFile的目标文件夹
    QDir dir;
    if(!dir.exists("AllData\\AllPicture\\"+IDstring))
    {
        dir.mkpath("AllData\\AllPicture\\"+IDstring);
    }
    QString sourceFilePath="AllData\\AddPicture\\";
    QString targetFilePath="AllData\\AllPicture\\"+IDstring + "\\";
    QString sourceFile;
    QString targetFile;
    for(int i=0;i<10;i++)
    {
        sourceFile.append(sourceFilePath+QString::number(i+1,10));
        sourceFile.append(".jpg");
        targetFile.append(targetFilePath);
        targetFile.append(QString::number(i,10));
        targetFile.append(".jpg");
        qDebug()<<targetFile;
        this->detectAndDisplay(sourceFile,targetFile);
        sourceFile.clear();
        targetFile.clear();
    }
}

//原始图片的具体处理函数
void opencvScene::detectAndDisplay(QString source, QString target)
{
    std::string face_cascade_name = "haarcascade_frontalface_alt.xml";
    cv::CascadeClassifier face_cascade;   //定义人脸分类器

    cv::Mat frame = cv::imread(source.toStdString());
    if(!frame.data)
    {
        qDebug()<<source;
        QMessageBox::warning(this,tr("提示"),tr("frame读取失败"),QMessageBox::Ok);
        return;
    }
    if (!face_cascade.load(face_cascade_name))
    {
        QMessageBox::warning(this,tr("错误"),tr("haarcascade_frontalface_alt.xml加载失败"),QMessageBox::Ok);
        return;
    }
    std::vector<cv::Rect> faces;
    cv::Mat img_gray;

    cv::cvtColor(frame, img_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(img_gray, img_gray);

    face_cascade.detectMultiScale(img_gray, faces, 1.1, 3, CV_HAAR_DO_ROUGH_SEARCH, cv::Size(50, 50));

    for (int j = 0; j < (int)faces.size(); j++)
    {
        cv::Mat faceROI = frame(faces[j]);
        cv::Mat MyFace;
        cv::Mat gray_MyFace;
        qDebug()<<faceROI.cols;
        if (faceROI.cols > 100)
        {
            cv::resize(faceROI, MyFace, cv::Size(92, 112));
            cv::cvtColor(MyFace, gray_MyFace, CV_BGR2GRAY);
            imwrite(target.toStdString(), gray_MyFace);
        }
    }
}

//生成csv文件，储存图片的绝对路径
void opencvScene::MakecsvFile()
{
    QDir csvFile("AllData/AllPicture/at.txt");
    QString csvPath=csvFile.absolutePath();
    QString csvFilePath=csvPath;
    csvPath.chop(6);    //删除csvpath后面的六个字符
    QString path=csvPath+IDstring+"/";
    for(int i=0;i<10;i++)
    {
        QString filepath=path;
        filepath.append(QString::number(i,10));
        filepath.append(".jpg;");
        filepath.append(IDstring);
        //this->AddPeople(csvFilePath,filepath);
        QFile file(csvFilePath);
        if(!file.open(QIODevice::WriteOnly|QIODevice::Append))
        {
            QMessageBox about;
            about.setText(tr("添加人员时文件打开失败"));
            about.exec();
            return;
        }
        QTextStream in(&file);
        //in.setCodec("UTF-8");
        in<<filepath<<"\r\n";
        file.close();
    }
}

//训练图片，记录每个人的人脸特征，存储到xml文件中
void opencvScene::TrainingModel()
{
    //读取你的CSV文件路径.
    //string fn_csv = string(argv[1]);
    std::string fn_csv = "AllData\\AllPicture\\at.txt";

    // 2个容器来存放图像数据和对应的标签
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    // 读取数据. 如果文件不合法就会出错
    // 输入的文件名已经有了.
    try
    {
        read_csv(fn_csv, images, labels);
    }
    catch (cv::Exception& e)
    {
        std::cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // 文件有问题，我们啥也做不了了，退出了
        return;
    }
    // 如果没有读取到足够图片，也退出.
    if (images.size() <= 1) {
        std::string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }

    // 下面的几行代码仅仅是从你的数据集中移除最后一张图片
    //[gm:自然这里需要根据自己的需要修改，他这里简化了很多问题]
    cv::Mat testSample = images[images.size() - 1];
//    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();

    cv::createEigenFaceRecognizer(0, 123.0);
    cv::Ptr<cv::FaceRecognizer> model = cv::createEigenFaceRecognizer();
    model->train(images, labels);
    model->save("MyFacePCAModel.xml");

          int predictedLabel = -1;
          double confidence = 0.0;
          model->predict(testSample, predictedLabel, confidence);
          qDebug()<<confidence;
}

void opencvScene::read_csv(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels)
{
    char separator = ';';
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file) {
        QMessageBox::warning(NULL,tr("错误"),tr("read_csv文件打开失败"),QMessageBox::Ok);
        return;
    }
    std::string line, path, classlabel;
    while (std::getline(file, line)) {
        std::stringstream liness(line);
        std::getline(liness, path, separator);
        std::getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty()) {
            images.push_back(cv::imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

void opencvScene::on_action_FaceRecognition_triggered()
{
    int label=0;
    double confidence=0.0;
    cv::VideoCapture cap(0);    //打开默认摄像头
    if (!cap.isOpened())
    {
        QMessageBox::warning(this,tr("错误"),tr("摄像头打开失败"),QMessageBox::Ok);
        return;
    }
    cv::Mat frame;
    cv::Mat gray;

    cv::CascadeClassifier cascade;
    bool stop = false;
    //训练好的文件名称，放置在可执行文件同目录下
    cascade.load("haarcascade_frontalface_alt.xml");

    cv::Ptr<cv::FaceRecognizer> modelPCA = cv::createEigenFaceRecognizer();
    modelPCA->load("MyFacePCAModel.xml");
    int sl=0;
    while (!stop)
    {
        cap >> frame;

        //建立用于存放人脸的向量容器
        std::vector<cv::Rect> faces(0);

        cv::cvtColor(frame, gray, CV_BGR2GRAY);
        //改变图像大小，使用双线性差值
        //resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
        //变换后的图像进行直方图均值化处理
        cv::equalizeHist(gray, gray);

        cascade.detectMultiScale(gray, faces,
            1.1, 2,cv::CASCADE_FIND_BIGGEST_OBJECT|cv::CASCADE_DO_ROUGH_SEARCH,
            cv::Size(30, 30));

        cv::Mat face;
        cv::Point text_lb;

        for (size_t i = 0; i < faces.size(); i++)
        {
            if (faces[i].height > 0 && faces[i].width > 0)
            {
                face = gray(faces[i]);
                text_lb = cv::Point(faces[i].x, faces[i].y);
                cv::rectangle(frame, faces[i], cv::Scalar(255, 0, 0), 1, 8, 0);
            }
        }

        cv::Mat face_test;

        int predictPCA = 0;
        if (face.rows >= 120)
        {
            cv::resize(face, face_test, cv::Size(92, 112));

        }
        //Mat face_test_gray;
        //cvtColor(face_test, face_test_gray, CV_BGR2GRAY);

        if (!face_test.empty())
        {
            //测试图像应该是灰度图
            int predictedLabel=-1;
            predictPCA = modelPCA->predict(face_test);
            modelPCA->predict(face_test,predictedLabel,confidence);
            qDebug()<<"predictedLabel:"<<predictedLabel;
            qDebug()<<"confidence:"<<confidence;
            qDebug()<<"s1:"<<sl;

        }
        if(sl>40)
        {
            if(label<5)
                QMessageBox::information(this,tr("失败"),tr("人脸库无此人"),QMessageBox::Ok);
            else
                QMessageBox::information(this,tr("失败"),tr("人脸确认度低"),QMessageBox::Ok);

            //关闭摄像头以及图像采集界面
            cap.release();
            cv::destroyWindow("face");

            IS_Land_OK = false;
            return;
        }
        if(predictPCA!=-1&&predictPCA!=1&&confidence<3200)
            label++;
        if(predictPCA==1||predictPCA==0)
            label--;
        qDebug()<<"label:"<<label;
        cv::waitKey(50);
        if(label>12)
        {
            //qDebug()<<predictPCA;
            //std::string name = "Being recognized";
            //cv::putText(frame, name, text_lb, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
            //qDebug()<<"姓名"<<addpeople.file.who[predictPCA];
            qDebug()<<predictPCA;
            //关闭摄像头以及图像采集界面
            cap.release();
            cv::destroyWindow("face");

            ID_opencv = QString::number(predictPCA,10);  //十进制int型数据转为QString
            IS_Land_OK = true;
            return;
        }
        sl++;

        imshow("face", frame);
        if (cv::waitKey(50) >= 0)
            stop = true;
    }
}

