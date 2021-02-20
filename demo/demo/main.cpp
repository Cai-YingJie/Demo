#include "core/core.hpp"  
#include "highgui/highgui.hpp"  
#include "imgproc/imgproc.hpp"  
#include "iostream"
#include "pmmintrin.h"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;


// 计算一幅图的局部标准差
void calcDeviationImage(Mat inputImg, Mat& devImg, int maskWidth, int maskHeight)
{
	Mat orignalImg;
	inputImg.convertTo(orignalImg, CV_32FC1);

	// 计算均值图像及均值图像的平方图像
	Mat meanImg, meanImg_pow;
	boxFilter(orignalImg, meanImg, CV_32FC1, Size(maskWidth, maskHeight));
	pow(meanImg, 2, meanImg_pow);

	// 计算原图的平方图像及该图像的均值图像
	Mat orignalImg_pow, orignalImg_pow_mean;
	pow(orignalImg, 2, orignalImg_pow);
	boxFilter(orignalImg_pow, orignalImg_pow_mean, CV_32FC1, Size(maskWidth, maskHeight));

	// 作差然后开根号，得出标准差图像
	Mat varianceImg;	//方差图像

	subtract(orignalImg_pow_mean, meanImg_pow, varianceImg);
	sqrt(varianceImg, devImg);

	// 转化为CV_8UC1格式
	//devImg.convertTo(devImg, CV_8UC1);
}

void invert(Mat &img, const uchar* const table)
{
	for (int i = 0; i < img.rows; i++)
	{
		uchar* temp = img.ptr<uchar>(i);
		for (int j = 0; j < img.cols; j++)
		{
			temp[j] = table[temp[j]];
		}
	}
}

void ACE(Mat src, Mat &des, const int D)
{
	Mat inputImg;
	src.convertTo(inputImg, CV_32FC1);

	//计算均值图像
	Mat meanImg;
	boxFilter(inputImg, meanImg, CV_32FC1, Size(3, 3));

	// 计算局部标准差
	Mat devImg;
	calcDeviationImage(src, devImg, 3, 3);
	// 全局标准差
	//calcDeviationImage(src, devImg, src.rows, src.cols);

	// 计算CG参数
	Mat weightImg = Mat::zeros(devImg.rows, devImg.cols, CV_32FC1);;
	for (int i = 0; i < weightImg.rows; i++)
	{
		float *tempWeiData = weightImg.ptr<float>(i);
		float *tempDevData = devImg.ptr<float>(i);
		for (int j = 0; j < weightImg.cols; j++)
		{
			tempWeiData[j] = (float)D / (tempDevData[j] + 0.00001);
			//tempWeiData[j] = (float)D *(tempDevData[j] + 0.00001);
		}
	}
	Mat meanDifferenceImg;

	// 计算原图与均值图像的差
	subtract(inputImg, meanImg, meanDifferenceImg);

	Mat weightingImg;
	weightingImg = weightImg.mul(meanDifferenceImg);

	Mat outputImg;
	add(weightingImg, meanImg, outputImg);

	weightingImg.convertTo(des, CV_8U);
}

//void canny(Mat src, Mat &dst, double lowThresh, double higeThresh, int apertureSize, bool L2Gradient)
//{
//	Mat inputImg;
//	inputImg.copyTo(src);
//	CV_Assert(inputImg.depth == CV_8U);
//	Mat outputImg;
//	outputImg.create(inputImg.size(), CV_8U);
//	if (!L2Gradient && (apertureSize & CV_CANNY_L2_GRADIENT) == CV_CANNY_L2_GRADIENT)
//	{
//		apertureSize &= ~CV_CANNY_L2_GRADIENT;
//		L2Gradient = true;
//	}
//
//}
int main(int argc, char *argv[])
{

	Mat src, gray, colorEdge;
	//src = imread("C:\\Users\\Administrator\\Desktop\\vs3013\\IMG\\color.jpg");
	src = imread("C:\\Users\\Administrator\\Desktop\\vs3013\\IMG\\1.jpg", 0);
	if (!src.data)
	{
		return -1;
	}
	Mat inputImg;
	//cvtColor(src, inputImg, CV_RGB2BGR);
	//calcDeviationImage(src, gray, 10,10);
	//uchar table[256];
	//for (int i = 0; i < 256; i++)
	//{
	//	table[i] = 255 - i;
	//}
	//invert(src, table);

	//Rect roi;
	//roi.x = 50;
	//roi.y = 10;
	//roi.width = 10;
	//roi.height = 50;
	//Mat subMat = src(roi);
	//subMat.setTo(0);
	Mat ACEImg;
	Mat ACEImg2;
	//src = ~src;
	boxFilter(src, src, CV_32FC1, Size(4, 4));
	boxFilter(src, src, CV_32FC1, Size(3, 3));
	ACE(src, ACEImg, 3);

	Mat src_gray;
	cvtColor(ACEImg, src_gray, CV_BGR2GRAY);
	int blockSize = 6;
	int apertureSize = 3;
	double k = 0.04;
	int thresh = 110;
	Mat dst, dst_norm, dst_norm_scaled;

	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > thresh)
			{
				circle(src, Point(i, j), 1, Scalar(0, 0, 255), 1, 8, 0);
			}
		}
	}
	imshow("img", ACEImg);
	waitKey(0);
	return 0;
}