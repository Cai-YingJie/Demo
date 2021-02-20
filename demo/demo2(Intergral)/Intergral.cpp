#include "imgproc//imgproc.hpp"
#include "core//core.hpp"
#include "highgui//highgui.hpp"
#include "iostream"
using namespace cv;
using namespace std;

int main()
{
	uchar a[5][5] =
	{
		{ 17, 24, 1, 8, 15 },
		{ 23, 5, 7, 14, 16 },
		{ 4, 6, 13, 20, 22 },
		{ 10, 12, 19, 21, 3 },
		{ 11, 18, 25, 2, 9 },
	};
	Mat src(5, 5, CV_8UC1, &a);
	cout << src << endl;
	Mat dst;
	integral(src, dst);
	cout << dst(Range(1, dst.rows), Range(1, dst.cols)) << endl;
	system("pause");
	return 0;
}