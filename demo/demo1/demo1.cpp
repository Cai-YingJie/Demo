#include "imgproc//imgproc.hpp"
#include "core//core.hpp"
#include "highgui//highgui.hpp"
#include "iostream"
using namespace cv;
using namespace std;

void Cannytest(InputArray _src, OutputArray _dst,
	double low_thresh, double high_thresh,
	int aperture_size, bool L2gradient)
{
	Mat src = _src.getMat();        // 将src转换为Mat格式
	CV_Assert(src.depth() == CV_8U); //确保输入的图像是CV_8U格式

	_dst.create(src.size(), CV_8U);
	Mat dst = _dst.getMat();

	if (!L2gradient && (aperture_size & CV_CANNY_L2_GRADIENT) == CV_CANNY_L2_GRADIENT)
	{
		//backward compatibility
		aperture_size &= ~CV_CANNY_L2_GRADIENT;
		L2gradient = true;
	}

	if ((aperture_size & 1) == 0 || (aperture_size != -1 && (aperture_size < 3 || aperture_size > 7)))
		CV_Error(CV_StsBadFlag, "");

	if (low_thresh > high_thresh)
		std::swap(low_thresh, high_thresh);

#ifdef HAVE_TEGRA_OPTIMIZATION
	if (tegra::canny(src, dst, low_thresh, high_thresh, aperture_size, L2gradient))
		return;
#endif

#ifdef USE_IPP_CANNY
	if (aperture_size == 3 && !L2gradient &&
		ippCanny(src, dst, (float)low_thresh, (float)high_thresh))
		return;
#endif

	const int cn = src.channels();
	Mat dx(src.rows, src.cols, CV_16SC(cn));
	Mat dy(src.rows, src.cols, CV_16SC(cn));

	Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, cv::BORDER_REPLICATE);
	Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0, cv::BORDER_REPLICATE);

	if (L2gradient)
	{
		low_thresh = std::min(32767.0, low_thresh);
		high_thresh = std::min(32767.0, high_thresh);

		if (low_thresh > 0) low_thresh *= low_thresh;
		if (high_thresh > 0) high_thresh *= high_thresh;
	}
	int low = cvFloor(low_thresh);
	int high = cvFloor(high_thresh);

	ptrdiff_t mapstep = src.cols + 2;
	AutoBuffer<uchar> buffer((src.cols + 2)*(src.rows + 2) + cn * mapstep * 3 * sizeof(int));

	int* mag_buf[3];
	mag_buf[0] = (int*)(uchar*)buffer;
	mag_buf[1] = mag_buf[0] + mapstep*cn;
	mag_buf[2] = mag_buf[1] + mapstep*cn;
	memset(mag_buf[0], 0, /* cn* */mapstep*sizeof(int));

	uchar* map = (uchar*)(mag_buf[2] + mapstep*cn);
	memset(map, 1, mapstep);
	memset(map + mapstep*(src.rows + 1), 1, mapstep);

	int maxsize = std::max(1 << 10, src.cols * src.rows / 10);
	std::vector<uchar*> stack(maxsize);
	uchar **stack_top = &stack[0];
	uchar **stack_bottom = &stack[0];

	/* sector numbers
	(Top-Left Origin)

	1   2   3
	*  *  *
	* * *
	0*******0
	* * *
	*  *  *
	3   2   1
	*/

#define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)
#define CANNY_POP(d)     (d) = *--stack_top

	// calculate magnitude and angle of gradient, perform non-maxima suppression.
	// fill the map with one of the following values:
	//   0 - the pixel might belong to an edge
	//   1 - the pixel can not belong to an edge
	//   2 - the pixel does belong to an edge
	for (int i = 0; i <= src.rows; i++)
	{
		int* _norm = mag_buf[(i > 0) + 1] + 1;
		if (i < src.rows)
		{
			short* _dx = dx.ptr<short>(i);
			short* _dy = dy.ptr<short>(i);

			if (!L2gradient)
			{
				for (int j = 0; j < src.cols*cn; j++)
					_norm[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
			}
			else
			{
				for (int j = 0; j < src.cols*cn; j++)
					_norm[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];
			}

			if (cn > 1)
			{
				for (int j = 0, jn = 0; j < src.cols; ++j, jn += cn)
				{
					int maxIdx = jn;
					for (int k = 1; k < cn; ++k)
						if (_norm[jn + k] > _norm[maxIdx]) maxIdx = jn + k;
					_norm[j] = _norm[maxIdx];
					_dx[j] = _dx[maxIdx];
					_dy[j] = _dy[maxIdx];
				}
			}
			_norm[-1] = _norm[src.cols] = 0;
		}
		else
			memset(_norm - 1, 0, /* cn* */mapstep*sizeof(int));

		// at the very beginning we do not have a complete ring
		// buffer of 3 magnitude rows for non-maxima suppression
		if (i == 0)
			continue;

		uchar* _map = map + mapstep*i + 1;
		_map[-1] = _map[src.cols] = 1;

		int* _mag = mag_buf[1] + 1; // take the central row
		ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
		ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

		const short* _x = dx.ptr<short>(i - 1);
		const short* _y = dy.ptr<short>(i - 1);

		if ((stack_top - stack_bottom) + src.cols > maxsize)
		{
			int sz = (int)(stack_top - stack_bottom);
			maxsize = std::max(sz + src.cols, maxsize * 3 / 2);
			stack.resize(maxsize);
			stack_bottom = &stack[0];
			stack_top = stack_bottom + sz;
		}

		int prev_flag = 0;
		for (int j = 0; j < src.cols; j++)
		{
#define CANNY_SHIFT 15
			const int TG22 = (int)(0.4142135623730950488016887242097*(1 << CANNY_SHIFT) + 0.5);

			int m = _mag[j];

			if (m > low)
			{
				int xs = _x[j];
				int ys = _y[j];
				int x = std::abs(xs);
				int y = std::abs(ys) << CANNY_SHIFT;

				int tg22x = x * TG22;

				if (y < tg22x)
				{
					if (m > _mag[j - 1] && m >= _mag[j + 1]) goto __ocv_canny_push;
				}
				else
				{
					int tg67x = tg22x + (x << (CANNY_SHIFT + 1));
					if (y > tg67x)
					{
						if (m > _mag[j + magstep2] && m >= _mag[j + magstep1]) goto __ocv_canny_push;
					}
					else
					{
						int s = (xs ^ ys) < 0 ? -1 : 1;
						if (m > _mag[j + magstep2 - s] && m > _mag[j + magstep1 + s]) goto __ocv_canny_push;
					}
				}
			}
			prev_flag = 0;
			_map[j] = uchar(1);
			continue;
		__ocv_canny_push:
			if (!prev_flag && m > high && _map[j - mapstep] != 2)
			{
				CANNY_PUSH(_map + j);
				prev_flag = 1;
			}
			else
				_map[j] = 0;
		}

		// scroll the ring buffer
		_mag = mag_buf[0];
		mag_buf[0] = mag_buf[1];
		mag_buf[1] = mag_buf[2];
		mag_buf[2] = _mag;
	}

	// now track the edges (hysteresis thresholding)
	while (stack_top > stack_bottom)
	{
		uchar* m;
		if ((stack_top - stack_bottom) + 8 > maxsize)
		{
			int sz = (int)(stack_top - stack_bottom);
			maxsize = maxsize * 3 / 2;
			stack.resize(maxsize);
			stack_bottom = &stack[0];
			stack_top = stack_bottom + sz;
		}

		CANNY_POP(m);

		if (!m[-1])         CANNY_PUSH(m - 1);
		if (!m[1])          CANNY_PUSH(m + 1);
		if (!m[-mapstep - 1]) CANNY_PUSH(m - mapstep - 1);
		if (!m[-mapstep])   CANNY_PUSH(m - mapstep);
		if (!m[-mapstep + 1]) CANNY_PUSH(m - mapstep + 1);
		if (!m[mapstep - 1])  CANNY_PUSH(m + mapstep - 1);
		if (!m[mapstep])    CANNY_PUSH(m + mapstep);
		if (!m[mapstep + 1])  CANNY_PUSH(m + mapstep + 1);
	}

	// the final pass, form the final image
	const uchar* pmap = map + mapstep + 1;
	uchar* pdst = dst.ptr();
	for (int i = 0; i < src.rows; i++, pmap += mapstep, pdst += dst.step)
	{
		for (int j = 0; j < src.cols; j++)
		{
			pdst[j] = (uchar)-(pmap[j] >> 1);
		}
		
	}
}

int main(int argc , char *argv[])
{
	Mat src = imread("C:\\Users\\Administrator\\Desktop\\vs3013\\IMG\\3.jpg", 0);
	Mat dst;
	vector<int> buffer = { 1, 2, 4, 5,5,5,5 };
	for (int i = 0; i < buffer.size(); i++)
	{
		//int a = buffer[(i > 1) + 1];
		int a = 1 << i;
		cout << "a = " << a << endl;
	}
	Cannytest(src, dst, 25, 60, 3, 0);
	Mat kerner = (Mat_<float>(1, 3) << 1, 0, -1);
	Mat filterImg;
	filter2D(dst, filterImg, CV_8U, kerner);
	imshow("dst", dst);
	waitKey(0);
}