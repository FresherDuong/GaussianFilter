#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

// Khởi tạo giá trị tần số cắt D, độ rộng dải W măc định 
int D = 20;
int W = 20;

const String trackBarNameBandpass = "Bo loc Gaussian Bandpass theo tan so cat va do rong dai";
const String trackBarNameBandstop = "Bo loc Gaussian Bandstop theo tan so cat va do rong dai";
Mat src;

// Sắp xếp lại các góc phần tư của hình ảnh Fourier sao cho điểm gốc nằm ở trung tâm hình ảnh
void shiftDFT(Mat& fImage)
{
	Mat tmp, q0, q1, q2, q3;

	// Đầu tiên là cắt hình ảnh, nếu nó có số lượng hàng hoặc cột lẻ
	fImage = fImage(Rect(0, 0, fImage.cols & -2, fImage.rows & -2));

	int cx = fImage.cols / 2;
	int cy = fImage.rows / 2;

	// Sắp xếp lại các góc phần tư của hình ảnh Fourier sao cho điểm gốc nằm ở trung tâm hình ảnh

	q0 = fImage(Rect(0, 0, cx, cy));
	q1 = fImage(Rect(cx, 0, cx, cy));
	q2 = fImage(Rect(0, cy, cx, cy));
	q3 = fImage(Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

// Tạo ảnh padded
Mat create_padded_image(Mat src)
{
	Mat padded;
	int m = getOptimalDFTSize(src.rows);
	int n = getOptimalDFTSize(src.cols);
	copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
	return padded;
}


// Tạo H(u,v) cho Gaussian Bandpass filter (tham số truyền vào là tần sô cắt d0 và độ rộng dải w)
void gaussian_bandpass_filter(Mat& dftFilter, double d0, double w)
{
	Mat tmp = Mat(dftFilter.rows, dftFilter.cols, CV_32F);
	Point center = Point(dftFilter.rows / 2, dftFilter.cols / 2);
	double d1 = d0 * d0;
	for (int i = 0; i < dftFilter.rows; i++)
	{
		for (int j = 0; j < dftFilter.cols; j++)
		{
			// Công thức của bộ lọc
			double d = pow((double)(i - center.x), 2) + pow((double)(j - center.y), 2);
			tmp.at<float>(i, j) = (float)expf((-1.0) * pow(((d - d1) / (sqrt(d) * w)), 2));
		}
	}

	Mat toMerge[] = { tmp, tmp };
	merge(toMerge, 2, dftFilter);
}

// Tạo H(u,v) cho Gaussian Bandstop filter (tham số truyền vào là tần sô cắt d0 và độ rộng dải w)
void gaussian_bandstop_filter(Mat& dftFilter, double d0, double w)
{
	Mat tmp = Mat(dftFilter.rows, dftFilter.cols, CV_32F);
	Point center = Point(dftFilter.rows / 2, dftFilter.cols / 2);
	double d1 = d0 * d0;
	for (int i = 0; i < dftFilter.rows; i++)
	{
		for (int j = 0; j < dftFilter.cols; j++)
		{
			// Công thức của bộ lọc
			double d = pow((double)(i - center.x), 2) + pow((double)(j - center.y), 2);
			tmp.at<float>(i, j) = (float)(1.0 - expf((-1.0) * pow(((d - d1) / (sqrt(d) * w)), 2)));
		}
	}

	Mat toMerge[] = { tmp, tmp };
	merge(toMerge, 2, dftFilter);
}

// Áp bộ lọc vào ảnh
Mat apply_filter(Mat img, Mat filter)
{
	Mat padded = create_padded_image(img);

	// Setup ảnh DFT
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);

	// Thực hiện DFT
	dft(complexI, complexI);

	// Apply filter
	shiftDFT(complexI);
	mulSpectrums(complexI, filter, complexI, 0);
	shiftDFT(complexI);

	// Thực hiện inverse DFT ở ảnh đã được filtered
	idft(complexI, complexI);

	// Trả về hình ảnh output
	split(complexI, planes);
	Mat magI = planes[0];
	normalize(magI, magI, 0, 1, NORM_MINMAX);

	return magI;
}

// Thực hiện bộ lọc Bandpass filter 
void apply_bandpass_filter(int d, int w) {
	Mat gauss_bandpass = src.clone();
	gaussian_bandpass_filter(gauss_bandpass, d, w);
	Mat dst_gauss_bandpass = apply_filter(src, gauss_bandpass);
	imshow(trackBarNameBandpass, dst_gauss_bandpass);
}

// Thực hiện bộ lọc Bandstop filter 
void apply_bandstop_filter(int d, int w) {
	Mat gauss_bandstop = src.clone();
	gaussian_bandstop_filter(gauss_bandstop, d, w);
	Mat dst_gauss_bandstop = apply_filter(src, gauss_bandstop);
	imshow(trackBarNameBandstop, dst_gauss_bandstop);
}

// Hàm Callback khi chỉnh Trackbar ở ảnh thực hiện Bandpass filter
void onChangeBandpass(int sliderValue, void* userData) {
	int dValue = getTrackbarPos("TanSoCat", trackBarNameBandpass);
	int wValue = getTrackbarPos("DoRongDai", trackBarNameBandpass);
	
	apply_bandpass_filter(dValue, wValue);
}

// Hàm Callback khi chỉnh Trackbar ở ảnh thực hiện Bandstop filter
void onChangeBandstop(int sliderValue, void* userData) {
	int dValue = getTrackbarPos("TanSoCat", trackBarNameBandstop);
	int wValue = getTrackbarPos("DoRongDai", trackBarNameBandstop);

	apply_bandstop_filter(dValue, wValue);
}

int main(int argc, char** argv)
{
	// Ảnh gốc
	src = imread("input_for_gaussian.png", IMREAD_GRAYSCALE);
	imshow("Anh ban dau", src);

	// Thực hiện Gaussian Bandpass Filter với độ rộng dải W và tần số cắt D
	namedWindow(trackBarNameBandpass, 1);
	createTrackbar("TanSoCat", trackBarNameBandpass, &D, 50, onChangeBandpass);
	createTrackbar("DoRongDai", trackBarNameBandpass, &W, 50, onChangeBandpass);
	apply_bandpass_filter(D, W);

	// Thực hiện Gaussian Bandstop Filter với độ rộng dải W và tần số cắt D
	namedWindow(trackBarNameBandstop, 1);
	createTrackbar("TanSoCat", trackBarNameBandstop, &D, 50, onChangeBandstop);
	createTrackbar("DoRongDai", trackBarNameBandstop, &W, 50, onChangeBandstop);
	apply_bandstop_filter(D, W);

	waitKey();
}