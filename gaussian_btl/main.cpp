#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

int D = 20;
int W = 20;
const String trackBarNameBandpass = "Bo loc Gaussian Bandpass theo tan so cat va do rong dai";
const String trackBarNameBandstop = "Bo loc Gaussian Bandstop theo tan so cat va do rong dai";
Mat src;

// Rearrange the quadrants of a Fourier image so that the origin is at the image center
void shiftDFT(Mat& fImage)
{
	Mat tmp, q0, q1, q2, q3;

	// first crop the image, if it has an odd number of rows or columns

	fImage = fImage(Rect(0, 0, fImage.cols & -2, fImage.rows & -2));

	int cx = fImage.cols / 2;
	int cy = fImage.rows / 2;

	// rearrange the quadrants of Fourier image
	// so that the origin is at the image center

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

Mat create_padded_image(Mat src)
{
	Mat padded;
	int m = getOptimalDFTSize(src.rows);
	int n = getOptimalDFTSize(src.cols);
	copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
	return padded;
}

void gaussian_bandpass_filter(Mat& dftFilter, double d0, double w)
{
	Mat tmp = Mat(dftFilter.rows, dftFilter.cols, CV_32F);
	Point center = Point(dftFilter.rows / 2, dftFilter.cols / 2);
	double d1 = d0 * d0;
	for (int i = 0; i < dftFilter.rows; i++)
	{
		for (int j = 0; j < dftFilter.cols; j++)
		{
			double d = pow((double)(i - center.x), 2) + pow((double)(j - center.y), 2);
			tmp.at<float>(i, j) = (float)expf((-1.0) * pow(((d - d1) / (sqrt(d) * w)), 2));
		}
	}

	Mat toMerge[] = { tmp, tmp };
	merge(toMerge, 2, dftFilter);
}

void gaussian_bandstop_filter(Mat& dftFilter, double d0, double w)
{
	Mat tmp = Mat(dftFilter.rows, dftFilter.cols, CV_32F);
	Point center = Point(dftFilter.rows / 2, dftFilter.cols / 2);
	double d1 = d0 * d0;
	for (int i = 0; i < dftFilter.rows; i++)
	{
		for (int j = 0; j < dftFilter.cols; j++)
		{
			double d = pow((double)(i - center.x), 2) + pow((double)(j - center.y), 2);
			tmp.at<float>(i, j) = (float)(1.0 - expf((-1.0) * pow(((d - d1) / (sqrt(d) * w)), 2)));
		}
	}

	Mat toMerge[] = { tmp, tmp };
	merge(toMerge, 2, dftFilter);
}

Mat apply_filter(Mat img, Mat filter)
{
	Mat padded = create_padded_image(img);

	// setup the DFT images
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);

	// do the DFT
	dft(complexI, complexI);

	// apply filter
	shiftDFT(complexI);
	mulSpectrums(complexI, filter, complexI, 0);
	shiftDFT(complexI);

	// do inverse DFT on filtered image
	idft(complexI, complexI);

	// split into planes and extract plane 0 as output image
	split(complexI, planes);
	Mat magI = planes[0];
	normalize(magI, magI, 0, 1, NORM_MINMAX);

	return magI;
}

void apply_bandpass_filter(int d, int w) {
	Mat gauss_bandpass = src.clone();
	gaussian_bandpass_filter(gauss_bandpass, d, w);
	Mat dst_gauss_bandpass = apply_filter(src, gauss_bandpass);
	imshow(trackBarNameBandpass, dst_gauss_bandpass);
}

void apply_bandstop_filter(int d, int w) {
	Mat gauss_bandstop = src.clone();
	gaussian_bandstop_filter(gauss_bandstop, d, w);
	Mat dst_gauss_bandstop = apply_filter(src, gauss_bandstop);
	imshow(trackBarNameBandstop, dst_gauss_bandstop);
}

void onChangeBandpass(int sliderValue, void* userData) {
	int dValue = getTrackbarPos("TanSoCat", trackBarNameBandpass);
	int wValue = getTrackbarPos("DoRongDai", trackBarNameBandpass);
	
	apply_bandpass_filter(dValue, wValue);
}

void onChangeBandstop(int sliderValue, void* userData) {
	int dValue = getTrackbarPos("TanSoCat", trackBarNameBandstop);
	int wValue = getTrackbarPos("DoRongDai", trackBarNameBandstop);

	apply_bandstop_filter(dValue, wValue);
}

int main(int argc, char** argv)
{
	src = imread("Lenna.png", IMREAD_GRAYSCALE);
	imshow("Input image", src);

	namedWindow(trackBarNameBandpass, 1);
	createTrackbar("TanSoCat", trackBarNameBandpass, &D, 50, onChangeBandpass);
	createTrackbar("DoRongDai", trackBarNameBandpass, &W, 50, onChangeBandpass);
	apply_bandpass_filter(D, W);

	namedWindow(trackBarNameBandstop, 1);
	createTrackbar("TanSoCat", trackBarNameBandstop, &D, 50, onChangeBandstop);
	createTrackbar("DoRongDai", trackBarNameBandstop, &W, 50, onChangeBandstop);
	apply_bandstop_filter(D, W);

	waitKey();
}