
#include "pch.h"
#include <conio.h>
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\core\core.hpp"
#include <iostream>
#include "opencv2/imgproc.hpp"
#include <fstream>
#include <cmath>

using namespace std;
using namespace cv;

// features of RGB image -----------------------------------------

// counting the amount of pixel of a specific colour in an image
void countColourSplit(Mat RGBImg, int degree, int colour[]) { // degree is number of division of 255
	for (int row = 0; row < RGBImg.rows; row++) {
		for (int col = 0; col < RGBImg.cols * 3; col = col + 3) {
			int index = 0;
			for (int RGBIndex = 0; RGBIndex < 3; RGBIndex++) {
				for (int separation = 1; separation <= degree; separation++) {
					if (RGBImg.at<uchar>(row, col + RGBIndex) <= separation * (255.00 / degree)) {
						if (RGBIndex == 0) {
							index = index + (separation - 1);
							break;
						}
						else if (RGBIndex == 1) {
							index = index + degree * (separation - 1);
							break;
						}
						else if (RGBIndex == 2) {
							index = index + pow(degree, RGBIndex)*(separation - 1);
							break;
						}
					}
				}
			}
			colour[index] = colour[index] + 1;
		}
	}
}

// finding the primary colour in the image
int primaryColour(int colour[]) {
	int maxValue = 0;
	int maxIndex = 0;
	for (int i = 0; i < 27; i++) {
		if (maxValue < colour[i]) {
			maxValue = colour[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

// finding the number of primary colour in the image
int numberOfPrimaryColour(int colour[]) {
	int max = primaryColour(colour);
	int counter = 0;
	for (int i = 0; i < 27; i++) {
		if (colour[i] > colour[max] * 0.83) { // threshold is at 83% as it is the lowest without plate having 2 primary colour
			counter++;
		}
	}
	return counter;
}

// finding the number of colour in the image
int numberOfColour(int colour[]) {
	int counter = 0;
	for (int i = 0; i < 27; i++) {
		if (colour[i] > 0) {
			counter++;
		}
	}
	return counter;
}

// converting to grey image -----------------------------------------
Mat convertToGrey(Mat RGBImg) {
	Mat greyedImg = Mat::zeros(RGBImg.size(), CV_8UC1);
	for (int r = 0; r < RGBImg.rows; r++) {
		for (int c = 0; c < RGBImg.cols * 3; c = c + 3) {
			greyedImg.at<uchar>(r, c / 3) = (RGBImg.at<uchar>(r, c) + RGBImg.at<uchar>(r, c + 1) + RGBImg.at<uchar>(r, c + 2)) / 3;
		}
	}
	return greyedImg;
}

// processing grey image -----------------------------------------

// OTSU value for grey image
int findOTSU(Mat greyImg) {
	int const level = 256;
	int count[level] = { 0 };
	float prob[level] = { 0 };
	float accprob[level] = { 0 };
	int row = greyImg.rows;
	int col = greyImg.cols;
	int totalCount = row * col;
	// counting number of same pixel
	for (int r = 0; r < row; r++) {
		for (int c = 0; c < col; c++) {
			for (int i = 0; i < level; i++) { // scroll through to find respective level to add count
				if (i == greyImg.at<uchar>(r, c)) {
					count[i] = count[i] + 1;
				}
			}
		}
	}
	// probability
	for (int i = 0; i < level; i++) {
		prob[i] = (float)count[i] / (float)totalCount;
	}
	// accumulative probability
	for (int i = 0; i < level; i++) {
		for (int j = 0; j < i + 1; j++) {
			accprob[i] = accprob[i] + prob[j];
		}
	}
	// OTSU
	float OTSU[level] = { 0 };
	float mew[level] = { 0 };
	// Mew Calculation
	for (int i = 1; i < level; i++) {
		mew[i] = mew[i - 1] + (float)i*prob[i];
	}
	// OTSU Calculation
	for (int i = 0; i < level; i++) {
		OTSU[i] = pow(mew[255] * accprob[i] - mew[i], 2) / (accprob[i] * (1 - accprob[i]));
	}
	int maximise = 0;
	int max = -1;
	// MAX Value
	for (int i = 0; i < level; i++) {
		if (maximise < OTSU[i]) {
			maximise = (int)OTSU[i];
			max = i;
		}
	}
	return max + 30;
}

// EH-ing grey image
Mat EHGreyImg(Mat greyImg) {
	Mat EHImg = Mat::zeros(greyImg.size(), CV_8UC1);
	int const level = 256;
	int count[level] = { 0 };
	float prob[level] = { 0 };
	float accprob[level] = { 0 };
	int newPixel[level] = { 0 };
	int row = greyImg.rows;
	int col = greyImg.cols;
	int totalCount = row * col;
	// counting number of same pixel
	for (int r = 0; r < row; r++) {
		for (int c = 0; c < col; c++) {
			for (int i = 0; i < level; i++) { // scroll through to find respective level to add count
				if (i == greyImg.at<uchar>(r, c)) {
					count[i] = count[i] + 1;
				}
			}
		}
	}
	// probability
	for (int i = 0; i < level; i++) {
		prob[i] = (float)count[i] / (float)totalCount;
	}
	// accumulative probability
	for (int i = 0; i < level; i++) {
		for (int j = 0; j < i + 1; j++) {
			accprob[i] = accprob[i] + prob[j];
		}
	}
	// new pixel level
	for (int i = 0; i < level; i++) {
		newPixel[i] = (int)((level - 1)*accprob[i]);
	}
	// EH image
	for (int r = 0; r < row; r++) {
		for (int c = 0; c < col; c++) {
			for (int k = 0; k < level; k++) {
				EHImg.at<uchar>(r, c) = newPixel[greyImg.at<uchar>(r, c)];
			}
		}
	}
	return EHImg;
}

// blurring image using average mask
Mat blurringImg(Mat greyImg) {
	Mat blurredImg = Mat::zeros(greyImg.size(), CV_8UC1);
	for (int r = 1; r < greyImg.rows; r++) {
		for (int c = 1; c < greyImg.cols; c++) {
			float total = 0.00;
			for (int rr = -1; rr < 2; rr++) {
				for (int cc = -1; cc < 2; cc++) {
					if (r + rr < 0) { // check for out of bounds
						break;
					}
					else if (c + cc < 0) {
						break;
					}
					else {
						total = total + greyImg.at<uchar>(r + rr, c + cc);
					}
				}
			}
			blurredImg.at<uchar>(r, c) = (int)(total / 9);
		}
	}
	return blurredImg;
}

// features of grey image

// pixel spread of grey image
double greySD(Mat greyImg) {
	int row = greyImg.rows;
	int col = greyImg.cols;
	int size = row * col;
	double total = 0;
	for (int r = 0; r < row; r++) {
		for (int c = 0; c < col; c++) {
			total = total + greyImg.at<uchar>(r, c);
		}
	}
	double mean = total / size;
	total = 0;
	for (int r = 0; r < row; r++) {
		for (int c = 0; c < col; c++) {
			total = total + pow(greyImg.at<uchar>(r, c) - mean, 2);
		}
	}
	double sd = total / size;
	return sqrt(sd);
}

// converting to binary image -----------------------------------------

// finding the edge using threshold
Mat greyToBinary(Mat greyImg, int threshold) {
	Mat binariedImg = Mat::zeros(greyImg.size(), CV_8UC1);
	int row = greyImg.rows;
	int col = greyImg.cols;
	for (int r = 0; r < greyImg.rows; r++) {
		for (int c = 0; c < greyImg.cols; c++) {
			if (greyImg.at<uchar>(r, c) > threshold) {
				binariedImg.at<uchar>(r, c) = 255;
			}
			else { // not required as image is set to 0
				binariedImg.at<uchar>(r, c) = 0;
			}
		}
	}
	return binariedImg;
}

// finding the edge
Mat edgingImg(Mat greyImg) {
	Mat edgedImg = Mat::zeros(greyImg.size(), CV_8UC1);
	for (int r = 1; r < greyImg.rows - 1; r++) {
		for (int c = 1; c < greyImg.cols - 1; c++) {
			int leftAVG = (greyImg.at<uchar>(r - 1, c - 1) + greyImg.at<uchar>(r, c - 1) + greyImg.at<uchar>(r + 1, c - 1)) / 3;
			int rightAVG = (greyImg.at<uchar>(r - 1, c + 1) + greyImg.at<uchar>(r, c + 1) + greyImg.at<uchar>(r + 1, c + 1)) / 3;
			if (abs(leftAVG - rightAVG) > 20) {
				edgedImg.at<uchar>(r, c) = 255;
			}
			else { // not required as image is set at 0
				edgedImg.at<uchar>(r, c) = 0;
			}
		}
	}
	return edgedImg;
}

// processing binary image -----------------------------------------

// horizontal dilation
Mat horizontalDilatingImg(Mat binarisedImg) {
	Mat dilatedImg = Mat::zeros(binarisedImg.size(), CV_8UC1);
	for (int r = 0; r < binarisedImg.rows; r++) {
		for (int c = 0; c < binarisedImg.cols; c++) {
			if (binarisedImg.at<uchar>(r, c) == 0) {
				if (c - 1 > 0 || c + 1 < binarisedImg.cols) {
					if (binarisedImg.at<uchar>(r, c - 1) == 255 || binarisedImg.at<uchar>(r, c + 1) == 255) {
						dilatedImg.at<uchar>(r, c) = 255;
					}
				}
			}
			else {
				dilatedImg.at<uchar>(r, c) = 255;
			}
		}
	}
	return dilatedImg;
}

// vertical dilation
Mat verticalDilatingImg(Mat binarisedImg) {
	Mat dilatedImg = Mat::zeros(binarisedImg.size(), CV_8UC1);
	for (int r = 0; r < binarisedImg.rows; r++) {
		for (int c = 0; c < binarisedImg.cols; c++) {
			if (binarisedImg.at<uchar>(r, c) == 0) {
				if (r - 1 > 0 || r + 1 < binarisedImg.rows) {
					if (binarisedImg.at<uchar>(r - 1, c) == 255 || binarisedImg.at<uchar>(r + 1, c) == 255) {
						dilatedImg.at<uchar>(r, c) = 255;
					}
				}
			}
			else {
				dilatedImg.at<uchar>(r, c) = 255;
			}
		}
	}
	return dilatedImg;
}

// features of binarised image -----------------------------------------

// counting the number of white pixels
int numberWhite(Mat binarisedImg) {
	int row = binarisedImg.rows;
	int col = binarisedImg.cols;
	int total = 0;
	// counting number of same pixel
	for (int r = 0; r < row; r++) {
		for (int c = 0; c < col; c++) {
			if (binarisedImg.at<uchar>(r, c) == 255) {
				total++;
			}
		}
	}
	return total;
}

// finding the percentage of white pixels
double numberPercentageWhite(Mat binarisedImg) {
	int row = binarisedImg.rows;
	int col = binarisedImg.cols;
	double total = 0.0;
	// counting number of same pixel
	for (int r = 0; r < row; r++) {
		for (int c = 0; c < col; c++) {
			if (binarisedImg.at<uchar>(r, c) == 255) {
				total++;
			}
		}
	}
	return total / (row*col) * 100;
}

// finding the largest blobs in the image
float biggestBlobsDensity(vector<vector<Point> > blobs) {
	int maximise = 0;
	int max = 0;
	Rect blobRect;
	float density;
	for (int i = 0; i < blobs.size(); i++) {
		blobRect = boundingRect(blobs[i]);
		density = ((float)blobs[i].size()) / ((float)blobRect.width*(float)blobRect.height);
		if (maximise < density) {
			maximise = density;
			max = i;
		}
	}
	return density;
}

int main()
{
	string type = "all";
	int size = 35;
	if (type == "noise") {
		size = 149;
	}
	if (type == "all")
	{
		size = 183;
	}
	string pathname = "C:\\Users\\chigo\\Desktop\\";
	int NoiseCounter = 0;
	int PlateCounter = 0;
	for (int index = 1; index < size; index++) {
		string path = pathname + type + "\\" + type + " (" + to_string(index) + ").bmp";

		// colour separation
		int degree = 3; // up to 10 only or colour have to increase size
		
		// RGB
		Mat InputImage = imread(path); //
		int colour[1000] = { 0 };
		countColourSplit(InputImage, degree, colour);

		// GREY
		Mat greyImg = convertToGrey(InputImage); //
		Mat EHImage = EHGreyImg(greyImg);
		Mat blurEHImg = blurringImg(EHImage); //

		// BINARY
		Mat edgeBlurEHImg = edgingImg(blurEHImg); //
		Mat dilationEdgeBlurEHImg = horizontalDilatingImg(edgeBlurEHImg);
		dilationEdgeBlurEHImg = horizontalDilatingImg(edgeBlurEHImg);
		dilationEdgeBlurEHImg = horizontalDilatingImg(edgeBlurEHImg);
		dilationEdgeBlurEHImg = verticalDilatingImg(edgeBlurEHImg); //
		Mat otsuGreyImg = greyToBinary(greyImg, findOTSU(greyImg)); //
		Mat otsuBlurEHImg = greyToBinary(blurEHImg, findOTSU(blurEHImg)); //

		// BLOBS
		vector<vector<Point> > edgeBlurEHBlobs;
		vector<Vec4i> hierachy1;
		findContours(edgeBlurEHImg, edgeBlurEHBlobs, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

		//imshow("Image", InputImage);
		//cvWaitKey();
		// NOISE -----------------------------------------
		
		// biggestBlobDensity - usually plates are not connected if it is, it would be a standard size
		if (biggestBlobsDensity(edgeBlurEHBlobs) < 0.151304) { // 14 noise detected
			cout << "Detected\tnoise\tbiggest blob density" << endl;
			NoiseCounter++;
			continue;
		}

		// RGB Analysis
		
		// number of colour out of 27 - plates usually contain less variety of colour
		if (numberOfColour(colour) < 7 || numberOfColour(colour) > 15) { // 15 noise detected
			cout << "Detected\tnoise\tnumber of colour" << endl;
			NoiseCounter++;
			continue;
		}

		// number of dominant colour - plates usually have one or two dominate colour
		if (numberOfPrimaryColour(colour) != 1) { // 19 noise detected
			cout << "Detected\tnoise\tnumber of primary colour" << endl;
			NoiseCounter++;
			continue;
		}
		
		// GREY ANALYSIS
		// grey spread - noise might have high or low spread of grey depending on the situation
		if (greySD(greyImg) < 43.0395) { // 68 noise detected
			cout << "Detected\tnoise\tspread of grey image" << endl;
			NoiseCounter++;
			continue;
		}
		
		// BINARY ANALYSIS
		// number of white - looking at foreground versus background/edge see the amount of white colour present
		if (numberWhite(edgeBlurEHImg) < 1550 || // 112 noise detected
			numberWhite(edgeBlurEHImg) > 5077) {
			cout << "Detected\tnoise\tnumber of white pixels" << endl;
			NoiseCounter++;
			continue;
		}

		
		if (numberWhite(otsuGreyImg) < 922) { // 114 noise detected
			cout << "Detected\tnoise\tnumber of white pixels through OTSU" << endl;
			NoiseCounter++;
			continue;
		}
		
		
		// PLATE -----------------------------------------
		if (numberPercentageWhite(otsuBlurEHImg) > 45.46) { // 1 plates
			cout << "Detected\tplate\tpercentage of white through OSTU with grey IP" << endl;
			PlateCounter++;
			continue;
		}

		if (numberPercentageWhite(otsuGreyImg) > 54.00) { // 2 plates
			cout << "Detected\tplate\tpercentage of white through OSTU" << endl;
			PlateCounter++;
			continue;
		}
		
		if (numberWhite(otsuGreyImg) > 12211) { // 2 plates
			cout << "Detected\tplate\tnumber of white pixel" << endl;
			PlateCounter++;
			continue;
		}
		
		if (greySD(greyImg) > 73.2474) { // 5 plates
			cout << "Detected\tplate\tspread of grey image" << endl;
			PlateCounter++;
			continue;
		}
		
		cout << "Predicted\tplate" << endl;
		PlateCounter++;
		
	}
	cout << NoiseCounter << "/148 noise" << endl;
	cout << PlateCounter << "/34 plate" << endl;
	_getch();
	return 0;
}
