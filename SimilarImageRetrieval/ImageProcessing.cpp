#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>
#include "Header.h"

using namespace cv;
using namespace std;

void displayQueryImage(Mat &image) {
	cout << "Image Height:\t" << image.rows << endl;
	cout << "Image Width :\t" << image.cols << endl;

	namedWindow("Query Image", WINDOW_AUTOSIZE);
	imshow("Query Image", image);
}

int calcColourBin(Vec3b colour) {
	int colourBin; //colour range from 0 - 511
	int b, g, r; //bgr range from 0 - 7

	b = colour[0] / 32;
	g = colour[1] / 32;
	r = colour[2] / 32;
	colourBin = b * pow(8, 2) + g * 8 + r;

	return colourBin;
}

Vec3b convert2Bgr(int colourBin) {
	int b, g, r;

	b = colourBin / 64 * 32 + 16; //16 is center intensity value of colour bin
	g = (colourBin / 8) % 8 * 32 + 16;
	r = colourBin % 8 * 32 + 16;

	return Vec3b(b, g, r);
}

void roi(Mat &image, Mat &roi_left, Mat &roi_right) {
	Rect left(0, 0, image.cols / 2, image.rows); //symmetry left part of the image
	Rect right(left.width, 0, image.cols - left.width, image.rows); //symmetry right part of the image

	roi_left = image(left);
	roi_right = image(right);

	//display images
	/*namedWindow("Left Part", WINDOW_AUTOSIZE);
	imshow("Left Part", roi_left);
	imwrite("Left Part.jpg", roi_left);
	namedWindow("Right Part", WINDOW_AUTOSIZE);
	imshow("Right Part", roi_right);
	imwrite("Right Part.jpg", roi_right);*/
}

int sortDominantColour(vector<int> &colourVote, vector<pair<int, int>> &bin_sort, int &max_fre) {
	int totalColour = 0;

	for (int i = 0; i < 512; i++) {
		if (colourVote[i] > 0) {
			bin_sort.push_back(pair<int, int>(colourVote[i], i));
			max_fre = max(max_fre, colourVote[i]);
			totalColour++;
		}
	}

	sort(bin_sort.rbegin(), bin_sort.rend()); //sort the quantized bin vector according to its vote

	return totalColour;
}

int wholeImage(Mat &image, Mat &resultImage, int *topRankColour, int *pixelPercentage, Mat &histogram) {
	//colour quantization
	Mat quanImage(image.size(), image.type());
	vector<int> colourVote(512);

	//for each pixel, compute the 8 bins value and store into vector 
	for (int r = 0; r < image.rows; r++){
		for (int c = 0; c < image.cols; c++){
			int colourValue = calcColourBin(image.at<Vec3b>(r, c)); //quantized colour bin

			colourVote[colourValue]++; //calculate vote of each colour

			//draw quantized image
			for (int channel = 0; channel < 3; channel++)
				quanImage.at<Vec3b>(r, c)[channel] = convert2Bgr(colourValue)[channel];
		}
	}

	/*namedWindow("Quantized Image", WINDOW_AUTOSIZE);
	imshow("Quantized Image", quanImage);
	imwrite("Quantized Image.jpg", quanImage);*/

	//sort quantized dominant colours
	vector<pair<int, int>> bin_sort; //a vector to store the quantized bin and sort it
	int max_fre = 0;
	int totalColour; //to store the total number of contained colours

	totalColour = sortDominantColour(colourVote, bin_sort, max_fre);

	//draw the histogram in canvas
	Mat canvas(620, 1000, CV_8UC3, Scalar(255, 255, 255));
	for (int i = 0; i < totalColour; i++)
		rectangle(canvas, Point((30 + 10) * i + 10, 620), Point((30 + 10) * i + 40, 620 - bin_sort[i].first * 600 / max_fre), convert2Bgr(bin_sort[i].second), -1);

	/*namedWindow("Sorted Quantized Dominant Colours Histogram", WINDOW_AUTOSIZE);
	imshow("Sorted Quantized Dominant Colours Histogram", canvas);
	imwrite("Sorted Quantized Dominant Colours Histogram.jpg", canvas);*/

	//merge similar colour pixel 
	int threshold = 32, colourIndex = 0;
	Mat mergedColourImage(image.size(), image.type());

	//convert similar colour into same colour within bin_sort vector according to colour rank

	while (colourIndex < totalColour) {
		for (int cnt = 0; cnt < totalColour; cnt++) {
			if (abs(convert2Bgr(bin_sort[colourIndex].second)[0] - convert2Bgr(bin_sort[cnt].second)[0]) <= threshold * 3 &&
				abs(convert2Bgr(bin_sort[colourIndex].second)[1] - convert2Bgr(bin_sort[cnt].second)[1]) <= threshold * 3 &&
				abs(convert2Bgr(bin_sort[colourIndex].second)[2] - convert2Bgr(bin_sort[cnt].second)[2]) <= threshold * 3 &&
				(abs(convert2Bgr(bin_sort[colourIndex].second)[0] - convert2Bgr(bin_sort[cnt].second)[0]) +
				abs(convert2Bgr(bin_sort[colourIndex].second)[1] - convert2Bgr(bin_sort[cnt].second)[1]) +
				abs(convert2Bgr(bin_sort[colourIndex].second)[2] - convert2Bgr(bin_sort[cnt].second)[2])) <= threshold * 6)
				bin_sort[cnt].second = bin_sort[colourIndex].second;
		}
		colourIndex++;
	}

	colourIndex = 0; //reset flag

	//draw merged colour image
	while (colourIndex < totalColour) {
		for (int r = 0; r < image.rows; r++) {
			for (int c = 0; c < image.cols; c++) {
				if (abs(convert2Bgr(bin_sort[colourIndex].second)[0] - quanImage.at<Vec3b>(r, c)[0]) <= threshold * 3 &&
					abs(convert2Bgr(bin_sort[colourIndex].second)[1] - quanImage.at<Vec3b>(r, c)[1]) <= threshold * 3 &&
					abs(convert2Bgr(bin_sort[colourIndex].second)[2] - quanImage.at<Vec3b>(r, c)[2]) <= threshold * 3 &&
					(abs(convert2Bgr(bin_sort[colourIndex].second)[0] - quanImage.at<Vec3b>(r, c)[0]) +
					abs(convert2Bgr(bin_sort[colourIndex].second)[1] - quanImage.at<Vec3b>(r, c)[1]) +
					abs(convert2Bgr(bin_sort[colourIndex].second)[2] - quanImage.at<Vec3b>(r, c)[2])) <= threshold * 6) {
					for (int channel = 0; channel < 3; channel++)
						mergedColourImage.at<Vec3b>(r, c)[channel] = convert2Bgr(bin_sort[colourIndex].second)[channel];
				}
			}
		}

		colourIndex++;
	}

	/*namedWindow("Merged Colour Image", WINDOW_AUTOSIZE);
	imshow("Merged Colour Image", mergedColourImage);
	imwrite("Merged Colour Image.jpg", mergedColourImage);*/

	resultImage = mergedColourImage;

	vector<int> merge_colourVote(512);

	//for each pixel, compute the 8 bins value and store into vector 
	for (int r = 0; r < mergedColourImage.rows; r++){
		for (int c = 0; c < mergedColourImage.cols; c++){
			int colourValue = calcColourBin(mergedColourImage.at<Vec3b>(r, c));

			merge_colourVote[colourValue]++; //calculate vote of each colour
		}
	}

	vector<pair<int, int>> merge_bin_sort; //a vector to store the quantized bin and sort it
	int merge_max_fre = 0;
	int merge_totalColour; //to store the total number of contained colours

	merge_totalColour = sortDominantColour(merge_colourVote, merge_bin_sort, merge_max_fre);

	//draw the histogram in merge_canvas
	Mat merge_canvas(620, 1000, CV_8UC3, Scalar(255, 255, 255));
	for (int i = 0; i < merge_totalColour; i++)
		rectangle(merge_canvas, Point((30 + 10) * i + 10, 620), Point((30 + 10) * i + 40, 620 - merge_bin_sort[i].first * 600 / merge_max_fre),
		convert2Bgr(merge_bin_sort[i].second), -1);

	/*namedWindow("Dominant Colours Histogram After Merging", WINDOW_AUTOSIZE);
	imshow("Dominant Colours Histogram After Merging", merge_canvas);
	imwrite("Dominant Colours Histogram After Merging.jpg", merge_canvas);*/

	histogram = merge_canvas;

	//store the dominant colours
	int domColour_no = 1; //number of dominant colour after neglecting noices

	*topRankColour = merge_bin_sort[0].second;
	*pixelPercentage = merge_bin_sort[0].first;

	for (int i = 1; i < merge_totalColour; i++) {
		if (merge_bin_sort[i].first > merge_bin_sort[i - 1].first * 0.1) {
			*(topRankColour + i) = merge_bin_sort[i].second;
			*(pixelPercentage + i) = merge_bin_sort[i].first;
			domColour_no++;
		}
		else
			break;
	}

	return domColour_no;
}

void centralNormalizedMoment(Mat &binaryImage, double *nuMoment) {
	Moments moment_output;
	moment_output = moments(binaryImage, true); //white foreground region
	nuMoment[0] = moment_output.nu02;
	nuMoment[1] = moment_output.nu03;
	nuMoment[2] = moment_output.nu11;
	nuMoment[3] = moment_output.nu12;
	nuMoment[4] = moment_output.nu20;
	nuMoment[5] = moment_output.nu21;
	nuMoment[6] = moment_output.nu30;
}

double euclideanDistance_colourRegion(Mat &image, Mat &trainImage, int *colour_query, int *colour_dataset, int totalColour_query, int totalColour_dataset) {
	vector<Mat> singleColour_query, singleColour_dataset; //binary images of each colour
	ostringstream oss;
	string binaryWindowName;

	for (int n = 0; n < totalColour_query; n++) {
		Mat imBw = Mat::zeros(image.size(), CV_8UC1);

		for (int r = 0; r < image.rows; r++) {
			for (int c = 0; c < image.cols; c++) {
				if (calcColourBin(image.at<Vec3b>(r, c)) == *(colour_query + n))
					imBw.at<uchar>(r, c) = uchar(255);
			}
		}

		singleColour_query.push_back(imBw);

		//display binary images
		//oss << "Query Binary Image " << n + 1;
		//binaryWindowName = oss.str();
		//namedWindow(binaryWindowName, WINDOW_AUTOSIZE);
		//imshow(binaryWindowName, imBw);
		//imwrite(binaryWindowName + ".jpg", imBw);
		//oss.str(""); //clear the string
		//oss.clear(); //clear any error flags
	}
	
	for (int n = 0; n < totalColour_dataset; n++) {
		Mat imBw = Mat::zeros(trainImage.size(), CV_8UC1);

		for (int r = 0; r < trainImage.rows; r++) {
			for (int c = 0; c < trainImage.cols; c++) {
				if (calcColourBin(trainImage.at<Vec3b>(r, c)) == *(colour_dataset + n))
					imBw.at<uchar>(r, c) = uchar(255);
			}
		}

		singleColour_dataset.push_back(imBw);

		//display binary images
		//oss << "Dataset Binary Image " << n + 1;
		//binaryWindowName = oss.str();
		//namedWindow(binaryWindowName, WINDOW_AUTOSIZE);
		//imshow(binaryWindowName, imBw);
		//imwrite(binaryWindowName + ".jpg", imBw);
		//oss.str(""); //clear the string
		//oss.clear(); //clear any error flags
	}

	double nuMoment_query[7], nuMoment_dataset[7];
	double total = 0.0, ed;

	for (int i = 0; i < totalColour_query; i++) { //check query
		int notMatched = 0;

		for (int j = 0; j < totalColour_dataset; j++) {
			if (abs(convert2Bgr(*(colour_query + i))[0] / 32 - convert2Bgr(*(colour_dataset + j))[0] / 32) <= 2 &&
				abs(convert2Bgr(*(colour_query + i))[1] / 32 - convert2Bgr(*(colour_dataset + j))[1] / 32) <= 2 &&
				abs(convert2Bgr(*(colour_query + i))[2] / 32 - convert2Bgr(*(colour_dataset + j))[2] / 32) <= 2 &&
				(abs(convert2Bgr(*(colour_query + i))[0] / 32 - convert2Bgr(*(colour_dataset + j))[0] / 32) +
				abs(convert2Bgr(*(colour_query + i))[1] / 32 - convert2Bgr(*(colour_dataset + j))[1] / 32) +
				abs(convert2Bgr(*(colour_query + i))[2] / 32 - convert2Bgr(*(colour_dataset + j))[2] / 32)) <= 3) {
				centralNormalizedMoment(singleColour_query[i], nuMoment_query);
				centralNormalizedMoment(singleColour_dataset[j], nuMoment_dataset);

				for (int k = 0; k < 7; k++)
					total += pow(*(nuMoment_query + k) - *(nuMoment_dataset + k), 2);

				continue;
			}
			else
				notMatched++;

			if (notMatched == totalColour_dataset) { //if no similar colour matched then minus 0
				for (int k = 0; k < 7; k++)
					total += pow(*(nuMoment_query + k) - 0, 2);
			}
		}
	}

	for (int i = 0; i < totalColour_dataset; i++) { //check dataset
		int notMatched = 0;

		for (int j = 0; j < totalColour_query; j++) {
			if (abs(convert2Bgr(*(colour_dataset + i))[0] / 32 - convert2Bgr(*(colour_query + j))[0] / 32) > 2 ||
				abs(convert2Bgr(*(colour_dataset + i))[1] / 32 - convert2Bgr(*(colour_query + j))[1] / 32) > 2 ||
				abs(convert2Bgr(*(colour_dataset + i))[2] / 32 - convert2Bgr(*(colour_query + j))[2] / 32) > 2 ||
				(abs(convert2Bgr(*(colour_dataset + i))[0] / 32 - convert2Bgr(*(colour_query + j))[0] / 32) +
				abs(convert2Bgr(*(colour_dataset + i))[1] / 32 - convert2Bgr(*(colour_query + j))[1] / 32) +
				abs(convert2Bgr(*(colour_dataset + i))[2] / 32 - convert2Bgr(*(colour_query + j))[2] / 32)) > 3)
				notMatched++;

			if (notMatched == totalColour_query) { //if no similar colour matched then minus 0
				for (int k = 0; k < 7; k++)
					total += pow(*(nuMoment_dataset + k) - 0, 2);
			}
		}
	}

	ed = sqrt(total);

	return ed;
}

void findContour(Mat &image, Mat &resultImage, double *nuMoment) {
	Mat imGray, imBw;

	cvtColor(image, imGray, CV_BGR2GRAY);
	threshold(imGray, imBw, 220, 255, THRESH_BINARY_INV); //set values equal to or above 220 to 0, values below 220 to 255

	// Floodfill from point (0, 0)
	Mat im_floodfill = imBw.clone();
	floodFill(im_floodfill, Point(0, 0), Scalar(255));

	// Invert floodfilled image
	Mat im_floodfill_inv;
	bitwise_not(im_floodfill, im_floodfill_inv);

	// Combine the two images to get the foreground.
	Mat imOut = (imBw | im_floodfill_inv);

	// Display images
	/*namedWindow("Step 1", WINDOW_AUTOSIZE);
	imshow("Step 1", imGray);
	imwrite("Step 2.jpg", imGray);
	namedWindow("Step 2", WINDOW_AUTOSIZE);
	imshow("Step 2", imBw);
	imwrite("Step 2.jpg", imBw);
	namedWindow("Step 3", WINDOW_AUTOSIZE);
	imshow("Step 3", im_floodfill);
	imwrite("Step 3.jpg", im_floodfill);
	namedWindow("Step 4", WINDOW_AUTOSIZE);
	imshow("Step 4", im_floodfill_inv);
	imwrite("Step 4.jpg", im_floodfill_inv);
	namedWindow("Foreground", WINDOW_AUTOSIZE);
	imshow("Foreground", imOut);
	imwrite("Foreground.jpg", imOut);*/

	resultImage = imOut;

	centralNormalizedMoment(imOut, nuMoment); //calculate the 7 central normalized moments
}

int eliminateBackground(Mat &image, Mat &processedImage, int *topRankColour, int *pixelPercentage, Mat &histogram) {
	vector<int> colourVote(512);

	for (int r = 0; r < image.rows; r++) {
		for (int c = 0; c < image.cols; c++) {
			if (processedImage.at<uchar>(r, c) == 255) {
				int colourValue = calcColourBin(image.at<Vec3b>(r, c)); //quantized colour bin

				colourVote[colourValue]++; //calculate vote of each colour
			}
		}
	}

	//sort quantized dominant colours
	vector<pair<int, int>> bin_sort; //a vector to store the quantized bin and sort it
	int max_fre = 0;
	int totalColour; //to store the total number of contained colours

	totalColour = sortDominantColour(colourVote, bin_sort, max_fre);

	//draw the histogram in canvas
	Mat canvas(620, 1000, CV_8UC3, Scalar(255, 255, 255));
	for (int i = 0; i < totalColour; i++)
		rectangle(canvas, Point((30 + 10) * i + 10, 620), Point((30 + 10) * i + 40, 620 - bin_sort[i].first * 600 / max_fre), convert2Bgr(bin_sort[i].second), -1);

	/*namedWindow("Sorted Quantized Dominant Colours Histogram", WINDOW_AUTOSIZE);
	imshow("Sorted Quantized Dominant Colours Histogram", canvas);
	imwrite("Sorted Quantized Dominant Colours Histogram.jpg", canvas);*/

	//merge similar colour pixel 
	int threshold = 32, colourIndex = 0;

	//convert similar colour into same colour within bin_sort vector according to colour rank
	while (colourIndex < totalColour) {
		for (int cnt = 0; cnt < totalColour; cnt++) {
			if (abs(convert2Bgr(bin_sort[colourIndex].second)[0] - convert2Bgr(bin_sort[cnt].second)[0]) <= threshold * 3 &&
				abs(convert2Bgr(bin_sort[colourIndex].second)[1] - convert2Bgr(bin_sort[cnt].second)[1]) <= threshold * 3 &&
				abs(convert2Bgr(bin_sort[colourIndex].second)[2] - convert2Bgr(bin_sort[cnt].second)[2]) <= threshold * 3 &&
				(abs(convert2Bgr(bin_sort[colourIndex].second)[0] - convert2Bgr(bin_sort[cnt].second)[0]) +
				abs(convert2Bgr(bin_sort[colourIndex].second)[1] - convert2Bgr(bin_sort[cnt].second)[1]) +
				abs(convert2Bgr(bin_sort[colourIndex].second)[2] - convert2Bgr(bin_sort[cnt].second)[2])) <= threshold * 6)
				bin_sort[cnt].second = bin_sort[colourIndex].second;
		}
		colourIndex++;
	}

	vector<int> merge_colourVote(512);
	vector<pair<int, int>> merge_bin_sort;
	int merge_max_fre = 0;
	int merge_totalColour;

	for (int i = 0; i < totalColour; i++){
		int colourValue = calcColourBin(convert2Bgr(bin_sort[i].second));

		merge_colourVote[colourValue]++;
	}

	merge_totalColour = sortDominantColour(merge_colourVote, merge_bin_sort, merge_max_fre);

	//draw the histogram in merge_canvas
	Mat merge_canvas(620, 1000, CV_8UC3, Scalar(255, 255, 255));
	for (int i = 0; i < merge_totalColour; i++)
		rectangle(merge_canvas, Point((30 + 10) * i + 10, 620), Point((30 + 10) * i + 40, 620 - merge_bin_sort[i].first * 600 / merge_max_fre),
		convert2Bgr(merge_bin_sort[i].second), -1);

	/*namedWindow("Dominant Colours Histogram After Merging", WINDOW_AUTOSIZE);
	imshow("Dominant Colours Histogram After Merging", merge_canvas);
	imwrite("Dominant Colours Histogram After Merging.jpg", merge_canvas);*/

	histogram = merge_canvas;

	//store the dominant colours
	int domColour_no = 1; //number of dominant colour after neglecting noices

	*topRankColour = merge_bin_sort[0].second;
	*pixelPercentage = merge_bin_sort[0].first;

	for (int i = 1; i < totalColour; i++) {
		if (merge_bin_sort[i].first > merge_bin_sort[i - 1].first * 0.1) {
			*(topRankColour + i) = merge_bin_sort[i].second;
			*(pixelPercentage + i) = merge_bin_sort[i].first;
			domColour_no++;
		}
		else
			break;
	}

	return domColour_no;
}

double euclideanDistance_nuMoment(double *param_query, double *param_dataset, int arrSize) {
	double total = 0.0, ed;

	for (int i = 0; i < arrSize; i++)
		total += pow(*(param_query + i) - *(param_dataset + i), 2);

	ed = sqrt(total);

	return ed;
}

double euclideanDistance_pixelPercentage(int *colour_query, int *colour_dataset, int *pp_query, int *pp_dataset, int arrSize_query, int arrSize_dataset, int imgSize_query,
	int imgSize_dataset) {
	double total = 0.0, ed;

	for (int i = 0; i < arrSize_query; i++) { //check query
		int notMatched = 0;

		for (int j = 0; j < arrSize_dataset; j++) {
			if (abs(convert2Bgr(*(colour_query + i))[0] / 32 - convert2Bgr(*(colour_dataset + j))[0] / 32) <= 2 &&
				abs(convert2Bgr(*(colour_query + i))[1] / 32 - convert2Bgr(*(colour_dataset + j))[1] / 32) <= 2 &&
				abs(convert2Bgr(*(colour_query + i))[2] / 32 - convert2Bgr(*(colour_dataset + j))[2] / 32) <= 2 &&
				(abs(convert2Bgr(*(colour_query + i))[0] / 32 - convert2Bgr(*(colour_dataset + j))[0] / 32) +
				abs(convert2Bgr(*(colour_query + i))[1] / 32 - convert2Bgr(*(colour_dataset + j))[1] / 32) +
				abs(convert2Bgr(*(colour_query + i))[2] / 32 - convert2Bgr(*(colour_dataset + j))[2] / 32)) <= 3) {
				total += pow(*(pp_query + i) / double(imgSize_query) - *(pp_dataset + j) / double(imgSize_dataset), 2);
				continue;
			}
			else
				notMatched++;

			if (notMatched == arrSize_dataset) //if no similar colour matched then minus 0
				total += pow(*(pp_query + i) / double(imgSize_query) - 0, 2);
		}
	}

	for (int i = 0; i < arrSize_dataset; i++) { //check dataset
		int notMatched = 0;

		for (int j = 0; j < arrSize_query; j++) {
			if (abs(convert2Bgr(*(colour_dataset + i))[0] / 32 - convert2Bgr(*(colour_query + j))[0] / 32) > 2 ||
				abs(convert2Bgr(*(colour_dataset + i))[1] / 32 - convert2Bgr(*(colour_query + j))[1] / 32) > 2 ||
				abs(convert2Bgr(*(colour_dataset + i))[2] / 32 - convert2Bgr(*(colour_query + j))[2] / 32) > 2 ||
				(abs(convert2Bgr(*(colour_dataset + i))[0] / 32 - convert2Bgr(*(colour_query + j))[0] / 32) +
				abs(convert2Bgr(*(colour_dataset + i))[1] / 32 - convert2Bgr(*(colour_query + j))[1] / 32) +
				abs(convert2Bgr(*(colour_dataset + i))[2] / 32 - convert2Bgr(*(colour_query + j))[2] / 32)) > 3)
				notMatched++;

			if (notMatched == arrSize_query) //if no similar colour matched then minus 0
				total += pow((*(pp_dataset + i) / double(imgSize_dataset)) - 0, 2);
		}
	}

	ed = sqrt(total);

	return ed;
}

void sort2DArray(double arr[][2], int row) {
	double tempValue;

	for (int current = 0; current < row; current++) {
		for (int back = current + 1; back < row; back++) {
			if (arr[current][1] > arr[back][1]) {
				for (int col = 0; col < 2; col++) {
					//sort index and euclidean distance
					tempValue = arr[current][col];
					arr[current][col] = arr[back][col];
					arr[back][col] = tempValue;
				}

			}
		}
	}
}

Mat attachResults2AWhiteBgWithHist(double arr[][2], vector<Mat> &trainImage, vector<Mat> &histogram, int number) {
	int size = 150;
	int x_train, x_hist, y_train, y_hist; //x is cols, y is rows
	int w = 5; //Maximum number of images in a row
	int h = 4; //Maximum number of images in a column
	int max_train, max_hist;
	double scale_train, scale_hist; //How much we have to resize the image
	Mat displayImage(Size(100 + size * w, 60 + size * h), CV_8UC3, Scalar(255, 255, 255));
	Mat resizedImage;

	for (int i = 0, m = 20, n = 20; i < 5; i++, m += (20 + size)){
		//Find the width and height of the image
		x_train = trainImage[arr[i][0]].cols;
		y_train = trainImage[arr[i][0]].rows;
		x_hist = histogram[arr[i][0]].cols;
		y_hist = histogram[arr[i][0]].rows;

		//Find whether height or width is greater in order to resize the image
		max_train = x_train > y_train ? x_train : y_train;
		max_hist = x_hist > y_hist ? x_hist : y_hist;

		//Find the scaling factor to resize the image
		scale_train = double(max_train) / size;
		scale_hist = double(max_hist) / size;

		//Align the images
		if (i % w == 0 && m != 20) {
			m = 20;
			n += 20 + size;
		}

		Rect roi(m, n, (int)(x_train / scale_train), (int)(y_train / scale_train)); //Set the image roi to display the current image
		resize(trainImage[arr[i][0]], resizedImage, Size(roi.width, roi.height)); //Resize the input image and copy the it to the large image
		resizedImage.copyTo(displayImage(roi));

		Rect roi2(m, n + size, (int)(x_hist / scale_hist), (int)(y_hist / scale_hist));
		resize(histogram[arr[i][0]], resizedImage, Size(roi2.width, roi2.height));
		resizedImage.copyTo(displayImage(roi2));
	}

	for (int i = 5, m = 20, n = 20; i < number; i++, m += (20 + size)){
		//Find the width and height of the image
		x_train = trainImage[arr[i][0]].cols;
		y_train = trainImage[arr[i][0]].rows;
		x_hist = histogram[arr[i][0]].cols;
		y_hist = histogram[arr[i][0]].rows;

		//Find whether height or width is greater in order to resize the image
		max_train = x_train > y_train ? x_train : y_train;
		max_hist = x_hist > y_hist ? x_hist : y_hist;

		//Find the scaling factor to resize the image
		scale_train = double(max_train) / size;
		scale_hist = double(max_hist) / size;

		//Align the images
		if (i % w == 0 && m != 20) {
			m = 20;
			n += 20 + size;
		}

		Rect roi(m, n + size * 2, (int)(x_train / scale_train), (int)(y_train / scale_train)); //Set the image roi to display the current image
		resize(trainImage[arr[i][0]], resizedImage, Size(roi.width, roi.height)); //Resize the input image and copy the it to the large image
		resizedImage.copyTo(displayImage(roi));

		Rect roi2(m, n + size * 3, (int)(x_hist / scale_hist), (int)(y_hist / scale_hist));
		resize(histogram[arr[i][0]], resizedImage, Size(roi2.width, roi2.height));
		resizedImage.copyTo(displayImage(roi2));
	}

	return displayImage;
}

Mat attachResults2AWhiteBg(double arr[][2], vector<Mat> &trainImage, int number) {
	int size = 150;
	int x, y; //x is cols, y is rows
	int w = 5; //Maximum number of images in a row
	int h = 2; //Maximum number of images in a column
	int max;
	double scale; //How much we have to resize the image
	Mat displayImage(Size(100 + size * w, 60 + size * h), CV_8UC3, Scalar(255, 255, 255));
	Mat resizedImage;

	for (int i = 0, m = 20, n = 20; i < number; i++, m += (20 + size)){
		//Find the width and height of the image
		x = trainImage[arr[i][0]].cols;
		y = trainImage[arr[i][0]].rows;

		//Find whether height or width is greater in order to resize the image
		max = x > y ? x : y;

		//Find the scaling factor to resize the image
		scale = double(max) / size;

		//Align the images
		if (i % w == 0 && m != 20) {
			m = 20;
			n += 20 + size;
		}

		Rect roi(m, n, (int)(x / scale), (int)(y / scale)); //Set the image roi to display the current image
		resize(trainImage[arr[i][0]], resizedImage, Size(roi.width, roi.height)); //Resize the input image and copy the it to the large image
		resizedImage.copyTo(displayImage(roi));
	}

	return displayImage;
}