#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "Header.h"

using namespace cv;
using namespace std;

int main(void) {
	char queryName[20];
	string option;
	bool terminate = false;
	int arrSize; //array size to be past into function

	//loading, processing, and displaying images
	Mat image, processedImage, roi_left_img, roi_right_img, roi_left_timg, roi_right_timg, displayResult;
	vector<String> fn_logo, fn_flag; //image file name
	vector<Mat> logo_trainImage, logo_processedTrainImage, flag_trainImage, flag_processedTrainImage; //training image
	
	//dominant colours histogram
	Mat tempImage;
	Mat canvas(620, 1000, CV_8UC3, Scalar(255, 255, 255)); //large image for display purpose
	vector<Mat> histogram; 
	for (int histSize = 0; histSize < SIZE_OF_ARRAY; histSize++)
		histogram.push_back(canvas);

	//dominant colour
	int topRankColour_img[SIZE_OF_ARRAY], topRankColour_img_left[SIZE_OF_ARRAY], topRankColour_img_right[SIZE_OF_ARRAY];
	int topRankColour_timg[SIZE_OF_ARRAY], topRankColour_timg_left[SIZE_OF_ARRAY], topRankColour_timg_right[SIZE_OF_ARRAY]; 
	int	totalDominantColour_img, totalDominantColour_timg;
	int totalDominantColour_roi_img[2], totalDominantColour_roi_timg[2]; //0 be left, 1 be right

	//pixel percentage
	int pixelPercentage_img[SIZE_OF_ARRAY], pixelPercentage_img_left[SIZE_OF_ARRAY], pixelPercentage_img_right[SIZE_OF_ARRAY]; //height of the colour histogram
	int	pixelPercentage_timg[SIZE_OF_ARRAY], pixelPercentage_timg_left[SIZE_OF_ARRAY], pixelPercentage_timg_right[SIZE_OF_ARRAY];
	
	//euclidean distance
	double ed_nuMoment[SIZE_OF_ARRAY][2], nuMoment_img[7], nuMoment_timg[7], index_huMoment[SIZE_OF_ARRAY][2]; //arrays to store moment
	double ed_pixelPercentage[SIZE_OF_ARRAY][2], ed_pixelPercentage_left[SIZE_OF_ARRAY][2], ed_pixelPercentage_right[SIZE_OF_ARRAY][2]; //arrarys to store pixel percentage
	double ed_colourRegion[SIZE_OF_ARRAY][2];
	double partitionFiltering[SIZE_OF_ARRAY][2]; //average ed between left ed and right ed

	while (cin) {
		//compute training images
		logo_processedTrainImage.clear(); //make sure the memory is empty
		flag_processedTrainImage.clear();

		glob("Logo_Datasets/*.jpg", fn_logo, false); //retrieve logo training image
		glob("Flag_Datasets/*.jpg", fn_flag, false); //retrieve flag training image

		for (int i = 0; i < LOGO_TRAINING_IMAGES; i++)
			logo_trainImage.push_back(imread(fn_logo[i], IMREAD_COLOR));

		for (int j = 0; j < LOGO_TRAINING_IMAGES; j++)
			logo_processedTrainImage.push_back(logo_trainImage[j]);

		for (int i = 0; i < FLAG_TRAINING_IMAGES; i++)
			flag_trainImage.push_back(imread(fn_flag[i], IMREAD_COLOR));

		for (int j = 0; j < FLAG_TRAINING_IMAGES; j++)
			flag_processedTrainImage.push_back(flag_trainImage[j]);

		cout << "******************************************\n";
		cout << "***** Similar Image Retrieval System *****\n";
		cout << "******************************************\n\n";
		cout << "Please input query image: ";
		cin.getline(queryName, 20);

		image = imread(queryName, IMREAD_COLOR);

		processedImage = image;

		if (image.empty()) {
			cout << "No image found.\n";
			system("pause");
			system("cls");
		}
		else {
			while (!terminate) {
				system("cls");
				cout << "******************************************\n";
				cout << "***** Similar Image Retrieval System *****\n";
				cout << "******************************************\n\n";
				cout << "    (1)\tColour Descripor (Flag)\n";
				cout << "    (2)\tShape Descripor (Logo)\n";
				cout << "    (3)\tCombined Descriptor (Logo)\n";
				cout << "    (4)\tTry Another Query Image\n";
				cout << "    (5)\tExit\n\n";
				cout << "Please enter your option: ";
				getline(cin, option);
				cout << endl;

				if (option.length() != 1) {
					cout << "Invalid input.\n";
					system("pause");
				}
				else {
					switch (option[0]) {
						case '1':
							displayQueryImage(image);
							
							//first stage filtering
							totalDominantColour_img = wholeImage(image, processedImage, topRankColour_img, pixelPercentage_img, tempImage);
							
							for (int imgIndex = 0; imgIndex < FLAG_TRAINING_IMAGES; imgIndex++) {
								totalDominantColour_timg = wholeImage(flag_trainImage[imgIndex], flag_processedTrainImage[imgIndex],
									topRankColour_timg, pixelPercentage_timg, histogram[imgIndex]);
								
								//similarity score computed by ed of dominant colour pixel percentage
								ed_pixelPercentage[imgIndex][0] = imgIndex;
								ed_pixelPercentage[imgIndex][1] = euclideanDistance_pixelPercentage(topRankColour_img, topRankColour_timg, pixelPercentage_img, 
									pixelPercentage_timg, totalDominantColour_img, totalDominantColour_timg, image.rows * image.cols, 
									flag_trainImage[imgIndex].rows * flag_trainImage[imgIndex].cols);
							}

							sort2DArray(ed_pixelPercentage, FLAG_TRAINING_IMAGES); //sort in ascending order

							//display the top 10 rank images in term of similarity measurement on a white background
							//for (int count = 0; count < 10; count++)
							//	cout << "ED of Pixel Percentage - " << fn_flag[ed_pixelPercentage[count][0]] << ": " <<ed_pixelPercentage[count][1] << endl;

							//displayResult = attachResults2AWhiteBgWithHist(ed_pixelPercentage, flag_trainImage, histogram, 10);
							//namedWindow("Pixel Percentage", WINDOW_AUTOSIZE);
							//imshow("Pixel Percentage", displayResult);
							//imwrite("Pixel Percentage.jpg", displayResult);

							//second stage filtering
							for (int imgIndex = 0; imgIndex < 10; imgIndex++) {
								totalDominantColour_timg = wholeImage(flag_trainImage[ed_pixelPercentage[imgIndex][0]], 
									flag_processedTrainImage[ed_pixelPercentage[imgIndex][0]], topRankColour_timg, pixelPercentage_timg, tempImage);

								ed_colourRegion[imgIndex][0] = ed_pixelPercentage[imgIndex][0];
								ed_colourRegion[imgIndex][1] = euclideanDistance_colourRegion(processedImage, flag_processedTrainImage[ed_pixelPercentage[imgIndex][0]], 
									topRankColour_img, topRankColour_timg, totalDominantColour_img, totalDominantColour_timg);
							}

							sort2DArray(ed_colourRegion, 10);

							//display the top 10 rank images in term of similarity measurement on a white background
							for (int count = 0; count < 10; count++)
								cout << "ED of Colour Region Moment - " << fn_flag[ed_colourRegion[count][0]] << ": " << ed_colourRegion[count][1] << endl;

							displayResult = attachResults2AWhiteBgWithHist(ed_colourRegion, flag_trainImage, histogram, 10);
							namedWindow("Colour Region Moment", WINDOW_AUTOSIZE);
							imshow("Colour Region Moment", displayResult);
							imwrite("Colour Region Moment.jpg", displayResult);

							roi(image, roi_left_img, roi_right_img);
							totalDominantColour_roi_img[0] = wholeImage(roi_left_img, processedImage, topRankColour_img_left, pixelPercentage_img_left, tempImage);
							totalDominantColour_roi_img[1] = wholeImage(roi_right_img, processedImage, topRankColour_img_right, pixelPercentage_img_right, tempImage);

							for (int imgIndex = 0; imgIndex < 20; imgIndex++) {
								roi(flag_trainImage[ed_pixelPercentage[imgIndex][0]], roi_left_timg, roi_right_timg);
								totalDominantColour_roi_timg[0] = wholeImage(roi_left_timg, flag_processedTrainImage[ed_pixelPercentage[imgIndex][0]],
									topRankColour_timg_left, pixelPercentage_timg_left, tempImage);
								totalDominantColour_roi_timg[1] = wholeImage(roi_right_timg, flag_processedTrainImage[ed_pixelPercentage[imgIndex][0]],
									topRankColour_timg_right, pixelPercentage_timg_right, tempImage);

								//pixel percentage
								ed_pixelPercentage_left[imgIndex][0] = ed_pixelPercentage[imgIndex][0];
								ed_pixelPercentage_left[imgIndex][1] = euclideanDistance_pixelPercentage(topRankColour_img_left, topRankColour_timg_left,
									pixelPercentage_img_left, pixelPercentage_timg_left, totalDominantColour_roi_img[0], totalDominantColour_roi_timg[0],
									roi_left_img.rows * roi_left_img.cols, roi_left_timg.rows * roi_left_timg.cols);

								ed_pixelPercentage_right[imgIndex][0] = ed_pixelPercentage[imgIndex][0];
								ed_pixelPercentage_right[imgIndex][1] = euclideanDistance_pixelPercentage(topRankColour_img_right, topRankColour_timg_right,
									pixelPercentage_img_right, pixelPercentage_timg_right, totalDominantColour_roi_img[1], totalDominantColour_roi_timg[1],
									roi_right_img.rows * roi_right_img.cols, roi_right_timg.rows * roi_right_timg.cols);

								partitionFiltering[imgIndex][0] = ed_pixelPercentage[imgIndex][0];
								partitionFiltering[imgIndex][1] = (ed_pixelPercentage_left[imgIndex][1] + ed_pixelPercentage_right[imgIndex][1]) / 2;
							}

							sort2DArray(partitionFiltering, 10);

							//display the top 10 rank images in term of similarity measurement on a white background
							for (int count = 0; count < 10; count++)
								cout << "ED of Average Pixel Percentage - " << fn_flag[partitionFiltering[count][0]] << ": " << partitionFiltering[count][1] << endl;

							displayResult = attachResults2AWhiteBgWithHist(partitionFiltering, flag_trainImage, histogram, 10);
							namedWindow("Average Pixel Percentage", WINDOW_AUTOSIZE);
							imshow("Average Pixel Percentage", displayResult);
							imwrite("Average Pixel Percentage.jpg", displayResult);

							waitKey();
							break;
						case '2':
							displayQueryImage(image);

							//central normalized moments filtering
							findContour(image, processedImage, nuMoment_img);
							
							for (int imgIndex = 0; imgIndex < LOGO_TRAINING_IMAGES; imgIndex++) {
								findContour(logo_trainImage[imgIndex], logo_processedTrainImage[imgIndex], nuMoment_timg);

								//assign image index
								ed_nuMoment[imgIndex][0] = imgIndex;

								//assign euclidean distance of moment
								ed_nuMoment[imgIndex][1] = euclideanDistance_nuMoment(nuMoment_img, nuMoment_timg, 7);		
							}

							sort2DArray(ed_nuMoment, LOGO_TRAINING_IMAGES); //sort in ascending order

							//display the top 10 rank images in term of similarity measurement on a white background
							for (int count = 0; count < 10; count++)
								cout << "ED of Moment - " << fn_logo[ed_nuMoment[count][0]] << ": " << ed_nuMoment[count][1] << endl;

							displayResult = attachResults2AWhiteBg(ed_nuMoment, logo_trainImage, 10);
							namedWindow("Moment", WINDOW_AUTOSIZE);
							imshow("Moment", displayResult);
							imwrite("Moment.jpg", displayResult);

							//hu moments filtering
							for (int imgIndex = 0; imgIndex < 10; imgIndex++) {
								//assign image index
								index_huMoment[imgIndex][0] = ed_nuMoment[imgIndex][0];

								//assign Hu Moment value
								index_huMoment[imgIndex][1] = matchShapes(processedImage, logo_processedTrainImage[ed_nuMoment[imgIndex][0]], CV_CONTOURS_MATCH_I1, 0);
							}

							sort2DArray(index_huMoment, 10);

							//display the top 10 rank images in term of similarity measurement on a white background
							for (int count = 0; count < 10; count++)
								cout << "ED of Hu Moment - " << fn_logo[index_huMoment[count][0]] << ": " << index_huMoment[count][1] << endl;

							displayResult = attachResults2AWhiteBg(index_huMoment, logo_trainImage, 10);
							namedWindow("Hu Moment", WINDOW_AUTOSIZE);
							imshow("Hu Moment", displayResult);
							imwrite("Hu Moment.jpg", displayResult);

							waitKey();
							break;
						case '3':
							displayQueryImage(image);

							//first stage, shape descriptor
							findContour(image, processedImage, nuMoment_img);
							
							for (int imgIndex = 0; imgIndex < LOGO_TRAINING_IMAGES; imgIndex++) {
								findContour(logo_trainImage[imgIndex], logo_processedTrainImage[imgIndex],nuMoment_timg);

								//assign image index
								ed_nuMoment[imgIndex][0] = imgIndex;

								//assign euclidean distance of moment
								ed_nuMoment[imgIndex][1] = euclideanDistance_nuMoment(nuMoment_img, nuMoment_timg, 7);
							}

							sort2DArray(ed_nuMoment, LOGO_TRAINING_IMAGES); //sort in ascending order
							
							//second stage, colour descriptor
							totalDominantColour_img = eliminateBackground(image, processedImage, topRankColour_img, pixelPercentage_img, tempImage);
							
							for (int imgIndex = 0; imgIndex < 10; imgIndex++) {
								totalDominantColour_timg = eliminateBackground(logo_trainImage[ed_nuMoment[imgIndex][0]], logo_processedTrainImage[ed_nuMoment[imgIndex][0]],
									topRankColour_timg, pixelPercentage_timg, histogram[ed_nuMoment[imgIndex][0]]);
								
								//similarity score computed by ed of dominant colour pixel percentage
								ed_pixelPercentage[imgIndex][0] = ed_nuMoment[imgIndex][0];
								ed_pixelPercentage[imgIndex][1] = euclideanDistance_pixelPercentage(topRankColour_img, topRankColour_timg, pixelPercentage_img,
									pixelPercentage_timg, totalDominantColour_img, totalDominantColour_timg, image.rows * image.cols,
									logo_trainImage[imgIndex].rows * logo_trainImage[imgIndex].cols);
							}

							sort2DArray(ed_pixelPercentage, 10); //sort in ascending order

							roi(image, roi_left_img, roi_right_img);
							totalDominantColour_roi_img[0] = eliminateBackground(roi_left_img, processedImage, topRankColour_img_left, pixelPercentage_img_left, tempImage);
							totalDominantColour_roi_img[1] = eliminateBackground(roi_right_img, processedImage, topRankColour_img_right, pixelPercentage_img_right,
								tempImage);

							for (int imgIndex = 0; imgIndex < 10; imgIndex++) {
								roi(logo_trainImage[ed_pixelPercentage[imgIndex][0]], roi_left_timg, roi_right_timg);
								totalDominantColour_roi_timg[0] = eliminateBackground(roi_left_timg, processedImage, topRankColour_timg_left, pixelPercentage_timg_left, 
									tempImage);
								totalDominantColour_roi_timg[1] = eliminateBackground(roi_right_timg, processedImage, topRankColour_timg_right, pixelPercentage_timg_right, 
									tempImage);

								//pixel percentage
								ed_pixelPercentage_left[imgIndex][0] = ed_pixelPercentage[imgIndex][0];
								ed_pixelPercentage_left[imgIndex][1] = euclideanDistance_pixelPercentage(topRankColour_img_left, topRankColour_timg_left,
									pixelPercentage_img_left, pixelPercentage_timg_left, totalDominantColour_roi_img[0], totalDominantColour_roi_timg[0],
									roi_left_img.rows * roi_left_img.cols, roi_left_timg.rows * roi_left_timg.cols);

								ed_pixelPercentage_right[imgIndex][0] = ed_pixelPercentage[imgIndex][0];
								ed_pixelPercentage_right[imgIndex][1] = euclideanDistance_pixelPercentage(topRankColour_img_right, topRankColour_timg_right,
									pixelPercentage_img_right, pixelPercentage_timg_right, totalDominantColour_roi_img[1], totalDominantColour_roi_timg[1],
									roi_right_img.rows * roi_right_img.cols, roi_right_timg.rows * roi_right_timg.cols);

								partitionFiltering[imgIndex][0] = ed_pixelPercentage[imgIndex][0];
								partitionFiltering[imgIndex][1] = (ed_pixelPercentage_left[imgIndex][1] + ed_pixelPercentage_right[imgIndex][1]) / 2;
							}

							sort2DArray(partitionFiltering, 10);

							//display the top 10 rank images in term of similarity measurement on a white background
							cout << "\t\t\t\tAverage Pixel Percentage\tMoment\n";
							for (int count = 0; count < 10; count++) {
								cout << fn_logo[partitionFiltering[count][0]] << ":\t" << partitionFiltering[count][1];

								for (int count2 = 0; count2 < 10; count2++) {
									if (partitionFiltering[count][0] == ed_nuMoment[count2][0])
										cout << "\t\t\t" << ed_nuMoment[count2][1] << endl;
								}
							}

							displayResult = attachResults2AWhiteBgWithHist(partitionFiltering, logo_trainImage, histogram, 10);
							namedWindow("Similarity Score of Combined Discriptor", WINDOW_AUTOSIZE);
							imshow("Similarity Score of Combined Discriptor", displayResult);
							imwrite("Similarity Score of Combined Discriptor.jpg", displayResult);
						
							waitKey();
							break;
						case '4':
							system("cls");
							break;
						case '5':
							return 0;
						default:
							cout << "Invalid input.\n";
							system("pause");
					}

					if (option[0] == '4')
						break;
					else
						destroyAllWindows();
				}
			}
		}
	}
}