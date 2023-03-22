#define LOGO_TRAINING_IMAGES 61
#define FLAG_TRAINING_IMAGES 76
#define SIZE_OF_ARRAY 100

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void displayQueryImage(Mat &image);
int calcColourBin(Vec3b colour);
Vec3b convert2Bgr(int colourBin);
void roi(Mat &image, Mat &roi_left, Mat &roi_right);
int sortDominantColour(vector<int> &colourVote, vector<pair<int, int>> &bin_sort, int &max_fre);
int wholeImage(Mat &image, Mat &resultImage, int *topRankColour, int *pixelPercentage, Mat &histogram);
void centralNormalizedMoment(Mat &binaryImage, double *nuMoment);
double euclideanDistance_colourRegion(Mat &image, Mat &trainImage, int *colour_query, int *colour_dataset, int totalColour_img, int totalColour_timg);
void findContour(Mat &image, Mat &resultImage, double *nuMoment);
int eliminateBackground(Mat &image, Mat &processedImage, int *topRankColour, int *pixelPercentage, Mat &histogram);
double euclideanDistance_nuMoment(double *param_query, double *param_dataset, int arrSize);
double euclideanDistance_pixelPercentage(int *colour_query, int *colour_dataset, int *pp_query, int *pp_dataset, int arrSize_query, int arrSize_dataset, int imgSize_query, 
	int imgSize_dataset);
void sort2DArray(double arr[][2], int row);
Mat attachResults2AWhiteBgWithHist(double arr[][2], vector<Mat> &trainImage, vector<Mat> &histogram, int number);
Mat attachResults2AWhiteBg(double arr[][2], vector<Mat> &trainImage, int number);