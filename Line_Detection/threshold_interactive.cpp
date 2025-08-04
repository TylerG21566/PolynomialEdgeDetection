#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat img;
cv::Mat out;

void thresh_onchange(int val, void* userdata)
{
	cv::threshold(img, out, val, 255, cv::THRESH_BINARY);
	cv::imshow("Result", out);
}

int main(int argc, char *argv[])
{
	img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

	// Check if the image was successfully loaded
	if (img.empty()) {
			printf("Failed to load image '%s'\n", argv[1]);
			return -1;
	}

	cv::namedWindow("Result", cv::WINDOW_NORMAL);
	cv::createTrackbar("Threshold", "Result", NULL, 255, thresh_onchange);

	// Call the callback function for the initial processing
	thresh_onchange(0, NULL);

	// Wait for a key press before quitting
	cv::waitKey(0);

	return 0;
}
