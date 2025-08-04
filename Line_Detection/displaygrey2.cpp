#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char *argv[])
{
	cv::Mat img;

	img = cv::imread(argv[1]); // Load image in colour

	// Check if the image was successfully loaded
	if (img.empty()) {
			printf("Failed to load image '%s'\n", argv[1]);
			return -1;
	}

	// Make a greyscale copy of the image
	cv::Mat grey_img;
	cv::cvtColor(img, grey_img, cv::COLOR_BGR2Luv);

	cv::namedWindow("Colour Image", cv::WINDOW_NORMAL);
	cv::resizeWindow("Colour Image",600,400);
	cv::imshow("Colour Image", img);

	cv::namedWindow("Greyscale Image", cv::WINDOW_NORMAL);
	cv::resizeWindow("Greyscale Image",600,400);
	cv::imshow("Greyscale Image", grey_img);

	// Wait for a key press before quitting
	cv::waitKey(0);

	return 0;
}
