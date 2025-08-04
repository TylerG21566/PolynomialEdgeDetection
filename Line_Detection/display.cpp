#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char *argv[])
{
	cv::Mat img; // Images are stored in Mat class (Mat = matrix)

	img = cv::imread(argv[1]);

	// Check if the image was successfully loaded
	if (img.empty()) {
			printf("Failed to load image '%s'\n", argv[1]);
			return -1;
	}

	cv::namedWindow("Image", cv::WINDOW_NORMAL);
	cv::resizeWindow("Image", 800, 500);
	cv::imshow("Image", img);

	// Wait for a key press before quitting
	cv::waitKey(0);

	return 0;
}
