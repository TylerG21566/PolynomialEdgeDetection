#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char *argv[])
{
	cv::Mat img;
	cv::Mat out;

	img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

	// Check if the image was successfully loaded
	if (img.empty()) {
			printf("Failed to load image '%s'\n", argv[1]);
			return -1;
	}

	// Blur the image
	cv::blur(img, out, cv::Size(40, 40));

	cv::namedWindow("Image", cv::WINDOW_NORMAL);
	cv::resizeWindow("Image", 800, 500);

	cv::imshow("Image", out);

	// Wait for a key press before quitting
	cv::waitKey(0);

	cv::imwrite("blurred_image.jpg", out);

	return 0;
}
