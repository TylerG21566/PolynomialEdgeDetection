#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char *argv[])
{
	cv::Mat img;

	img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

	// Check if the image was successfully loaded
	if (img.empty()) {
			printf("Failed to load image '%s'\n", argv[1]);
			return -1;
	}

	// Invert the image
	for(int r = 0; r < img.rows; r++)
	{
		for (int c = 0; c < img.cols; c++)
		{
			img.at<uchar>(r, c) = 255 - img.at<uchar>(r, c);
		}
	}

	cv::namedWindow("Image", cv::WINDOW_NORMAL);
	cv::imshow("Image", img);

	// Wait for a key press before quitting
	cv::waitKey(0);

	return 0;
}
