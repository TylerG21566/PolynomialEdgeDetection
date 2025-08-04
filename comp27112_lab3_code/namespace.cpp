#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

int main(int argc, char *argv[])
{
	Mat img;
	Mat out;

	img = imread(argv[1], IMREAD_GRAYSCALE);

	// Check if the image was successfully loaded
	if (img.empty()) {
			printf("Failed to load image '%s'\n", argv[1]);
			return -1;
	}

	// Blur the image
	blur(img, out, Size(25, 25));

	namedWindow("Image", WINDOW_NORMAL);
	imshow("Image", out);

	// Wait for a key press before quitting
	waitKey(0);

	imwrite("blurred_image.jpg", out);

	return 0;
}
