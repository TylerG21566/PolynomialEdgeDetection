#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

const int HIST_IMG_HEIGHT = 400;

// Declare global variables
Mat img; // Original image
Mat out; // Thresholded output image

void createHistogram(Mat &img, Mat &hist)
{
    int histSize = 256; // Number of bins (256 possible intensity values for grayscale)
    vector<int> histData(histSize, 0);
    // Create a black canvas for the histogram image
    Mat histImage(HIST_IMG_HEIGHT, 512, CV_8UC3, Scalar(255, 255, 255)); // Black canvas to draw the histogram

    bool uniform = true, accumulate = false;
    int imgRows = img.rows;
    int imgCols = img.cols;
    int max_height = 0;

    // 1. Iterate through the image to calculate the histogram (frequency of pixel intensities)
    for (int i = 0; i < imgRows; i++)
    {
        for (int j = 0; j < imgCols; j++)
        {
            // Get the pixel intensity value (grayscale value)
            int pixelValue = img.at<uchar>(i, j);

            // Increment the corresponding histogram bin for this pixel intensity
            histData[pixelValue]++;
            max_height = max(histData[pixelValue], max_height);
        }
    }

    // 2. Normalize the histogram
    int hist_w = 512;
    int hist_h = HIST_IMG_HEIGHT;
    int bin_w = cvRound((double)hist_w / histSize); // Width of each bin

    for (int i = 0; i < histSize; i++)
    {
        histData[i] = cvRound((float)histData[i] * hist_h / max_height); // Normalize based on max count
    }

    for (int i = 0; i < histSize; i++)
    {
        int binVal = histData[i]; // already normalized value
        // Calculate the x position and width of each bin
        int x = i * bin_w;
        // Draw a filled rectangle for each bin
        rectangle(histImage, Point(x, hist_h - binVal), Point(x + bin_w, hist_h),
                  Scalar(25, 25, 25), cv::FILLED);
    }

    // Add vertical lines at intensity 127 and 255
    int lineHeight = hist_h; // The line spans the full height of the histogram
    // Create an overlay from histImage
    Mat overlay = histImage.clone();

    // Define line color and thickness
    Scalar lineColor(50, 50, 50); // Dark gray
    int thickness = 1;
    int lineType = LINE_AA; // Anti-aliased line

    // Draw vertical lines at the desired intensities.
    // Note: Since each bin is 2 pixels wide (512/256),
    // the x position is calculated as intensity * bin_w.
    line(overlay, Point(255 * bin_w, 0), Point(255 * bin_w, lineHeight), lineColor, thickness, lineType, 0);
    line(overlay, Point(192 * bin_w, 0), Point(192 * bin_w, lineHeight), lineColor, thickness, lineType, 0);
    line(overlay, Point(128 * bin_w, 0), Point(128 * bin_w, lineHeight), lineColor, thickness, lineType, 0);
    line(overlay, Point(64 * bin_w, 0), Point(64 * bin_w, lineHeight), lineColor, thickness, lineType, 0);
    line(overlay, Point(0 * bin_w, 0), Point(0 * bin_w, lineHeight), lineColor, thickness, lineType, 0);

    // Blend the overlay with the original histogram image.
    // alpha controls the transparency of the lines (0.0 - 1.0)
    double alpha = 0.7; // 50% transparency for the lines
    addWeighted(overlay, alpha, histImage, 1 - alpha, 0, histImage);
    // Set the calculated histogram image in the input parameter "hist"
    hist = histImage;
}

void thresh_onchange(int val, void *userdata)
{
    // Access the image pointer from userdata
    Mat *img_ptr = (Mat *)userdata;
    // Perform thresholding and store the result in 'out'
    cv::threshold(*img_ptr, out, val - 1, 256, cv::THRESH_BINARY);
    // Show the output image
    cv::imshow("Result", out);
}

int main(int argc, char *argv[])
{
    img = imread(argv[1], IMREAD_GRAYSCALE);
    // Set target width
    int targetWidth = 800;

    // Calculate new height to maintain aspect ratio
    double aspectRatio = static_cast<double>(img.cols) / img.rows;
    int newHeight = static_cast<int>(targetWidth / aspectRatio);

    // Resize both images
    cv::resize(img, img, cv::Size(targetWidth, newHeight));
    cv::imshow("a window", img);

    // Check if the image was successfully loaded
    if (img.empty())
    {
        printf("Failed to load image '%s'\n", argv[1]);
        return -1;
    }

    Mat hist;
    // Create image histogram
    createHistogram(img, hist);

    // Show the histogram in a window
    namedWindow("Histogram", WINDOW_NORMAL);
    imwrite("hist.jpg", hist); // Save the histogram image to file
    imshow("Histogram", hist);

    // Create a window for the result and set up a trackbar
    namedWindow("Result", WINDOW_NORMAL);

    // Pass the image pointer as the userdata to the callback function
    cv::createTrackbar("Threshold", "Result", NULL, 256, thresh_onchange, (void *)&img);

    // Call the callback function for the initial processing
    thresh_onchange(0, (void *)&img);

    // Wait for a key press before quitting
    waitKey(0);

    return 0;
}
