#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

//Richardson-Lucy deconvolution
Mat rl_deconvol(Mat observed, Mat psf, int iterations)
{
	// Uniform grey starting estimation
	Mat latent_est = Mat(observed.size(), observed.type());
	observed.copyTo(latent_est);

	Mat psf_hat;
	flip(psf, psf_hat, -1);

	Mat est_conv;
	Mat relative_blur;
	Mat error_est;

	// Iterate
	for (int i=0; i<iterations; i++) {
		filter2D(latent_est, est_conv, -1, psf);
		// Element-wise division
		relative_blur = observed.mul(1.0/est_conv);
		filter2D(relative_blur, error_est, -1, psf_hat);
		// Element-wise multiplication
		latent_est = latent_est.mul(error_est);
	}

	return latent_est;
}

// Calculate a gaussian blur psf.
Mat genPSF(int sz) {
	Mat psf = Mat(Size(sz, sz), CV_32FC1, 0.0f);
	
	float sigma_row = sz/8;
	float sigma_col = sz/8;
	float mean_row = sz/2.0;
	float mean_col = sz/2.0;
		
	Scalar sum0;
	for (int j = 0; j<psf.rows; j++) {
		for (int k = 0; k<psf.cols; k++) {
			float temp = exp(
					-(pow((j - mean_row) / sigma_row, 2.0) + 
					  pow((k - mean_col) / sigma_col, 2.0))
					  ) / (2* M_PI * sigma_row * sigma_col);			
			psf.at<float>(j,k) = temp;
		}
	}

	sum0 = sum(psf);
	psf = psf / sum0;
	
	// float dmin, dmax;
	// Point minLoc, maxLoc;
	// minMaxLoc(psf, &dmin, &dmax, &minLoc, &maxLoc);
	// float alpha = 255 / (dmax - dmin), beta = - alpha * dmin; 
	// Mat psf_view;
	// convertScaleAbs(psf, psf_view, alpha, beta);
	// //imshow("Float", psf_view); waitKey(0); 

	return psf;
}

int main( int argc, const char** argv )
{
	if (argc != 3) {
		cout << "Usage: " << argv[0] << " image iterations" << "\n";
		return -1;
	}
	int iterations = atoi(argv[2]);
	Mat original_image = imread(argv[1], 0);
	Size imsz = original_image.size();

	// From here on, use 64-bit floats	// Convert original_image to float
	int type_f = CV_32FC1;
	Mat org_flt;
	original_image.convertTo(org_flt, type_f, 1.f/255);
	
	Mat psf = genPSF(35);

	// Blur the float_image with the psf.
	Mat blurred_flt = org_flt.clone();
	filter2D(org_flt, blurred_flt, -1, psf);

	Mat estimation = rl_deconvol(blurred_flt, psf, iterations);

	Mat est_view = Mat(imsz.height, imsz.width*3, type_f);
	Rect rc_lft = Rect(0, 				0, imsz.width, imsz.height);
	Rect rc_mid = Rect(imsz.width,		0, imsz.width, imsz.height);
	Rect rc_rht = Rect(imsz.width*2,	0, imsz.width, imsz.height);

	org_flt.copyTo(Mat(est_view, rc_lft));
	blurred_flt.copyTo(Mat(est_view, rc_mid));
	estimation.copyTo(Mat(est_view, rc_rht));

	namedWindow("Estimation", 0);	
	imshow("Estimation", est_view);
	waitKey(10); //cause update 
	setWindowProperty("Estimation", WND_PROP_FULLSCREEN, 1); 
	waitKey(0); 

	return 0;
}