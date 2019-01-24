#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <omp.h>
#include <cmath>
using namespace cv;
using namespace std;


float *gussianKernel(float sigma = 1.0) {
	float temp[3][3];
	float pi = 3.14;
	float s = 2 * sigma *sigma;
	float sum = 0.0;
	for (int x = -1; x <= 1; x++) {
		for (int y = -1; y <= 1; y++) {
			float r = sqrt(x*x + y*y);
			temp[x + 1][y + 1] = (exp((-1 * r*r) / s) / s*pi);
			sum += temp[x + 1][y + 1];

		}
	}

	float* result = new float[8];
	memcpy(result, temp, sizeof(float) * 3 * 3);
	for (int i = 0; i < 9; i++) {
		result[i] /= sum;
	}
	return result;
	delete(result);
}
__global__ void bgr2img(uchar3 *src_img, uchar *dst_img, int rows, int cols) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int offset = y * cols + x;
	if (0 < x && x < cols && 0 < y && y < rows) {
		dst_img[offset] = src_img[offset].x * 0.3 + src_img[offset].y * 0.3 + src_img[offset].z * 0.3;

	}

}

__global__ void gussian_filter(uchar *src_img, uchar *dst_img, int rows, int cols, float gussian_kernel[]) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int offset = y * cols + x;
	
	if (0 < x && x < cols && 0 < y && y < rows) {
		float gradient = src_img[offset - cols - 1] * gussian_kernel[0] + src_img[offset - cols] * gussian_kernel[1] + src_img[offset - cols + 1] * gussian_kernel[2] +
			src_img[offset - 1] * gussian_kernel[3] + src_img[offset] * gussian_kernel[4] + src_img[offset + 1] * gussian_kernel[5] +
			src_img[offset + cols - 1] * gussian_kernel[6] + src_img[offset + cols] * gussian_kernel[7] + src_img[offset + cols + 1] * gussian_kernel[8];

	//	gradient = gradient < 0 ? -1 * gradient : gradient;
		dst_img[offset] = gradient;
	}
	
}
__global__ void Laplace_filter(uchar *src_img, uchar* dst_img, int rows, int cols) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int offset = y * cols + x;

	float laplaceKernel[9] = { -1 ,-1, -1,
							  -1 ,  8, -1,
							  -1 , -1, -1 };
	if (0 < x&& x < cols && 0 < y && y < rows) {
		float temp = src_img[offset - cols - 1] * laplaceKernel[0] + src_img[offset - cols] * laplaceKernel[1] + src_img[offset - cols + 1] * laplaceKernel[2] +
			src_img[offset - 1] * laplaceKernel[3] + src_img[offset] * laplaceKernel[4] + src_img[offset] * laplaceKernel[5] +
			src_img[offset + cols - 1] * laplaceKernel[6] + src_img[offset + cols] * laplaceKernel[7] + src_img[offset + cols + 1] * laplaceKernel[8];

		dst_img[offset] = temp < 0 ? temp * -1 : temp;

	}


}
main() {

	Mat image;
	float *sigma_1, *sigma_2, *sigma_3;
	sigma_1 = gussianKernel(1.0);
	float sumg = 0;
	for (int i = 0; i < 9; i++)
		sumg += sigma_1[i];
	cout << sumg << endl;
	sigma_2 = gussianKernel(4.0);
	sigma_3 = gussianKernel(16.0);
 	image = imread("BUTTERFLY.jpg", CV_LOAD_IMAGE_COLOR);

	uchar3 *bgrimg;
	uchar *grayimg;
	uchar *gussianimg;
	uchar *laplaceimg;
	int ncols = image.cols * image.rows;

	cudaMalloc(&bgrimg, ncols * sizeof(uchar3));
	cudaMalloc(&grayimg, ncols * sizeof(uchar));
	cudaMalloc(&gussianimg, ncols *sizeof(uchar));
	cudaMalloc(&laplaceimg, ncols * sizeof(uchar));
	cudaMemcpy(bgrimg, image.ptr(), ncols * sizeof(uchar3), cudaMemcpyHostToDevice);
	
	
	dim3 grid(16, 16);
	dim3 block(32, 32);
	bgr2img << <grid, block >> >(bgrimg, grayimg, image.rows, image.cols); 


	float *sigmaGpu;
	cudaMalloc(&sigmaGpu, 9 * sizeof(float));
	cudaMemcpy(sigmaGpu, sigma_1, 9, cudaMemcpyHostToDevice);
	gussian_filter << <grid, block >> >(grayimg, gussianimg, image.rows, image.cols,sigmaGpu);
	Mat result1(image.rows, image.cols, CV_8UC1);
	cudaMemcpy(result1.ptr(), gussianimg, ncols * sizeof(uchar), cudaMemcpyDeviceToHost);
	imshow("image1", result1);

	//Laplace_filter<<<grid,block>>>(gussianimg,laplaceimg,image.rows,image.cols);
	//Mat result1(image.rows,image.cols,CV_8UC1);
	//cudaMemcpy(result1.ptr(), laplaceimg, ncols * sizeof(uchar), cudaMemcpyDeviceToHost);
	//imshow("image1", result1);


	cudaMemcpy(sigmaGpu, sigma_2, 9, cudaMemcpyHostToDevice);
	gussian_filter << <grid, block >> >(grayimg, gussianimg, image.rows, image.cols, sigmaGpu);
	Mat result2(image.rows, image.cols, CV_8UC1);
	cudaMemcpy(result2.ptr(), gussianimg, ncols * sizeof(uchar), cudaMemcpyDeviceToHost);
	imshow("image2", result2);




	uchar *diff , *image1_ptr , *image2_ptr;
	Mat diff12(image.rows, image.cols,CV_8UC1);
	diff = diff12.ptr<uchar>(0);
	image1_ptr = result1.ptr<uchar>(0);
	image2_ptr = result2.ptr<uchar>(0);
	int negCount = 0;
	for (int i = 0; i < ncols; i++) {
		int temp = image1_ptr[i] - image2_ptr[i];
			
		if (temp < 0.5) {
			diff[i] = 0;
		}
		else
			diff[i] = 255;
	}
	//cout << negCount << endl;
	imshow("diff12", diff12);

	cudaFree(bgrimg);
	cudaFree(grayimg);
	cudaFree(gussianimg);

	waitKey(0);




}