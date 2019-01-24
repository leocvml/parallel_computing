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
__global__ void GPUbgr2gray(uchar3 *rgbimg, uchar *grayimg, int ncols) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < ncols; i += stride) {
		grayimg[i] = rgbimg[i].x * 0.3 + rgbimg[i].y * 0.3 + rgbimg[i].z * 0.3;
	}

}

__global__ void GPUbgr2gray2D(uchar3 *rgbimg, uchar *grayimg, int rows, int cols) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = cols * y + x;
	if (x < cols && y < rows) {
		grayimg[i] = rgbimg[i].x * 0.3 + rgbimg[i].y * 0.3 + rgbimg[i].z * 0.3;
	}

}

__global__ void GPUGussian(uchar *grayimg, uchar *GussainImg, int rows, int cols,float Gussian_kernel[]) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = cols * y + x;
	



	if (0 < x && x <cols && 0 < y && y<rows) {
		double gy = grayimg[i - cols - 1] * Gussian_kernel[0] + grayimg[i - cols] * Gussian_kernel[1] + grayimg[i - cols + 1] * Gussian_kernel[2] +
			grayimg[i - 1] * Gussian_kernel[3] + grayimg[i] * Gussian_kernel[4] + grayimg[i + 1] * Gussian_kernel[5] +
			grayimg[i + cols - 1] * Gussian_kernel[6] + grayimg[i + cols] * Gussian_kernel[7] + grayimg[i + cols + 1] * Gussian_kernel[8];
		

		gy = gy < 0 ? -1 * gy : gy;
		GussainImg[i] = gy;
	}

}
__global__ void GPULaplace(uchar *grayimg, uchar *LaplaceImg, int rows, int cols) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = cols * y + x;

	int laplace[9] = { -1, -1, -1,
		-1,  8, -1,
		-1, -1, -1 };

	if (0 < x && x <cols && 0 < y && y<rows) {
		int g = grayimg[i - cols - 1] * laplace[0] + grayimg[i - cols] * laplace[1] + grayimg[i - cols + 1] * laplace[2] +
			grayimg[i - 1] * laplace[3] + grayimg[i] * laplace[4] + grayimg[i + 1] * laplace[5] +
			grayimg[i + cols - 1] * laplace[6] + grayimg[i + cols] * laplace[7] + grayimg[i + cols + 1] * laplace[8];


		g = g < 0 ? -1 * g : g;
		LaplaceImg[i] = g;
	}

}


void CPUbgr2gray(uchar3 *rgbimg, uchar *grayimg, int ncols) {
	for (int i = 0; i < ncols; i++) {
		grayimg[i] = rgbimg[i].x *0.3 + rgbimg[i].y * 0.3 + rgbimg[i].z * 0.3;
	}
}

float *gussianKernel(float sigma, const int size = 3) {
	float temp[3][3];
	float pi = 3.14;
	float s = 2  *sigma *sigma;
	float sum = 0.0;
	for (int x = -1; x <= 1; x++) {
		for (int y = -1; y <= 1; y++) {
			float r = sqrt(x*x + y*y);
			temp[x+1][y+1] = (exp((-1 * r*r) / s) / s*pi);
			sum += temp[x+1][y+1];
			
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
int main(int argc, char** argv)
{
	float *sigma_temp, *sigmaGPU;

	
	sigma_temp = gussianKernel(2.5);



	cudaFree(0);
	Mat image;
	image = imread("BUTTERFLY.jpg", CV_LOAD_IMAGE_COLOR);
	resize(image, image, Size(), 0.75, 0.75);
	
	int img_rows = image.rows;
	int img_cols = image.cols;


	cout << img_rows * img_cols << endl;
	int ncols = img_rows *img_cols;


	uchar3 *bgr_ptr;
	uchar *gray_ptr;
	uchar *gussian_ptr;
	uchar *Laplace_ptr;
	Mat result = Mat::zeros(Size(image.cols, image.rows), CV_8UC1);

	bgr_ptr = image.ptr<uchar3>(0);
	gray_ptr = result.ptr<uchar>(0);
	double time = omp_get_wtime();
	
	cudaMalloc(&bgr_ptr, ncols * sizeof(uchar3));
	cudaMalloc(&gray_ptr, ncols * sizeof(uchar));
	cudaMalloc(&gussian_ptr, ncols*sizeof(uchar));
	cudaMalloc(&Laplace_ptr, ncols*sizeof(uchar));

	cudaMalloc(&sigmaGPU, 9*sizeof(float));
	
	cudaMemcpy(bgr_ptr, image.ptr(), ncols * sizeof(uchar3), cudaMemcpyHostToDevice);
	cudaMemcpy(sigmaGPU, sigma_temp,9, cudaMemcpyHostToDevice);

	//int block = 319;
	//int grid = ncols / block;
	//

	dim3 grid(16, 16); // grid = 16 x 16 blocks
	dim3 block(32, 32);

	//GPUbgr2gray << <grid, block >> >(input_ptr, result_ptr, ncols);
	GPUbgr2gray2D<<<grid,block>>>(bgr_ptr, gray_ptr, image.rows,image.cols);
	GPUGussian << <grid, block >> >(gray_ptr, gussian_ptr, image.rows, image.cols, sigmaGPU);
//	GPULaplace << <grid, block >> >(gussian_ptr, Laplace_ptr, image.rows, image.cols);
	

	
	cudaMemcpy(result.ptr(), gussian_ptr, ncols * sizeof(uchar), cudaMemcpyDeviceToHost);


	//CPUbgr2gray(input_ptr, result_ptr, ncols);
 
	double dtime = (omp_get_wtime() - time);
	cout << dtime << endl;

	imshow("GPU", result);
	cudaFree(gussian_ptr);
	cudaFree(gray_ptr);
	cudaFree(bgr_ptr);
	cudaFree(Laplace_ptr);


	waitKey(0);

}