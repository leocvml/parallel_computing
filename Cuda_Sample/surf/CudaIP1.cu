#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <omp.h>

using namespace cv;
using namespace std;



__global__ void GPU_BGR2gray(uchar *input, uchar *output,int ncols) {

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i = index; i < ncols; i += stride) {
	
			output[i] = input[i] * 0.3 + input[i + 1] * 0.3 + input[i + 2] * 0.3;
			output[i + 1] = output[i];
			output[i + 2] = output[i];
		
		
	}

}

void CPU_BGR2gray(uchar *input, uchar *output, int ncols) {

	for (int i = 0; i < ncols; i+=3) {
		output[i] = input[i] * 0.3 + input[i + 1] * 0.3 + input[i + 2] * 0.3;
		output[i + 1] = output[i];
		output[i + 2] = output[i];
	}
}

int main(int argc, char** argv)
{
	cudaFree(0);
	Mat image;
	image = imread("512.jpg", CV_LOAD_IMAGE_COLOR); 

	int channels = image.channels();
	int img_rows = image.rows;
	int img_cols = image.cols;

	cout << "channel :" << channels << endl;
	cout << "rows :" << img_rows << endl;
	cout << "cols :" << img_cols << endl;

	cout << channels * img_rows * img_cols<<endl;
	int ncols = channels * img_rows *img_cols;
	

	uchar *input_ptr, *result_ptr;

	Mat result = Mat::zeros(Size(image.cols, image.rows), CV_8UC3);



	int blockSize = 256;
	int numBlocks = (ncols/blockSize);
	double time = omp_get_wtime();

	//cudaMallocManaged(&input_ptr, ncols*sizeof(uchar));
	//cudaMallocManaged(&result_ptr, ncols*sizeof(uchar));
	cudaMalloc<uchar>(&input_ptr, ncols);
	cudaMalloc<uchar>(&result_ptr, ncols);

	cudaMemcpy(input_ptr, image.ptr(), ncols, cudaMemcpyHostToDevice);

	GPU_BGR2gray <<<numBlocks, blockSize >> >(input_ptr, result_ptr, ncols);

	//cudaDeviceSynchronize();
	cudaMemcpy(result.ptr(), result_ptr, ncols, cudaMemcpyDeviceToHost);

	double dtime = (omp_get_wtime() - time);
	cout << dtime << endl;
	imshow("GPU", result);


	// CPU~~~~~~~~~~

	time = omp_get_wtime();

	input_ptr = image.ptr<uchar>(0);
	result_ptr = result.ptr<uchar>(0);
	CPU_BGR2gray(input_ptr, result_ptr, ncols);

	dtime = (omp_get_wtime() - time);
	cout << dtime << endl;

	imshow("CPU", result);
	cudaFree(input_ptr);
	cudaFree(result_ptr);
	waitKey(0);                                         

}