/*
* TP 1 - Premiers pas en CUDA
* --------------------------
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{

	__global__ void greyCUDA(const int width, const int height, 
		const uchar *const dev_input, uchar *const dev_output)
	{
			// id global en x
		const int idThreadGX = threadIdx.x // id du thread dans le block 
							+ blockIdx.x  // id du block dans la grid
							* blockDim.x;  // taille d'un block, nb threads dans blocks
			// nb threads global en x
		const int nbThreadsGX = blockDim.x 
							* gridDim.x; // nb blocks dans grid

							// id global en y
		const int idThreadGY = threadIdx.y // id du thread dans le block 
							+ blockIdx.y  // id du block dans la grid
							* blockDim.y;  // taille d'un block, nb threads dans blocks
			// nb threads global en y
		const int nbThreadsGY = blockDim.y 
							* gridDim.y; // nb blocks dans grid

		for (int idY = idThreadGY; idY < height; idY += nbThreadsGY)
		{
			for(int idX = idThreadGX; idX < width; idX += nbThreadsGX){
				int id = (idY * width + idX) * 3;

				uchar greyVal = fminf(255.f, 0.299f * dev_input[id] + 0.587f * dev_input[id+1] + 0.114f * dev_input[id+2]);
				dev_output[id] = greyVal;
				dev_output[id+1] = greyVal;
				dev_output[id+2] = greyVal;
			}
		}	
	}

	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		uchar *dev_input = NULL;
		uchar *dev_output = NULL;
		uchar *dev_greyLvl = NULL;
		
		/// TODOOOOOOOOOOOOOO
		std::cout 	<< "Allocating 2 arrays: ";
		chrGPU.start();
		const size_t bytes = input.size() * sizeof(uchar);
		
		cudaMalloc((void **) &dev_input, bytes);
		cudaMalloc((void **) &dev_output, bytes);
		cudaMalloc((void **) &dev_greyLvl, 255 * sizeof(uchar));

		chrGPU.stop();
		std::cout 	<< "Allocation -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		std::cout 	<< "Copying data to GPU : ";
		chrGPU.start();
		// Copy data from host to device (input arrays) 
		cudaMemcpy(dev_input, input.data(), bytes, cudaMemcpyHostToDevice);
		chrGPU.stop();
		std::cout 	<< "Copying -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// Launch the kernel for the greylvl image
		chrGPU.start();//dim3
		std::cout 	<< "Lauching the kernel";
		greyCUDA<<<dim3(16, 16), dim3(32, 32)>>>(width, height, dev_input, dev_output);
		chrGPU.stop();
		std::cout 	<< "Calculations -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		std::cout 	<< "Copying data to CPU : ";
		chrGPU.start();
		// Copy data from device to host (output array)  
		cudaMemcpy(output.data(), dev_output, bytes, cudaMemcpyDeviceToHost);
		chrGPU.stop();
		std::cout 	<< "Copying -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// Free arrays on device
		cudaFree(dev_input);
		cudaFree(dev_output);
		cudaFree(dev_greyLvl);
	}
}
