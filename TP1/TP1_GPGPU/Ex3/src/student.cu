/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	/*__global__ void sepiaCUDA(const int n, const uchar *const dev_input, uchar *const dev_output)
	{
			// id global
		const int idThreadG = threadIdx.x // id du thread dans le block 
							+ blockIdx.x  // id du block dans la grid
							* blockDim.x;  // taille d'un block, nb threads dans blocks
		// nb threads global
		const int nbThreadsG = blockDim.x 
							* gridDim.x; // nb blocks dans grid

		for (int id = idThreadG; id < n; id += nbThreadsG)
		{
			int idInTab = id * 3;
			dev_output[idInTab] = uchar(fminf(255, (dev_input[id]*0.393 + dev_input[id+1]*0.769 + dev_input[id+2]*0.189)));
			dev_output[idInTab+1] = uchar(fminf(255, (dev_input[id]*0.349 + dev_input[id+1]*0.686 + dev_input[id+2]*0.168)));
			dev_output[idInTab+2] = uchar(fminf(255, (dev_input[id]*0.272 + dev_input[id+1]*0.534 + dev_input[id+2]*0.131)));
		}	
	}*/

	__global__ void sepiaCUDA(const int width, const int height, const uchar *const dev_input, uchar *const dev_output)
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

				dev_output[id] = (uchar)fminf(255.f, (dev_input[id]*0.393f + dev_input[id+1]*0.769f + dev_input[id+2]*0.189f));
				dev_output[id+1] = (uchar)(fminf(255.f, (dev_input[id]*0.349f + dev_input[id+1]*0.686f + dev_input[id+2]*0.168f)));
				dev_output[id+2] = (uchar)(fminf(255.f, (dev_input[id]*0.272f + dev_input[id+1]*0.534f + dev_input[id+2]*0.131f)));
			}
		}	
	}

	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar *dev_input = NULL;
		uchar *dev_output = NULL;
		
		/// TODOOOOOOOOOOOOOO
		std::cout 	<< "Allocating 2 arrays: ";
		chrGPU.start();
		const size_t bytes = input.size() * sizeof(uchar);
		
		cudaMalloc((void **) &dev_input, bytes);
		cudaMalloc((void **) &dev_output, bytes);

		chrGPU.stop();
		std::cout 	<< "Allocation -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		std::cout 	<< "Copying data to GPU : ";
		chrGPU.start();
		// Copy data from host to device (input arrays) 
		cudaMemcpy(dev_input, input.data(), bytes, cudaMemcpyHostToDevice);
		chrGPU.stop();
		std::cout 	<< "Copying -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// Launch kernel
		chrGPU.start();//dim3
		std::cout 	<< "Lauching the kernel";
		sepiaCUDA<<<dim3(16, 16), dim3(32, 32)>>>(width, height, dev_input, dev_output);
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
	}
}
