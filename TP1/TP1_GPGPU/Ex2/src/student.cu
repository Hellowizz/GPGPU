/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	__global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	{
		// id global
		const int idThreadG = threadIdx.x // id du thread dans le block 
							+ blockIdx.x  // id du block dans la grid
							* blockDim.x;  // taille d'un block, nb threads dans blocks
		// nb threads global
		const int nbThreadsG = blockDim.x 
							* gridDim.x; // nb blocks dans grid

		for (int id = idThreadG; i < n; i += nbThreadsG)
		{
			dev_res[id] = dev_a[id] + dev_b[id];
		}
/*		int idBlock = blockIdx.x;
		int idThread = threadIdx.x;
		int nbBlocks = gridDim.x;
		int nbThreads = blockDim.x;

		if(n > nbBlocks * nbThreads)
			for(int i=0; i < n/nbBlocks; i++){
				int id = i * (nbBlocks * nbThreads) + idBlock * nbThreads + idThread;
				dev_res[id] = dev_a[id] + dev_b[id];
			}*/
	}

    void studentJob(const int size, const int *const a, const int *const b, int *const res)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		int *dev_a = NULL;
		int *dev_b = NULL;
		int *dev_res = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int);
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		
		/// TODO
		cudaMalloc((void **) &dev_a, bytes);
		cudaMalloc((void **) &dev_b, bytes);
		cudaMalloc((void **) &dev_res, bytes);
		
		chrGPU.stop();
		std::cout 	<< "Allocation -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);

		chrGPU.start();
		// Launch kernel
		int nbThreads = 100;
		sumArraysCUDA<<<2, nbThreads>>>(size, dev_a, dev_b, dev_res);
		chrGPU.stop();
		std::cout 	<< "Calculations -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from device to host (output array)  
		cudaMemcpy(res, dev_res, bytes, cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_res);
	}
}

