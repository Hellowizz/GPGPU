/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"

namespace IMAC
{
	// ==================================================== Ex 0
    __global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ int shared[];

		const int idThreadG = threadIdx.x // id du thread dans le block 
							+ blockIdx.x  // id du block dans la grid
							* blockDim.x;  // taille d'un block, nb threads dans blocks
		
		const uint tid = threadIdx.x;

		if(tid >= size)
			shared[tid] = 0;
		else
			shared[tid] = dev_array[idThreadG];
		
		__syncthreads();

		for (int dec = 2; dec < blockDim.x; dec *= 2)
		{
			if(tid*dec + dec/2 < blockDim.x)
			{
				shared[tid*dec] = umax(shared[tid*dec], shared[tid*dec + dec/2]);
			}	
		}

		__syncthreads();
		
		if (threadIdx.x == 0)
			dev_partialMax[blockIdx.x] = shared[0];
	}

	void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */)
    {

    	// 2 arrays for GPU
		uint *dev_inputArray = NULL;

        const size_t bytes = array.size() * sizeof(uint);

		// Allocate array on GPU
		HANDLE_ERROR( cudaMalloc( (void**)&dev_inputArray, bytes ) );

		// Copy data from host to device
		HANDLE_ERROR( cudaMemcpy( dev_inputArray, array.data(), bytes, cudaMemcpyHostToDevice ) );

		std::cout << "========== Ex 1 " << std::endl;
		uint res1 = 0; // result
		// Launch reduction and get timing
		float2 timing1 = reduce<KERNEL_EX1>(dev_inputArray, array.size(), res1);

        printTiming(timing1);
		compare(res1, resCPU); // Compare results

		
		std::cout << "========== Ex 2 " << std::endl;
		/// TODO

		std::cout << "========== Ex 3 " << std::endl;
		/// TODO
		
		std::cout << "========== Ex 4 " << std::endl;
		/// TODO
		
		std::cout << "========== Ex 5 " << std::endl;
		/// TODO
		

		// Free array on GPU
		cudaFree( dev_inputArray );
    }

	void printTiming(const float2 timing)
	{
		std::cout << ( timing.x < 1.f ? 1e3f * timing.x : timing.x ) << " us on device and ";
		std::cout << ( timing.y < 1.f ? 1e3f * timing.y : timing.y ) << " us on host." << std::endl;
	}

    void compare(const uint resGPU, const uint resCPU)
	{
		if (resGPU == resCPU)
		{
			std::cout << "Well done ! " << resGPU << " == " << resCPU << " !!!" << std::endl;
		}
		else
		{
			std::cout << "You failed ! " << resGPU << " != " << resCPU << " !!!" << std::endl;
		}
	}
}
