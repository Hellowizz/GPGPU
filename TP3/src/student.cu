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
	// ==================================================== Ex 1
    __global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint shared[];

		const int idThreadG = threadIdx.x // id du thread dans le block 
							+ blockIdx.x  // id du block dans la grid
							* blockDim.x;  // taille d'un block, nb threads dans blocks
		
		const uint tid = threadIdx.x;

		if(idThreadG >= size)
			shared[tid] = 0;
		else 
			shared[tid] = dev_array[idThreadG];
		
		__syncthreads();

		for (int dec = 1; dec < blockDim.x; dec *= 2)
		{
			const uint id = 2 * dec * tid;
			if(id < blockDim.x)
			{
				shared[id] = umax(shared[id], shared[id + dec]);
			}	
			__syncthreads();
		}
		
		if (threadIdx.x == 0)
			dev_partialMax[blockIdx.x] = shared[0];
	}

	// ==================================================== Ex 2
    __global__
    void maxReduce_ex2(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint shared[];

		const int idThreadG = threadIdx.x // id du thread dans le block 
							+ blockIdx.x  // id du block dans la grid
							* blockDim.x;  // taille d'un block, nb threads dans blocks
		
		const uint tid = threadIdx.x;

		if(idThreadG >= size)
			shared[tid] = 0;
		else
			shared[tid] = dev_array[idThreadG];
		
		__syncthreads();

		for (int dec = blockDim.x/2; dec >= 1 ; dec /= 2)
		{
			if(tid < dec)
			{
				shared[tid] = umax(shared[tid], shared[dec + tid]);
			}
			__syncthreads();
		}		
		
		if (threadIdx.x == 0)
			dev_partialMax[blockIdx.x] = shared[0];
	}

	// ==================================================== Ex 2
    __global__
    void maxReduce_ex3(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint shared[];

		const int idThreadG = threadIdx.x // id du thread dans le block 
							+ blockIdx.x  // id du block dans la grid
							* blockDim.x * 2;  // taille d'un block, nb threads dans blocks
		
		const uint tid = threadIdx.x;

		if(idThreadG >= size)
			shared[tid] = 0;
		else if(blockDim.x + idThreadG < size)
			shared[tid] = umax(dev_array[idThreadG], dev_array[blockDim.x + idThreadG]);
		else if(idThreadG < size)
			shared[tid] = dev_array[idThreadG];
		
		__syncthreads();

		for (int dec = blockDim.x/2; dec >= 1 ; dec /= 2)
		{
			if(tid < dec)
			{
				shared[tid] = umax(shared[tid], shared[dec + tid]);
			}
			__syncthreads();
		}
		
		if (tid == 0)
			dev_partialMax[blockIdx.x] = shared[0];
	}

	// ==================================================== Ex 3
    __global__
    void maxReduce_ex4(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint shared[];

		const int idThreadG = threadIdx.x // id du thread dans le block 
							+ blockIdx.x  // id du block dans la grid
							* blockDim.x * 2;  // taille d'un block, nb threads dans blocks
		
		const uint tid = threadIdx.x;

		if(idThreadG >= size)
			shared[tid] = 0;
		else if(blockDim.x + idThreadG < size)
			shared[tid] =  umax(dev_array[idThreadG], dev_array[blockDim.x + idThreadG]);
		else if(idThreadG < size)
			shared[tid] = dev_array[idThreadG];

		__syncthreads();

		for (int dec = blockDim.x/2; dec > 32 ; dec /= 2)
		{
			if(tid < dec)
			{
				shared[tid] = umax(shared[tid], shared[dec + tid]);
			}
			__syncthreads();
		}

		if(tid < 32) {
			volatile uint * sharedVolatile = shared;
			sharedVolatile[tid] = umax(sharedVolatile[tid], sharedVolatile[32 + tid]);
			sharedVolatile[tid] = umax(sharedVolatile[tid], sharedVolatile[16 + tid]);
			sharedVolatile[tid] = umax(sharedVolatile[tid], sharedVolatile[8 + tid]);
			sharedVolatile[tid] = umax(sharedVolatile[tid], sharedVolatile[4 + tid]);
			sharedVolatile[tid] = umax(sharedVolatile[tid], sharedVolatile[2 + tid]);
			sharedVolatile[tid] = umax(sharedVolatile[tid], sharedVolatile[1 + tid]);
		}
		
		if (tid == 0)
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
		uint res2 = 0; // result
		// Launch reduction and get timing
		float2 timing2 = reduce<KERNEL_EX2>(dev_inputArray, array.size(), res2);

        printTiming(timing2);
		compare(res2, resCPU); // Compare results
		/// TODO

		std::cout << "========== Ex 3 " << std::endl;
		/// TODO
		uint res3 = 0; // result
		// Launch reduction and get timing
		float2 timing3 = reduce<KERNEL_EX3>(dev_inputArray, array.size(), res3);

        printTiming(timing3);
		compare(res3, resCPU); // Compare results
		
		std::cout << "========== Ex 4 " << std::endl;
		/// TODO
		uint res4 = 0; // result
		// Launch reduction and get timing
		float2 timing4 = reduce<KERNEL_EX4>(dev_inputArray, array.size(), res4);

        printTiming(timing4);
		compare(res4, resCPU); // Compare results
		
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
