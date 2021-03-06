/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{

// ================================================== For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar4 &c)
	{
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";  
    	return os; 
	}

	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b)
	{
		bool error = false;
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			error = true;
		}
		else
		{
			for (uint i = 0; i < a.size(); ++i)
			{
				// Floating precision can cause small difference between host and device
				if (	std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 
					|| std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2)
				{
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true; 
					break;
				}
			}
		}
		if (error)
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
		else
		{
			std::cout << " -> Well done!" << std::endl;
		}
	}
// ==================================================


	__constant__ float dev_matConv[225];
	texture<uchar4, 2, cudaReadModeElementType> texRef;


	__device__ float clampfGPU(const float val, const float min , const float max) 
	{
		return fminf(max, fmaxf(min, val));
	}

	__global__ void convGPU(const uchar4 *input, const uint imgWidth, const uint imgHeight,
					const uint matSize, uchar4 *output){

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

		for (int idY = idThreadGY; idY < imgHeight; idY += nbThreadsGY)
		{
			for(int idX = idThreadGX; idX < imgWidth; idX += nbThreadsGX){

				float3 sum = make_float3(0.f,0.f,0.f);
				
				// Apply convolution
				for ( uint j = 0; j < matSize; ++j ) 
				{
					for ( uint i = 0; i < matSize; ++i ) 
					{
						int dX = idX + i - matSize / 2;
						int dY = idY + j - matSize / 2;

						// Handle borders 
						if ( dX < 0 ) 
							dX = 0;

						if ( dX >= imgWidth ) 
							dX = imgWidth - 1;

						if ( dY < 0 ) 
							dY = 0;

						if ( dY >= imgHeight ) 
							dY = imgHeight - 1;

						const int idMat		= j * matSize + i;
						uchar4 value = tex2D(texRef, dX, dY); 

						sum.x += (float)value.x * dev_matConv[idMat];
						sum.y += (float)value.y * dev_matConv[idMat];
						sum.z += (float)value.z * dev_matConv[idMat];

						/*sum.x += (float)input[idPixel].x * matConv[idMat];
						sum.y += (float)input[idPixel].y * matConv[idMat];
						sum.z += (float)input[idPixel].z * matConv[idMat];*/
					}
				}
				const int idOut = idY * imgWidth + idX;
				output[idOut].x = (uchar)clampfGPU( sum.x, 0.f, 255.f );
				output[idOut].y = (uchar)clampfGPU( sum.y, 0.f, 255.f );
				output[idOut].z = (uchar)clampfGPU( sum.z, 0.f, 255.f );
				output[idOut].w = 255;
			}
		}	
	}

	void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar4 *dev_inputImg = NULL;
		uchar4 *dev_output = NULL;

		std::cout 	<< "Allocating 3 arrays: ";
		chrGPU.start();
		const size_t bytesImg = inputImg.size() * sizeof(uchar4);
		const size_t bytesMat = matConv.size() * sizeof(float);
		size_t pitchImg = 0;

		HANDLE_ERROR(cudaMallocPitch((void**) &dev_inputImg, &pitchImg, imgWidth * sizeof(uchar4), imgHeight));
		HANDLE_ERROR(cudaMalloc((void **) &dev_output, bytesImg));

		chrGPU.stop();
		std::cout 	<< "Allocation -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;
	
		std::cout 	<< "Copying data to GPU : ";
		chrGPU.start(); 

		// Copy data from host to device (input arrays) 
		
		HANDLE_ERROR(cudaMemcpy2D(dev_inputImg, pitchImg, inputImg.data(),
									imgWidth * sizeof(uchar4), imgWidth * sizeof(uchar4), 
									imgHeight, cudaMemcpyHostToDevice));
		
		HANDLE_ERROR(cudaMemcpyToSymbol(dev_matConv, matConv.data(), bytesMat));
		chrGPU.stop();
		std::cout 	<< "Copying -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// Texture
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

		HANDLE_ERROR(cudaBindTexture2D(NULL, texRef, dev_inputImg, cudaCreateChannelDesc<uchar4>(), imgWidth, imgHeight, pitchImg));

		// Launch kernel
		chrGPU.start();//dim3
		std::cout 	<< "Lauching the kernel";
		convGPU<<<dim3(16, 16), dim3(32, 32)>>>(dev_inputImg, imgWidth, imgHeight,
															matSize, dev_output);
		chrGPU.stop();
		std::cout 	<< "Calculations -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		std::cout 	<< "Copying data to CPU : ";
		chrGPU.start();
		// Copy data from device to host (output array)  
		HANDLE_ERROR (cudaMemcpy(output.data(), dev_output, bytesImg, cudaMemcpyDeviceToHost));
		chrGPU.stop();
		std::cout 	<< "Copying -> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		compareImages(resultCPU, output);

		// Free arrays on device
		HANDLE_ERROR(cudaUnbindTexture(texRef));
		HANDLE_ERROR(cudaFree(dev_inputImg));
		HANDLE_ERROR(cudaFree(dev_output));

	}
}
