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

	__global__ void rgbTOhsvCUDA(const uchar *const dev_input, const int width, const int height, 
		float *const dev_outputH, float *const dev_outputS, float *const dev_outputV)
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

		double min, max, delta;

		for (int idY = idThreadGY; idY < height; idY += nbThreadsGY)
		{
			for(int idX = idThreadGX; idX < width; idX += nbThreadsGX){

				const uint idInRGB = (idY * width + idX) * 3;
				const uint idInHSV = (idY * width + idX);

				const uchar inR = dev_input[idInRGB];
				const uchar inG = dev_input[idInRGB + 1];
				const uchar inB = dev_input[idInRGB + 2];

				min = inR < inG ? inR : inG;
		    	min = min  < inB ? min : inB;

		    	max = inR > inG ? inR : inG;
		    	max = max  > inB ? max  : inB;

		    	dev_outputV[idInHSV] = max;

		    	delta = max - min;
		    	if (delta < 0.00001)
			    {
			        dev_outputS[idInHSV] = 0;
			        dev_outputH[idInHSV] = 0;
			        continue;
			    }
			    if (max > 0.0)
			    {
			    	dev_outputS[idInHSV] = (delta/ max);
			    } else {
			    	dev_outputS[idInHSV] = 0.0;
			        dev_outputH[idInHSV] = NAN;                           // its now undefined
			        continue;
			    }

				if( inR >= max )                           // > is bogus, just keeps compilor happy
			        dev_outputH[idInHSV] = ( inG - inB ) / delta;        // between yellow & magenta
			    else if( inG >= max )
			        dev_outputH[idInHSV] = 2.0 + ( inB - inR ) / delta;  // between cyan & yellow
			    else
			        dev_outputH[idInHSV] = 4.0 + ( inR - inG ) / delta;  // between magenta & cyan

			    dev_outputH[idInHSV] *= 60.0;                              // degrees

			    if( dev_outputH[idInHSV] < 0.0 )
			        dev_outputH[idInHSV] += 360.0;
			}
		}	
	}

	__global__ void hsvTOrgbCUDA(const float *const dev_inputH, const float *const dev_inputS, const float *const dev_inputV,
		const int width, const int height, 
		uchar *const dev_outputRGB)
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

		double hh, p, q, t, ff;
	    long k;

		for (int idY = idThreadGY; idY < height; idY += nbThreadsGY)
		{
			for(int idX = idThreadGX; idX < width; idX += nbThreadsGX){
				const uint idInRGB = (idY * width + idX) * 3;
				const uint idInHSV = (idY * width + idX);

				const uint idOutR = idInRGB;
				const uint idOutG = idInRGB + 1;
				const uint idOutB = idInRGB + 2;

				hh = dev_inputH[idInHSV];

			    if(hh >= 360.0) hh = 0.0;
			    hh /= 60.0;
			    k = (long)hh;
			    ff = hh - k;
			    p = dev_inputV[idInHSV] * (1.0 - dev_inputS[idInHSV]);
			    q = dev_inputV[idInHSV] * (1.0 - (dev_inputS[idInHSV] * ff));
			    t = dev_inputV[idInHSV] * (1.0 - (dev_inputS[idInHSV] * (1.0 - ff)));

			    switch(k) {
			    case 0:
			        dev_outputRGB[idOutR] = dev_inputV[idInHSV];
			        dev_outputRGB[idOutG] = t;
			        dev_outputRGB[idOutB] = p;
			        break;
			    case 1:
			        dev_outputRGB[idOutR] = q;
			        dev_outputRGB[idOutG] = dev_inputV[idInHSV];
			        dev_outputRGB[idOutB] = p;
			        break;
			    case 2:
			        dev_outputRGB[idOutR] = p;
			        dev_outputRGB[idOutG] = dev_inputV[idInHSV];
			        dev_outputRGB[idOutB] = t;
			        break;
			    case 3:
			        dev_outputRGB[idOutR] = p;
			        dev_outputRGB[idOutG] = q;
			        dev_outputRGB[idOutB] = dev_inputV[idInHSV];
			        break;
			    case 4:
			        dev_outputRGB[idOutR] = t;
			        dev_outputRGB[idOutG] = p;
			        dev_outputRGB[idOutB] = dev_inputV[idInHSV];
			        break;
			    case 5:
			    default:
			        dev_outputRGB[idOutR] = dev_inputV[idInHSV];
			        dev_outputRGB[idOutG] = p;
			        dev_outputRGB[idOutB] = q;
			        break;
			    }
			}
		}
	}

	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		uchar *dev_input = NULL;
		float *dev_outputH = NULL;
		float *dev_outputS = NULL;
		float *dev_outputV = NULL;
		uchar *dev_output = NULL;
		uchar *dev_histo = NULL;
		
		std::cout 	<< "Allocating 2 arrays: ";
		chrGPU.start();
		const size_t bytes = input.size() * sizeof(uchar);
		const size_t HSVbytes = input.size() * sizeof(float);
		
		cudaMalloc((void **) &dev_input, bytes);
		cudaMalloc((void **) &dev_outputH, HSVbytes);
		cudaMalloc((void **) &dev_outputS, HSVbytes);
		cudaMalloc((void **) &dev_outputV, HSVbytes);
		cudaMalloc((void **) &dev_output, bytes);
		cudaMalloc((void **) &dev_histo, 255 * sizeof(uchar));

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
		rgbTOhsvCUDA<<<dim3(16, 16), dim3(32, 32)>>>(dev_input, width, height, dev_outputH, dev_outputS, dev_outputV);
		hsvTOrgbCUDA<<<dim3(16, 16), dim3(32, 32)>>>(dev_outputH, dev_outputS, dev_outputV, width, height, dev_output);
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
		cudaFree(dev_outputH);
		cudaFree(dev_outputS);
		cudaFree(dev_outputV);
		cudaFree(dev_output);
		cudaFree(dev_histo);
	}
}
