/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: main.cpp
* Author: Maxime MARIA
*/

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>     
#include <cstring>
#include <exception>

#include "student.hpp"
#include "chronoCPU.hpp"
#include "lodepng.h"

namespace IMAC
{
	// Print program usage
	void printUsageAndExit(const char *prg) 
	{
		std::cerr	<< "Usage: " << prg << std::endl
					<< " \t -f <F>: <F> image file name" << std::endl << std::endl;
		exit(EXIT_FAILURE);
	}

	// Computes grey of 'input' and stores result in 'output'
	void greyCPU(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		std::cout << "Process on CPU (sequential)"	<< std::endl;
		ChronoCPU chrCPU;
		chrCPU.start();
		for (uint i = 0; i < height; ++i) 
		{
			for (uint j = 0; j < width; ++j) 
			{
				const uint id = (i * width + j) * 3;
				const uchar inR = input[id];
				const uchar inG = input[id + 1];
				const uchar inB = input[id + 2];
				//float greyVal = std::min<float>(255.f, ( inR * .299f + inG * .587f + inB * .114f )));
				float greyVal = std::min<float>(255.f, ( inR * .299f + inG * .587f + inB * .114f ));
				output[id] = static_cast<uchar>( greyVal );
				output[id + 1] = static_cast<uchar>( greyVal );
				output[id + 2] = static_cast<uchar>( greyVal );
			}
		}
		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
	}

	// create the histogram of the computed image
	void histoCPU(const std::vector<uchar> &greyLvl, const uint width, const uint height, std::vector<int> &output)
	{
		std::cout << "Process histogram on CPU" << std::endl;

		for (uint i = 0; i < height; ++i) 
		{
			for (uint j = 0; j < width; ++j) 
			{
				uint id = greyLvl[i * width + j];
				output[id] += 1;
			}
		}
	}

	// print all the values in the histogram
	void printHisto(const std::vector<int> greyLvl)
	{
		for(int i = 0; i<256; ++i)
		{
			std::cout << "[" << i << "] = " << greyLvl[i] << std::endl;
		}
	}

	// verify if there is the same number of pixel in the image and in the histogram
	bool verifyHisto(const std::vector<int> greyLvl, const uint width, const uint height)
	{
		int nbPixel = width * height, sum = 0;
		for(int i = 0; i<256; ++i)
		{
			sum+= greyLvl[i];
		}

		if(sum == nbPixel)
			return true;
		else{
			std::cout << "there is a sushi ! nbPixel = " << nbPixel << " and sum = " << sum << std::endl;
			return false;
		}
	}


	// Compare two vectors
	bool compare(const std::vector<uchar> &a, const std::vector<uchar> &b)
	{
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			return false;
		}
		for (uint i = 0; i < a.size(); ++i)
		{
			// Floating precision can cause small difference between host and device
			if (std::abs(a[i] - b[i]) > 1)
			{
				std::cout << "Error at index " << i << ": a = " << uint(a[i]) << " - b = " << uint(b[i]) << std::endl;
				return false; 
			}
		}
		return true;
	}

	// Main function
	void main(int argc, char **argv) 
	{	
		char fileName[2048];

		// Parse command line
		if (argc == 1) 
		{
			std::cerr << "Please give a file..." << std::endl;
			printUsageAndExit(argv[0]);
		}

		for (int i = 1; i < argc; ++i) 
		{
			if (!strcmp(argv[i], "-f")) 
			{
				if (sscanf(argv[++i], "%s", fileName) != 1)
				{
					printUsageAndExit(argv[0]);
				}
			}
			else
			{
				printUsageAndExit(argv[0]);
			}
		}
		
		// Get input image
		std::vector<uchar> input;
		uint width;
		uint height;

		std::cout << "Loading " << fileName << std::endl;
		unsigned error = lodepng::decode(input, width, height, fileName, LCT_RGB);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		std::cout << "Image has " << width << " x " << height << " pixels (RGB)" << std::endl;

		// Create 2 output images
		std::vector<uchar> outputImageCPU(3 * width * height);
		std::vector<uchar> outputImageGPU(3 * width * height);

		// Prepare output file name
		const std::string fileNameStr(fileName);
		std::size_t lastPoint = fileNameStr.find_last_of(".");
		std::string ext = fileNameStr.substr(lastPoint);
		std::string name = fileNameStr.substr(0,lastPoint);
		std::string outputImageCPUName = name + "_GreyCPU" + ext;
		std::string outputImageGPUName = name + "_GreyGPU" + ext;

		// Create the outputs for the histogram
		std::vector<int> outputHistoCPU(255);
		std::vector<int> outputHistoGPU(255);

		// Fill the outputs histo
		for(int i=0; i<256; ++i)
		{
			outputHistoCPU[i] = 0;
			outputHistoGPU[i] = 0;
		}

		// Computation on CPU
		greyCPU(input, width, height, outputImageCPU);
		histoCPU(outputImageCPU, width, height, outputHistoCPU);
		printHisto(outputHistoCPU);
		verifyHisto(outputHistoCPU, width, height);

		std::cout << "Valeaur de output : " << outputImageCPU[200] << std::endl;
		
		std::cout << "Save image as: " << outputImageCPUName << std::endl;
		error = lodepng::encode(outputImageCPUName, outputImageCPU, width, height, LCT_RGB);
		if (error)
		{
			throw std::runtime_error("Error loadpng::encode: " + std::string(lodepng_error_text(error)));
		}
		
		std::cout 	<< "============================================"	<< std::endl
					<< "              STUDENT'S JOB !               "	<< std::endl
					<< "============================================"	<< std::endl;

		studentJob(input, width, height, outputImageGPU);

		std::cout << "Save image as: " << outputImageGPUName << std::endl;
		error = lodepng::encode(outputImageGPUName, outputImageGPU, width, height, LCT_RGB);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		
		std::cout << "============================================"	<< std::endl << std::endl;

		std::cout << "Checking result..." << std::endl;
		if (compare(outputImageCPU, outputImageGPU))
		{
			std::cout << " -> Well done!" << std::endl;
		}
		else
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
	}
}

int main(int argc, char **argv) 
{
	try
	{
		IMAC::main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}
