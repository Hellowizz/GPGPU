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

	void rgbTOhsvCPU(const std::vector<uchar> &input, const uint width, const uint height, std::vector<float> &dev_outputH, 
		std::vector<float> &dev_outputS, std::vector<float> &dev_outputV)
	{
		std::cout << "Process rgbTohsv on CPU (sequential)"	<< std::endl;
		ChronoCPU chrCPU;
		chrCPU.start();

		double min, max, delta;

		for (uint i = 0; i < height; ++i) 
		{
			for (uint j = 0; j < width; ++j) 
			{
				const uint idInRGB = (i * width + j) * 3;
				const uint idInHSV = (i * width + j);

				const uchar inR = input[idInRGB];
				const uchar inG = input[idInRGB + 1];
				const uchar inB = input[idInRGB + 2];

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
		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
	}

	void hsvTOrgbCPU(const std::vector<float> &input_h, const std::vector<float> &input_s, const std::vector<float> &input_v, 
		const uint width, const uint height,
		std::vector<uchar> &outputRGB)
	{
		std::cout << "Process hsvTorgb on CPU (sequential)"	<< std::endl;
		ChronoCPU chrCPU;
		chrCPU.start();

		double hh, p, q, t, ff;
	    long k;

		for (uint i = 0; i < height; ++i) 
		{
			for (uint j = 0; j < width; ++j) 
			{
				const uint idInRGB = (i * width + j) * 3;
				const uint idInHSV = (i * width + j);

				const uint idOutR = idInRGB;
				const uint idOutG = idInRGB + 1;
				const uint idOutB = idInRGB + 2;

				hh = input_h[idInHSV];

			    if(hh >= 360.0) hh = 0.0;
			    hh /= 60.0;
			    k = (long)hh;
			    ff = hh - k;
			    p = input_v[idInHSV] * (1.0 - input_s[idInHSV]);
			    q = input_v[idInHSV] * (1.0 - (input_s[idInHSV] * ff));
			    t = input_v[idInHSV] * (1.0 - (input_s[idInHSV] * (1.0 - ff)));

			    switch(k) {
			    case 0:
			        outputRGB[idOutR] = input_v[idInHSV];
			        outputRGB[idOutG] = t;
			        outputRGB[idOutB] = p;
			        break;
			    case 1:
			        outputRGB[idOutR] = q;
			        outputRGB[idOutG] = input_v[idInHSV];
			        outputRGB[idOutB] = p;
			        break;
			    case 2:
			        outputRGB[idOutR] = p;
			        outputRGB[idOutG] = input_v[idInHSV];
			        outputRGB[idOutB] = t;
			        break;
			    case 3:
			        outputRGB[idOutR] = p;
			        outputRGB[idOutG] = q;
			        outputRGB[idOutB] = input_v[idInHSV];
			        break;
			    case 4:
			        outputRGB[idOutR] = t;
			        outputRGB[idOutG] = p;
			        outputRGB[idOutB] = input_v[idInHSV];
			        break;
			    case 5:
			    default:
			        outputRGB[idOutR] = input_v[idInHSV];
			        outputRGB[idOutG] = p;
			        outputRGB[idOutB] = q;
			        break;
			    }
			}
		}
		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
	}

	// create the histogram of the computed image
	void fillHistoCPU(const std::vector<float> &valueLvl, const uint width, const uint height, std::vector<int> &histogramOutput)
	{
		std::cout << "Process histogram on CPU" << std::endl;
		ChronoCPU chrCPU;
		chrCPU.start();

		for (uint i = 0; i < height; ++i) 
		{
			for (uint j = 0; j < width; ++j) 
			{
				uint id = valueLvl[i * width + j];
				if(valueLvl[i * width + j] > 255)
					std::cout 	<< " problem : " <<  valueLvl[i * width + j]  << std::endl;
				histogramOutput[id] += 1;
			}
		}

		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
	}

	// print all the values in the histogram
	void printHisto(const std::vector<int> values)
	{
		for(int i = 0; i < values.size(); ++i)
		{
			std::cout << "[" << i << "] = " << values[i] << std::endl;
		}
	}

	// verify if there is the same number of pixel in the image and in the histogram
	bool verifyHisto(const std::vector<int> values, const uint width, const uint height)
	{
		int nbPixel = width * height, sum = 0;
		for(int i = 0; i < values.size(); ++i)
		{	
			sum+= values[i];
		}

		if(sum == nbPixel)
			return true;
		else{
			std::cout << "there is a sushi ! nbPixel = " << nbPixel << " and sum = " << sum << std::endl;
			return false;
		}
	}

	// fill the repartition function tab from the histogram
	void fillRepartCPU(const std::vector<int> &histogram, std::vector<int> &outputRepart)
	{
		int acc = 0;
		for(int i=0; i<histogram.size(); ++i)
		{
			acc += histogram[i];
			outputRepart[i] = acc;
		}
	}

	// equalize the hue part of the HSV image
	void equalize(std::vector<float> &values, const std::vector<int> &repart)
	{
		for(int i = 1 ; i < values.size() ; ++i){
            values[i] = uchar(repart[values[i]] / double(values.size()) * 255);
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

	// Compare two vectors
	bool compareInt(const std::vector<int> &a, const std::vector<uint> &b)
	{
		for (uint i = 0; i < a.size(); ++i)
		{
			// Floating precision can cause small difference between host and device
			if (std::abs(double(a[i] - b[i])) > 1)
			{
				std::cout << "Error at index " << i << ": a = " << 

				a[i] << " - b = " << b[i] << std::endl;
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

		std::cout 	<< "============================================"	<< std::endl
					<< "              CPU'S JOB !               "	<< std::endl
					<< "============================================"	<< std::endl;

		// Create 2 output images
		std::vector<uchar> outputImageCPU(3 * width * height);
		std::vector<uchar> outputImageGPU(3 * width * height);

		// Create 3 output for HSV
		std::vector<float> hue(width * height);
		std::vector<float> saturation(width * height);
		std::vector<float> value(width * height);

		// Prepare output file name
		const std::string fileNameStr(fileName);
		std::size_t lastPoint = fileNameStr.find_last_of(".");
		std::string ext = fileNameStr.substr(lastPoint);
		std::string name = fileNameStr.substr(0,lastPoint);
		std::string outputImageCPUName = name + "_equalizeCPU" + ext;
		std::string outputImageGPUName = name + "_equalizeGPU" + ext;

		// Create the histogram
		int histo_size = 256;
		std::vector<int> histoCPU(histo_size);
		std::vector<int> histoGPU(histo_size);

		// Create the repartition
		std::vector<int> repartCPU(histo_size);
		// test
		std::vector<uint> repartGPU(3 * width * height);

		// Fill the histogram and repartitions
		for(int i=0; i<histo_size; ++i)
		{
			histoGPU[i] = 0;
			repartCPU[i] = 0;
		}

		// computation on CPU HSV
		rgbTOhsvCPU(input, width, height, hue, saturation, value);
		fillHistoCPU(value, width, height, histoCPU);
		//printHisto(outputHistoCPU);
		verifyHisto(histoCPU, width, height);
		fillRepartCPU(histoCPU, repartCPU);
		equalize(value, repartCPU);
		hsvTOrgbCPU(hue, saturation, value, width, height, outputImageCPU);		
		
		std::cout << "Save image as: " << outputImageCPUName << std::endl;
		error = lodepng::encode(outputImageCPUName, outputImageCPU, width, height, LCT_RGB);
		if (error)
		{
			throw std::runtime_error("Error loadpng::encode: " + std::string(lodepng_error_text(error)));
		}
		
		std::cout 	<< "============================================"	<< std::endl
					<< "              GPU'S JOB !               "	<< std::endl
					<< "============================================"	<< std::endl;

		studentJob(input, width, height, outputImageGPU, repartGPU);

		std::cout << "Save image as: " << outputImageGPUName << std::endl;
		error = lodepng::encode(outputImageGPUName, outputImageGPU, width, height, LCT_RGB);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		
		std::cout << "============================================"	<< std::endl << std::endl;

		std::cout << "Checking result..." << std::endl;
		if (compareInt(repartCPU, repartGPU))
		{
			std::cout << " Repart -> Well done!" << std::endl;
		}
		else
		{
			std::cout << " Repart -> You failed, retry!" << std::endl;
		}

		if (compare(outputImageCPU, outputImageGPU))
		{
			std::cout << " Image -> Well done!" << std::endl;
		}
		else
		{
			std::cout << " Image -> You failed, retry!" << std::endl;
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
