#include "chronoCPU.hpp"

#include <iostream>

Histo_CPU::Histo_CPU()  
{
	for(int i=0; i<size; ++i)
	{
		histo[i] = 0;
	}
}

Histo_CPU::calculate(const std::vector<uchar> &greyLvl, const uint width, const uint height)  
{
	std::cout << "Process histogram on CPU" << std::endl;

	for (uint i = 0; i < height; ++i) 
	{
		for (uint j = 0; j < width; ++j) 
		{
			uint id = greyLvl[i * width + j];
			histo[id] += 1;
		}
	}
}

Histo_CPU::print() 
{
	for(int i = 0; i<256; ++i)
	{
		std::cout << "[" << i << "] = " << greyLvl[i] << std::endl;
	}
}

Histo_CPU::checkRight()
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