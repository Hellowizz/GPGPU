#ifndef __HISTO_CPU_HPP
#define __HISTO_CPU_HPP

class Histo_CPU {
private:
	const uint size = 255;
	std::vector<int> histo;

public:
	Histo_CPU();
	~Histo_CPU();

	void	calculate();
	void	print();
	bool	checkRight();