#pragma once
#include <iostream>
#include <vector>
#include <string>

class NN
{
public:
	int xdim, ydim, numxy, l1dim, l2dim, l3dim, epoch;
	std::string act1, act2, act3, lossfunc;
	std::vector<std::vector<double>> x, tipt, tl1, tl1a, tl2, tl2a, tl3, py, y, Dloss_l;
	std::vector<double> ipt, l1, l1a, l2, l2a, l3, l3a;
	std::vector<std::vector<double>> w[3], wo[3];
	double loss, learnrate1=1e-5, learnrate2=1e-5, learnrate3=1e-5, learnrate = 9e-1;
	
	std::vector<double> initial(int n);
	void initialw(int wn, int m, int n);
	void initialnn(int xd, int dim1, std::string a1, int dim2, std::string a2, int dim3, std::string a3);

	double relu(double u);
	double Drelu(double u);
	double leakyrelu(double u);
	double Dleakyrelu(double u);
	double sigmoid(double u);
	double Dsigmoid(double u);

	void tol1();
	void tol2();
	void tol3();

	double lossMSEr();
	double DlossMSEr_py(int i, int j);
	double lossBCEn();
	double DlossBCEn_py(int i, int j);

	void calcloss();
	void backl3();
	void backl2();
	void backl1();

	void traincyc();
	void train();
};

