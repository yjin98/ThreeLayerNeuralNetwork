#include <iostream>
#include <vector>
#include "NN.h"

int main()
{
    int xd = 8;
    std::vector<std::vector<double>> x, y, predy;
    std::vector<double> x1, y1;
    for (int i = 0; i < 50; i++) {
        for (int j = 0; j < xd; j++) {
            x1.push_back((rand() % 1000) / 100.0 - 5);
        }
        //for (int j = 0; j < 4; j++) {
        //    y1.push_back((rand() % 1000) / 100.0 - 5);
        //}
        y1.push_back(x1[0] + x1[2] * x1[4]);
        y1.push_back(x1[1] * x1[1]);
        y1.push_back(x1[3] + x1[7]);
        y1.push_back(x1[5] + x1[6] + x1[6]);
        x.push_back(x1);
        y.push_back(y1);
        x1.clear();
        y1.clear();
    }

    NN reg;
    reg.x = x;
    reg.xdim = 8;
    reg.y = y;
    reg.ydim = 4;
    reg.numxy = 50;
    reg.initialnn(xd, 16, "leakyrelu", 8, "leakyrelu", 4, "id");
    reg.lossfunc = "MSE";
    reg.epoch = 10000;
    reg.train();

    predy = reg.py;
    for (int i = 0; i < 20; i++) {
        std::cout << "Y" << i << ": (";
        for (int j = 0; j < 4; j++) {
            std::cout << y[i][j] << ", ";
        }
        std::cout << ")\n" << "PY" << i << ": (";
        for (int j = 0; j < 4; j++) {
            std::cout << predy[i][j] << ", ";
        }
        std::cout << ")\n\n";
    }
}

