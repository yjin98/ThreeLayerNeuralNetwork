#include "NN.h"


std::vector<double> NN::initial(int n) {
    std::vector<double> w;
    double v;
    for (int i = 0; i < n + 1; i++) {
        v = (rand() % 500) / (1000.0 * (n + 1));
        w.push_back(v);
    }
    return w;
}

void NN::initialw(int wn, int m, int n) {
    for (int i = 0; i < m; i++) {
        w[wn - 1].push_back(initial(n));
    }
}

void NN::initialnn(int xd, int dim1, std::string a1, int dim2, std::string a2, int dim3, std::string a3) {
    xdim = xd;
    l1dim = dim1;
    act1 = a1;
    l2dim = dim2;
    act2 = a2;
    l3dim = dim3;
    act3 = a3;
    initialw(1, l1dim, xdim);
    initialw(2, l2dim, l1dim);
    initialw(3, l3dim, l2dim);
}

//.................................................................................

double NN::relu(double u) {
    if (u < 0) {
        return 0;
    }
    else {
        return u;
    }
}

double NN::Drelu(double u) {
    if (u < 0) {
        return 0;
    }
    else {
        return 1;
    }
}

double NN::leakyrelu(double u) {
    if (u <= 0) {
        return 0.01 * u;
    }
    else {
        return u;
    }
}

double NN::Dleakyrelu(double u) {
    if (u <= 0) {
        return 0.01;
    }
    else {
        return 1;
    }
}

double NN::sigmoid(double u) {
    double A = exp(u);
    double B = 1 + A;
    return A / B;
}

double NN::Dsigmoid(double u) {
    double A = exp(u);
    double B = (A + 1) * (A + 1);
    return A / B;
}

//.................................................................................

void NN::tol1() {
    l1.clear();
    l1a.clear();
    ipt.push_back(1);
    double nod;
    if (act1 == "leakyrelu") {
        for (int i = 0; i < l1dim; i++) {
            nod = 0;
            for (int j = 0; j < xdim + 1; j++) {
                nod = nod + ipt[j] * w[0][i][j];
            }
            l1.push_back(nod);
            l1a.push_back(leakyrelu(nod));
        }
    }
    else if (act1 == "relu") {
        for (int i = 0; i < l1dim; i++) {
            nod = 0;
            for (int j = 0; j < xdim + 1; j++) {
                nod = nod + ipt[j] * w[0][i][j];
            }
            l1.push_back(nod);
            l1a.push_back(relu(nod));
        }
    }
    else if (act1 == "sigmoid") {
        for (int i = 0; i < l1dim; i++) {
            nod = 0;
            for (int j = 0; j < xdim + 1; j++) {
                nod = nod + ipt[j] * w[0][i][j];
            }
            l1.push_back(nod);
            l1a.push_back(sigmoid(nod));
        }
    }
    else if (act1 == "id") {
        for (int i = 0; i < l1dim; i++) {
            nod = 0;
            for (int j = 0; j < xdim + 1; j++) {
                nod = nod + ipt[j] * w[0][i][j];
            }
            l1.push_back(nod);
            l1a.push_back(nod);
        }
    }
}

void NN::tol2() {
    l2.clear();
    l2a.clear();
    l1a.push_back(1);
    double nod;
    if (act2 == "leakyrelu") {
        for (int i = 0; i < l2dim; i++) {
            nod = 0;
            for (int j = 0; j < l1dim + 1; j++) {
                nod = nod + l1a[j] * w[1][i][j];
            }
            l2.push_back(nod);
            l2a.push_back(leakyrelu(nod));
        }
    }
    else if (act2 == "relu") {
        for (int i = 0; i < l2dim; i++) {
            nod = 0;
            for (int j = 0; j < l1dim + 1; j++) {
                nod = nod + l1a[j] * w[1][i][j];
            }
            l2.push_back(nod);
            l2a.push_back(relu(nod));
        }
    }
    else if (act2 == "sigmoid") {
        for (int i = 0; i < l2dim; i++) {
            nod = 0;
            for (int j = 0; j < l1dim + 1; j++) {
                nod = nod + l1a[j] * w[1][i][j];
            }
            l2.push_back(nod);
            l2a.push_back(sigmoid(nod));
        }
    }
    else if (act2 == "id") {
        for (int i = 0; i < l2dim; i++) {
            nod = 0;
            for (int j = 0; j < l1dim + 1; j++) {
                nod = nod + l1a[j] * w[1][i][j];
            }
            l2.push_back(nod);
            l2a.push_back(nod);
        }
    }
}

void NN::tol3() {
    l3.clear();
    l3a.clear();
    l2a.push_back(1);
    double nod;
    if (act3 == "leakyrelu") {
        for (int i = 0; i < l3dim; i++) {
            nod = 0;
            for (int j = 0; j < l2dim + 1; j++) {
                nod = nod + l2a[j] * w[2][i][j];
            }
            l3.push_back(nod);
            l3a.push_back(leakyrelu(nod));
        }
    }
    else if (act3 == "relu") {
        for (int i = 0; i < l3dim; i++) {
            nod = 0;
            for (int j = 0; j < l2dim + 1; j++) {
                nod = nod + l2a[j] * w[2][i][j];
            }
            l3.push_back(nod);
            l3a.push_back(relu(nod));
        }
    }
    else if (act3 == "sigmoid") {
        for (int i = 0; i < l3dim; i++) {
            nod = 0;
            for (int j = 0; j < l2dim + 1; j++) {
                nod = nod + l2a[j] * w[2][i][j];
            }
            l3.push_back(nod);
            l3a.push_back(sigmoid(nod));
        }
    }
    else if (act3 == "id") {
        for (int i = 0; i < l3dim; i++) {
            nod = 0;
            for (int j = 0; j < l2dim + 1; j++) {
                nod = nod + l2a[j] * w[2][i][j];
            }
            l3.push_back(nod);
            l3a.push_back(nod);
        }
    }
}

//....................................................................................

double NN::lossMSEr() {
    double lo = 0;
    for (int i = 0; i < numxy; i++) {
        for (int j = 0; j < ydim; j++) {
            lo = lo + (py[i][j] - y[i][j]) * (py[i][j] - y[i][j]);
        }
    }
    return lo; // numxy;
}

double NN::DlossMSEr_py(int i, int j) {
    return 2 * (py[i][j] - y[i][j]); // numxy;
}

double NN::lossBCEn() {
    double lo = 0;
    for (int i = 0; i < numxy; i++) {
        for (int j = 0; j < ydim; j++) {
            lo = lo + y[i][j] * log(py[i][j]) + (1 - y[i][j]) * log(1 - py[i][j]);
        }
    }
    lo = -lo / numxy;
    return lo;
}

double NN::DlossBCEn_py(int i, int j) {
    return (-y[i][j] / py[i][j] + (1 - y[i][j]) / (1 - py[i][j])) / numxy;
}

//.................................................................................

void NN::calcloss() {
    if (lossfunc == "MSE") {
        loss = lossMSEr();
    }
    else if (lossfunc == "BCE") {
        loss = lossBCEn();
    }
}

void NN::backl3() {
    Dloss_l.clear();
    std::vector<double> m;
    if (lossfunc == "MSE") {
        for (int i = 0; i < numxy; i++) {
            m.clear();
            for (int j = 0; j < l3dim; j++) {
                m.push_back(DlossMSEr_py(i, j));
            }
            Dloss_l.push_back(m);
        }
    }
    else if (lossfunc == "BCE") {
        for (int i = 0; i < numxy; i++) {
            m.clear();
            for (int j = 0; j < l3dim; j++) {
                m.push_back(DlossBCEn_py(i, j));
            }
            Dloss_l.push_back(m);
        }
    }

    if (act3 == "leakyrelu") {
        for (int i = 0; i < numxy; i++) {
            for (int j = 0; j < l3dim; j++) {
                Dloss_l[i][j] = Dleakyrelu(tl3[i][j]) * Dloss_l[i][j];
            }
        }
    }
    else if (act3 == "relu") {
        for (int i = 0; i < numxy; i++) {
            for (int j = 0; j < l3dim; j++) {
                Dloss_l[i][j] = Drelu(tl3[i][j]) * Dloss_l[i][j];
            }
        }
    }
    else if (act3 == "sigmoid") {
        for (int i = 0; i < numxy; i++) {
            for (int j = 0; j < l3dim; j++) {
                Dloss_l[i][j] = Dsigmoid(tl3[i][j]) * Dloss_l[i][j];
            }
        }
    }

    wo[2] = w[2];
    //learnrate3 = learnrate / numxy;
    double d;
    for (int i = 0; i < l3dim; i++) {
        for (int j = 0; j < l2dim + 1; j++) {
            d = 0;
            for (int k = 0; k < numxy; k++) {
                d = d + Dloss_l[k][i] * tl2a[k][j];
            }
            w[2][i][j] = wo[2][i][j] - learnrate3 * d;
        }
    }
}

void NN::backl2() {
    std::vector<std::vector<double>> Dloss_lnex = Dloss_l;
    Dloss_l.clear();
    std::vector<double> m;
    double d, r = 0;
    for (int i = 0; i < numxy; i++) {
        m.clear();
        for (int j = 0; j < l2dim; j++) {
            d = 0;
            for (int k = 0; k < l3dim; k++) {
                d = d + Dloss_lnex[i][k] * wo[2][k][j];
                r = r + wo[2][k][j];
            }
            m.push_back(d);
        }
        Dloss_l.push_back(m);
    }

    if (act2 == "leakyrelu") {
        for (int i = 0; i < numxy; i++) {
            for (int j = 0; j < l2dim; j++) {
                Dloss_l[i][j] = Dleakyrelu(tl2[i][j]) * Dloss_l[i][j];
            }
        }
    }
    else if (act2 == "relu") {
        for (int i = 0; i < numxy; i++) {
            for (int j = 0; j < l2dim; j++) {
                Dloss_l[i][j] = Drelu(tl2[i][j]) * Dloss_l[i][j];
            }
        }
    }
    else if (act2 == "sigmoid") {
        for (int i = 0; i < numxy; i++) {
            for (int j = 0; j < l2dim; j++) {
                Dloss_l[i][j] = Dsigmoid(tl2[i][j]) * Dloss_l[i][j];
            }
        }
    }

    wo[1] = w[1];
    //learnrate2 = learnrate3 / r;
    for (int i = 0; i < l2dim; i++) {
        for (int j = 0; j < l1dim + 1; j++) {
            d = 0;
            for (int k = 0; k < numxy; k++) {
                d = d + Dloss_l[k][i] * tl1a[k][j];
            }
            w[1][i][j] = wo[1][i][j] - learnrate2 * d;
        }
    }
}

void NN::backl1() {
    std::vector<std::vector<double>> Dloss_lnex = Dloss_l;
    Dloss_l.clear();
    std::vector<double> m;
    double d, r = 0;
    for (int i = 0; i < numxy; i++) {
        m.clear();
        for (int j = 0; j < l1dim; j++) {
            d = 0;
            for (int k = 0; k < l2dim; k++) {
                d = d + Dloss_lnex[i][k] * wo[1][k][j];
                r = r + wo[1][k][j];
            }
            m.push_back(d);
        }
        Dloss_l.push_back(m);
    }

    if (act1 == "leakyrelu") {
        for (int i = 0; i < numxy; i++) {
            for (int j = 0; j < l1dim; j++) {
                Dloss_l[i][j] = Dleakyrelu(tl1[i][j]) * Dloss_l[i][j];
            }
        }
    }
    else if (act1 == "relu") {
        for (int i = 0; i < numxy; i++) {
            for (int j = 0; j < l1dim; j++) {
                Dloss_l[i][j] = Drelu(tl1[i][j]) * Dloss_l[i][j];
            }
        }
    }
    else if (act1 == "sigmoid") {
        for (int i = 0; i < numxy; i++) {
            for (int j = 0; j < l1dim; j++) {
                Dloss_l[i][j] = Dsigmoid(tl1[i][j]) * Dloss_l[i][j];
            }
        }
    }

    //learnrate1 = learnrate2 / r;
    for (int i = 0; i < l1dim; i++) {
        for (int j = 0; j < xdim + 1; j++) {
            d = 0;
            for (int k = 0; k < numxy; k++) {
                d = d + Dloss_l[k][i] * tipt[k][j];
            }
            w[0][i][j] = w[0][i][j] - learnrate1 * d;
        }
    }
}

//...........................................................................................

void NN::traincyc() {
    for (int i = 0; i < numxy; i++) {
        ipt = x[i];
        tol1();
        tol2();
        tol3();
        tipt.push_back(ipt);
        tl1.push_back(l1);
        tl1a.push_back(l1a);
        tl2.push_back(l2);
        tl2a.push_back(l2a);
        tl3.push_back(l3);
        py.push_back(l3a);
    }
    calcloss();
    backl3();
    backl2();
    backl1();
    tipt.clear();
    tl1.clear();
    tl1a.clear();
    tl2.clear();
    tl2a.clear();
    tl3.clear();
    py.clear();
}

void NN::train() {
    for (int i = 0; i < epoch; i++) {
        traincyc();
        //std::cout << i << " Loss: " << loss << "\n";
    }
    for (int i = 0; i < numxy; i++) {
        ipt = x[i];
        tol1();
        tol2();
        tol3();
        tipt.push_back(ipt);
        tl1.push_back(l1);
        tl1a.push_back(l1a);
        tl2.push_back(l2);
        tl2a.push_back(l2a);
        tl3.push_back(l3);
        py.push_back(l3a);
    }
    calcloss();
    std::cout << "Loss: " << loss << "\n";
}
