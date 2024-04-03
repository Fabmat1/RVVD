#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <fstream>
using namespace std;

extern "C" {
double gaussian_kernel(double x, double sigma) {
    double result = exp(-(x*x)/(2.*sigma*sigma));
    if (isnan(result)){
        return 0.;
    }
    else {
        return result;
    }
}

double photamp(double x, double amp) {
    double zeropoint = 1.-amp/2.;
    double result =  1./(sqrt((x-zeropoint)*(zeropoint+amp-x)));

    if (isnan(result)){
        return 0.;
    }
    else {
        return result;
    }
}


void write_vector_data(const double* x_space, const double*  y_space, int x_size, const string& filename) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        // Error: could not open the file
        return;
    }

    for (size_t i = 0; i < x_size; ++i) {
        file << x_space[i] << "," << y_space[i] << endl;
    }

    file.close();
}


double* linspace(double start, double stop, int num) {
    auto* result = new double[num];

    double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }

    return result;
}


double simpson_integral(const double* x, const double* y, int x_size) {
    int n = x_size - 1; // Number of intervals
    double h = (x[n] - x[0]) / n; // Step size

    double sum = y[0] + y[n]; // Sum of the first and last terms
    for (int i = 1; i < n; i += 2) {
        sum += 4 * y[i]; // Multiply odd-indexed terms by 4
    }
    for (int i = 2; i < n - 1; i += 2) {
        sum += 2 * y[i]; // Multiply even-indexed terms by 2
    }

    return sum * h / 3.0;
}


double* distribution(const double* x, int x_size, double amp, double sigma) {
    if (x_size <= 0) {
        cerr << "Error: Invalid array size!" << endl;
        return nullptr;
    }

    double span = x[x_size - 1] - x[0];
    double* tauspace = linspace(-span/2, span/2, x_size);
    auto* result = new double[x_size];

    #pragma omp parallel for
    for (int i = 0; i < x_size; ++i) {
        auto* y_tauspace = new double[x_size];
        for (int j = 0; j < x_size; ++j) {
            y_tauspace[j] = gaussian_kernel(tauspace[j], sigma) * photamp(x[i] - tauspace[j], amp);
        }
        result[i] = simpson_integral(tauspace, y_tauspace, x_size);
        delete[] y_tauspace;
    }

    delete[] tauspace;
    return result;
}}


int main() {
    double sigma = 0.01;
    double amp = 0.5;
    int resolution = 10000;
    double* x_space = linspace(0, 2, resolution);
    double* y_space = distribution(x_space, resolution, amp, sigma);

    write_vector_data(x_space, y_space, resolution, "/home/fabian/CLionProjects/auxilliary/test.txt");
    delete[] x_space;
    delete[] y_space;
}
