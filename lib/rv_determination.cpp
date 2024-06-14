#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
using namespace std;



vector<double> lines_to_fit = {6562.79,4861.35,4340.472,4101.734,3970.075,3888.052,3835.387,4026.19,4471.4802,4921.9313,5015.678,5875.6,6678.15,4541.59,4685.70,5411.52
//      "He_I_4100": 4100.0,
//      "He_I_4339": 4338.7,
//      "He_I_4859": 4859.35,
//      "He_I_6560": 6560.15,
};

// Implementation of Gaussian function
double gaussian(double x, double sigma) {
  double a = 1.0 / (sigma * sqrt(2.0 * M_PI));
  double b = exp(-1.0 * pow(x, 2.0) / (2.0 * pow(sigma, 2.0)));
  return a * b;
}

// Function to apply Gaussian filter
vector<double> apply_gaussian_filter(vector<double>& arr_x, vector<double>& arr_y, double sigma){
    // Create container for result vector
    vector<double> result(arr_y.size());

    // Loop over each element in the array
    for(int i = 0; i < arr_x.size(); i++){
        if (arr_x[i]-arr_x[0] < 2*sigma || arr_x[i] > arr_x[arr_x.size()-1]-2*sigma){
            result[i] = INFINITY;
            continue;
        }
        // Initial value for each output point
        double output = 0.0;

        // Apply Gaussian filter
        for(int j = 0; j < arr_x.size(); j++){
            double x_off = arr_x[j] - arr_x[i]; // Distance from center
            output += arr_y[j] * gaussian(x_off, sigma);
        }

        // Save to result
        result[i] = output;
    }

    return result;
}



vector<vector<double>> loadFromFile(const string &filename) {
    ifstream file(filename);
    vector<vector<double>> values(3); // 3 vectors for storing the values

    if (!file.is_open()) {
        cerr << "Error: Could not open the file " << filename << endl;
        return values;
    }

    string line;
    while (getline(file, line)) {
        istringstream lineStream(line);
        double val1, val2, val3;

        if (lineStream >> val1 >> val2 >> val3) {
            values[0].push_back(val1);
            values[1].push_back(val2);
            values[2].push_back(val3);
        } else {
            cerr << "Error: Invalid line format: " << line << endl;
        }
    }

    file.close();
    return values;
}
extern "C" {

double radvel(double wl, double v){ // wl in angstrom, velocity in km/s
    // (dwl)/wl = v/c
    wl *= 1e-10;
    double dwl = wl*(v/299792.458);
    dwl *= 1e10;
    return dwl;
}


double loadMJD(const string& filename) {
    ifstream file(filename);
    double value = 0.0;

    if (!file.is_open()) {
        cerr << "Error: Could not open the file " << filename << endl;
        return value; // Returning 0.0 as a default error value
    }

    file >> value;
    if (file.fail()) {
        cerr << "Error: Failed to read a double value from the file " << filename << endl;
        value = 0.0; // Resetting value to 0.0 in case of read failure
    }

    file.close();
    return value;
}

vector<double> linspace(double start, double end, int num) {
    vector<double> result;
    if (num <= 0) {
        return result;
    }

    if (num == 1) {
        result.push_back(start);
        return result;
    }

    double step = (end - start) / (num - 1);

    for (int i = 0; i < num; ++i) {
        result.push_back(start + step * i);
    }

    return result;
}


vector<double> loadLines(const string& filename) {
    vector<double> doubles;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return doubles;
    }

    double value;
    while (file >> value) {
        doubles.push_back(value);
    }

    if (file.bad()) {
        cerr << "Error reading file: " << filename << endl;
    }

    file.close();
    return doubles;
}


double interpolate1d(const vector<double>& x_vec, const vector<double>& y_vec, const vector<double>& x_of_interest){
    double sum = 0.;
    int j;
    double y;
    double x0;
    double x1;
    double y0;
    double y1;

    for (double xi: x_of_interest) {
        if (xi < x_vec[0] || xi > x_vec[x_vec.size() - 1]) { continue; }

        auto it = lower_bound(x_vec.begin(), x_vec.end(), xi);
        j = distance(x_vec.begin(), it);

        if (j == 0) {
            y = 0;
        } else if (j == x_vec.size() - 1) {
            y = 0;
        } else {
            x0 = x_vec[j - 1];
            x1 = x_vec[j];
            y0 = y_vec[j - 1];
            y1 = y_vec[j];
            y = y0 + (y1 - y0) * ((xi - x0) / (x1 - x0));
        }
//                            #pragma omp atomic
        sum += y;
    }

    return sum;
}


vector<double> ArrayToVector(double* arr, size_t arr_len) {
    return vector<double>(arr, arr + arr_len);
}



double determine_radvel(double* wl_array, double* flx_array, double* lines_to_fit,
                                       int lenvec, int lenlines){

//    cout << "lenvec " << lenvec << endl;
//    cout << "lenlines " << lenlines << endl;
//    cout << "wl arr " << wl_array[0] << endl;
//    cout << "flx arr " << flx_array[0] << endl;
//    cout << "lines to fit " << lines_to_fit[0] << endl;
    vector<double> wl_array_vec = ArrayToVector(wl_array, lenvec);
    vector<double> flx_array_vec = ArrayToVector(flx_array, lenvec);
    vector<double> velspace = linspace(-2500., 2500., 5000);
    vector<double> velspace_y;


    for (double vel: velspace) {
        vector<double> these_lines(lenlines);
        for (int k = 0; k < lenlines; ++k) {
            these_lines[k] = lines_to_fit[k]+radvel(lines_to_fit[k], vel);
        }
        velspace_y.push_back(interpolate1d(wl_array_vec, flx_array_vec, these_lines));
    }

    velspace_y = apply_gaussian_filter(velspace, velspace_y, 200);

    ofstream outfile("outvel.txt");

    for (int i = 0; i < velspace.size(); ++i) {
        outfile << velspace[i] << "," << velspace_y[i] << "\n";
    }

    outfile.close();

    // Find the minimum point of the Gaussian
    auto min_value_it = min_element(velspace_y.begin(), velspace_y.end());
    auto index = min_value_it - velspace_y.begin();
    double min_value = velspace.at(index);

    // Calculate the error as the standard deviation of the last ten percent of velspace_y
    auto stdev_start = velspace_y.begin() + 9 * velspace_y.size() / 10;
    double mean = accumulate(stdev_start, velspace_y.end(), 0.0) / distance(stdev_start, velspace_y.end());
    double sq_sum = inner_product(stdev_start, velspace_y.end(), stdev_start, 0.0);
    double stdev = sqrt(sq_sum / distance(stdev_start, velspace_y.end()) - pow(mean, 2));

//    double result[2] = {min_value, stdev};
    return min_value;
}



int main(){
    auto start = chrono::high_resolution_clock::now();

    lines_to_fit = loadLines("H_sd_lines.txt");

    vector<vector<double>> data = loadFromFile("spectra_processed/0780_9_Gaia_444485_930_01.txt");
    double mjd = loadMJD("spectra_processed/0780_9_Gaia_444485_930_mjd.txt");
    vector<double> x = data[0];
    vector<double> y = data[1];
    vector<double> z = data[2];

//    for (int i = 0; i < x.size(); ++i) {
//        cout << x[i] << "," << y[i] << "," <<  z[i] << endl;
//    }
    cout << "MJD: " << mjd << endl;

    auto end = chrono::high_resolution_clock::now();
    cout << "Program took " << chrono::duration_cast<chrono::milliseconds>(end-start).count()/1000 << "s to execute." << endl;

    return 0;
}

}