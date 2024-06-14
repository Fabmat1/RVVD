#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;


vector<double> lines_to_fit = {6562.79,4861.35,4340.472,4101.734,3970.075,3888.052,3835.387,4026.19,4471.4802,4921.9313,5015.678,5875.6,6678.15,4541.59,4685.70,5411.52
//      "He_I_4100": 4100.0,
//      "He_I_4339": 4338.7,
//      "He_I_4859": 4859.35,
//      "He_I_6560": 6560.15,
};


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
    cout << "Delta_wl: " << radvel(5000, 10) << endl;


    vector<double> velspace = linspace(-1000., 1000, 10000);
    vector<double> velspace_y;


    for (double vel: velspace) {
        vector<double> these_lines(lines_to_fit.size());
        for (int k = 0; k < lines_to_fit.size(); ++k) {
            these_lines[k] = lines_to_fit[k]+radvel(lines_to_fit[k], vel);
        }
        velspace_y.push_back(interpolate1d(x, y, these_lines));
    }

    for (int i = 0; i < velspace.size(); ++i) {
        cout << velspace[i] << "," << velspace_y[i] << endl;
    }
    auto end = chrono::high_resolution_clock::now();
    cout << "Program took " << chrono::duration_cast<chrono::milliseconds>(end-start).count()/1000 << "s to execute." << endl;

    return 0;
}