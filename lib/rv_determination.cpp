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

int main(){
    auto start = chrono::high_resolution_clock::now();
    cout << "Hello World" << endl;
    auto end = chrono::high_resolution_clock::now();

    vector<vector<double>> data = loadFromFile("spectra_processed/0148_2_gaia_dr3_1746_930_01.txt");
    double mjd = loadMJD("spectra_processed/0148_2_gaia_dr3_1746_930_mjd.txt");
    vector<double> x = data[0];
    vector<double> y = data[1];
    vector<double> z = data[2];

    for (int i = 0; i < x.size(); ++i) {
        cout << x[i] << "," << y[i] << "," <<  z[i] << endl;
    }
    cout << "MJD: " << mjd << endl;
    cout << "Delta_wl: " << radvel(5000, 10) << endl;

    cout << "Program took " << chrono::duration_cast<chrono::milliseconds>(end-start).count()/1000 << "s to execute." << endl;
    return 0;
}