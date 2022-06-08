# Radial Velocity Variation Determination
## _A tool for spectroscopic determination of the radial Velocity of Stars_

### Usage
#### Providing spectral data

This tool was initially designed for data presented in a specific way.
The spectra for the stars whose RV-Curves you wish to calculate need to be put in the `/spectra`-directory.<br> 
Each set of subspectra for a given star need to have a common file prefix,
followed by a unique integer number identifier from _01_ to _99_ seperated by an underscore.<br> 
For example `spec-xxxxxx_01.txt`,`spec-xxxxxx_02.txt`,`spec-xxxxxx_03.txt` would be valid filenames for the three
subspectra of a star.<br> 
These files need to contain in this order the air wavelength in Angström, Flux in ergs/s/cm^2/Å and standard deviation of the Flux in the same units.<br> 
The separator used in these files can be specified in the `SPECTRUM_FILE_SEPARATOR` constant in `main.py`, subsequent data points need to be separated by line breaks.

Furthermore, the times at which the spectra where taken need to be supplied in mjd format in an additional .txt-file.<br> 
This file needs to have the same prefix as the affiliated spectra, and `mjd` as a suffix, again seperated by an underscore.<br> 
For example, `spec-xxxxxx_mjd.txt` is the correct filename for the spectra of the example in the last paragraph.<br> 
This file needs to contain the mjds for the associated subspectra, in the same order as the indices and separated by line breaks.

Comments may be made in these files, starting with a `#`.

If this format of providing data does not fit your use case, modify the `load_spectrum()` function in `main.py` to accomodate your Needs.

#### Settings

The output and execution of the script can be modified via various constants that are located at the tops of each python script files. <br> 
These are identifiable by their full capitalization, and their purpose is explained in comments in the code. <br>
Lines that can potentially be used for analysis can also be set in the `lines` dict in `main.py`,with lines that should be cut out being set in `disturbing_lines`.

#### Using a catalogue

If desired, a "catalogue" of stars can be used in the analysis, which enables only using a subset of the stars provided (only the ones present in the "catalogue" file) and assigning GAIA source IDs to the systems used. <br>
The catalogue file location can be specified in the settings in `main.py`, with the file being provided in `.csv` format and containing at least two columns with the headers `file` and `source_id`.
These columns need to contain file prefixes and corresponding GAIA source ids of the stars to be looked at.

#### Execution

The package contains multiple different python scripts with varying purposes:
1. `main.py` - The main script, which calculates RV-Curves and fit parameters from provided spectra.
2. `analyse_results.py` - To be executed after `main.py` finishes. Calculates various parameters - including the log p parameter - from the output of the main script.
3. `plot_interesting_area.py` - To be executed after `main.py` finishes. Used to plot specific lines of specific subspectra. Lines may be plotted with or without their determined fit, in manually defined limits, and with various other options.
4. `fit_rv_curve.py` - Not yet ready for generalized use!! DON'T USE!
5. `select_observation_targets.py` - Not yet ready for generalized use!! DON'T USE!

#### Output

The Output of `main.py` is saved in the `\output` directory. <br>
A seperate folder is created for each set of spectra, containing tables of the fit parameters for all used lines, the RV-Values and subfolders named after the indices of the subspectra. <br>
These subfolders contain images of the fitted lines for the subspectra, if generated.<br>
Each outer folder holds three Tables and, if generated, two images:
1. `single_spec_vals.csv` - A comma-separated array of fit- and additional parameters from the single line fits for each line.
2. `culum_spec_vals.csv` - A comma-separated array of fit- and additional parameters from the cumulative line fits for each line.
3. `RV_variation.csv` - A comma-separated array of radial velocities and their uncertainties, with associated mjds, calculated from the single and cumulative fit parameters.
4. `RV_variation.png` - A simple plot of the RV-Variation over time.
5. `RV_variation_broken_axis.png` - A plot of the RV-Variation over time, with the time axis being broken in areas where no new values were measured for a long time.

##### Output headers

The columns in the above mentioned tables hold the following information:
* `subspectrum` - Integer index of the subspectrum of the line investigated.
* `line_name` - Line name (from the `lines` variable in `main.py`)
* `line_loc` - Line source wavelength [Å]
* `height` and `u_height` - True peak height and uncertainty thereof calculated from the parameters assuming a pseudo-voigt profile.
* `reduction_factor` and `u_reduction_factor` - Wavelength reduction factor lambda_o/lambda_s and uncertainty thereof.
* `lambda_0` and `u_lambda_0` - Wavelength the peak was detected at and uncertainty [Å]
* `eta` and `u_eta` - Parameter eta from pseudo voigt profile and uncertainty. [no unit]
* `sigma` and `u_sigma` - Standard deviation of the pseudo voigt peak, calculated from gamma, and uncertainty. [Å]
* `gamma` and `u_gamma` - Parameter gamma from pseudo voigt profile and uncertainty. [Å]
* `scaling` and `u_scaling` - Scalar factor from pseudo voigt profile and uncertainty. [ergs/s/cm^2/Å]
* `flux_0` and `u_flux_0` - Shift from linear part of the fit function and uncertainty. [ergs/s/cm^2/Å]
* `slope` and `u_slope` - Slope from linear part of the fit function and uncertainty. [ergs/s/cm^2/Å^2]
* `RV` and `u_RV` - Radial velocity from fit parameters and uncertainty [m/s]
* `signal_strength` and `noise_strength` - Signal and Noise strength, determined by mean squared displacement from the linear part of the fit function in the peak and noise regions. [arb. units]
* `SNR` - Signal-to-Noise Ratio [no unit]
* `sanitized` - deprecated, will be removed in future version
* `cr_ind` - Indices of cosmic ray events detected in the flux array.


✨
