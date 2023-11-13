# Radial Velocity Variability Determination

## Installation

Clone the latest release version from git using 

```bash
git clone https://www.github.com/Fabmat1/RVVD --branch *latest_release_tag*
```

Then, navigate to the root directory of the repository and install dependencies using
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
### Preparation
Move the spectra you are looking to process to `/spectra_raw`. 
Currenty, it is required that they are in ASCII format with a comment (`#`) in the first line that specifies either the GAIA DR3 ID or the RA/DEC coordinates of the star it belongs to, as well as the MJD/HJD in this header. 
The header must be of the format `# ('RA', X.XXXX), ('DEC, X.XXXXXX), ...`
> **Note**
> .fits support will be added in the next release

### Execution
Run RVVD using 
```bash
python interactive.py
```

and follow the instructions in the interactive window to calculate the radial velocities of the spectra.