variable qualies_for_spectroscopy_automated =
  struct{xrange=500., % Adjust x-range for each plotted panel.
         error_estimation=0, % Set to 1 to use conf_loop for uncertainties (not covariance)
         auto_freeze_vsini=1, % Set to 0 to turn off automatic freezing of vsini.
         add_telluric_model=0, % set to 1 to also model the telluric transmission spectrum.
         apply_mask=0, % Use 'create_ignore_list_from_spectral_mask'
%         sdss_individual=0, % 0 -> fit coadded; 1 -> fit individual spectra
%         filter_snr=15., % remove spectra with low SNR
%         require_blue=4300., % remove spectra that start redder than this
%         save_model="fits", % Set to 'ascii' or 'fits' to save model spectra and rebinned data.
         xfig_ignore=-1, % 'ignore' qualifier for 'xfig_residual_plot'
         untie={"vrad"}}; % List of parameter names that should not be tied.
modelgrid = search_grid_fit_photometry(bpaths, modelgrid, "grid.fits");
variable sout = spectroscopy_automated(input, modelgrid, initial_guess_params_values;;
                                       qualies_for_spectroscopy_automated);

exit;
