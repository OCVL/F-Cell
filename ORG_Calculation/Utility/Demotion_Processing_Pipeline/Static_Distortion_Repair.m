function [ static_grid_distortion ] = Static_Distortion_Repair( fringes_fullpath, fitstartind )
% [ static_grid_distortion ] = Static_Distortion_Repair( fringes_fullpath )
%   Robert F Cooper, 2017-09-15
%
% This function estimates the residual static distortion not currently
% captured by the dewarping software. Unlike the original version, it loads the fringes inside the function.
%
%    @fringes_fullpath: The path to the fringes file from Alf Dubra's calibration software.
%
%    @fitstartind: The pixel value at which we begin to calculate the fit. Increase this if you have issues
%                  with noisy fits.

load(fringes_fullpath,'horizontal_fringes_indices_minima');



horizontal_fringes_indices_minima=horizontal_fringes_indices_minima';

if ~exist('fitstartind','var')
     fitstartind=min(horizontal_fringes_indices_minima); % The point at which we'll start fitting (as we know the beginning is distorted)
end

xvals = (1:length(horizontal_fringes_indices_minima))';
% Determine the linreg fit coefficients (X\y)
coeff = [ones(size(xvals(fitstartind:end))) xvals(fitstartind:end)] \ horizontal_fringes_indices_minima(fitstartind:end);

fringe_reg = coeff(2).*xvals + coeff(1);

% figure; plot(xvals,horizontal_fringes_indices_minima,'r',xvals,fringe_reg,'k'); 

residuals = (fringe_reg-horizontal_fringes_indices_minima);

p=fit(horizontal_fringes_indices_minima,residuals,'exp2');

% Set for how far into the fringes you want to attempt to repair distortion
dist_fix_rng = 1000;

static_grid_distortion = zeros(dist_fix_rng,1);
for i=1:dist_fix_rng
	static_grid_distortion(i) = -p(i);
end

% figure; plot(1:dist_fix_rng, -static_grid_distortion,horizontal_fringes_indices_minima,residuals);
end

