%%CorrectIrradiacneForLumEquivalence
%
% Description:
%    This script corrects light power for the finite bandwidth of our
%    nominally monochromatic stimuli.  
%
%    The particular correction is to find the factor that converts
%    stimulus irradiance to an equivalent monochromatic irradiance,
%    (emi) such that a purely monochromatic light of the emi will have the
%    same luminance as the actual narrowband stimulus.
%
%    The reason this is useful is a) to evaluate whether we want to apply
%    a correction factor to our data for comparison with the photopic
%    luminosity function and b) if so to compute what that correction
%    factor should be.
%
%    One could also consider applying a similar correction for the
%    spectral sensitivity of the radiometer we use to measure irradiance,
%    if that spectral sensitivity is known.

% 09/21/17  dhb  Wrote it, following conversation and code snippets from Rob Cooper.

%% Clear
clear; close;

%% Narrowband stimulus parameters
specifiedPowerUW = 0.145; 
centerWl = 480;
specifiedFWHM = 30;
fprintf('Analyzing center wl %d, FWHM %0.1f\n',centerWl,specifiedFWHM);

%% Wavelength support
%
% Keep spacing at 1 nm (assumed below) and don't extend beyond
% ranger of specification of T_xyzCIE2 (390 to 830).
wls = (390:1:780)';

%% Compute the spectral power distribution of the narrowband stimulus
%
% Formula for std from BrainardLabToolbox function, and independently derived by
% Rob Cooper.
spdStd = FWHMToStd(specifiedFWHM);
narrowbandSpd = specifiedPowerUW*normpdf(wls,centerWl,spdStd);

%% Make a plot of the narrowband stimulus
figure; clf; hold on
plot(wls,narrowbandSpd,'r','LineWidth',2);
xlabel('Wavlength')
ylabel('Power');

%% Get total power of spd and fwhm as a check.
%
% With 1 nm spacing as above, just need to sum to get the power.
% The fwhm routine extracts the FWHM from the spd.
fromspdPowerUW = sum(narrowbandSpd);
fromspdFWHM = fwhm(wls,narrowbandSpd,0);
fprintf('Specified narrowband power %0.4f, from spd %0.4f\n',specifiedPowerUW,fromspdPowerUW);
fprintf('Specified FWHM: %d, from spd %d\n',specifiedFWHM,fromspdFWHM);

%% Create the monochromatic stimulus with same total irradiance
index = find(wls == centerWl);
if (length(index) ~= 1)
    error('Oops.  Cannot find center wl in wl samples.');
end
monochromaticSpd = zeros(size(wls));
monochromaticSpd(index) = specifiedPowerUW;

%% Load the photopic luminosity function
lum_Tfilename = 'T_xyzCIEPhys2';
load(lum_Tfilename); 
T_Y = SplineCmf(S_xyzCIEPhys2,683*T_xyzCIEPhys2(2,:),wls);

%% Compute luminance in arbitrary units
narrowbandY = T_Y*narrowbandSpd;
monochromaticY = T_Y*monochromaticSpd;
fprintf('Computing luminance using PTB file %s for specification of photopic luminosity\n',lum_Tfilename);
fprintf('Narrowband Y %0.3g, monochromatic Y %0.3g (arb. units)\n',narrowbandY,monochromaticY);

%% Compute equivalent monochromatic irradiance
equivMonochromaticPowerUW = narrowbandY*specifiedPowerUW/monochromaticY;
equivMonochromaticSpd = zeros(size(wls));
equivMonochromaticSpd(index) = equivMonochromaticPowerUW;
equivMonochromaticY = T_Y*equivMonochromaticSpd;
fprintf('Specified power %0.4f uW, equivalent monochromatic power %0.4f uW, equivalent monochromatic Y %0.3g (arb. units)\n',specifiedPowerUW,equivMonochromaticPowerUW,equivMonochromaticY);

%% Compute size of effect in log units
log10Effect = log10(equivMonochromaticPowerUW/specifiedPowerUW);
fprintf('Log10 equivMono / specified power %0.3f\n',log10Effect);


