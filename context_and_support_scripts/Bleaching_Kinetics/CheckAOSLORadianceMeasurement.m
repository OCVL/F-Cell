% CheckAOSLORadianceMeasurements
%
% We measured the radiance of the OneLight diffuser using the PR-670.
%
% Power measured at the cornea: ~0.4 uW/cm2
% Stimulus distance: ~39 cm
% Stimulus diameter: ~0.35 cm
% 
% 680 power at cornea for 1 degree field, 27 uW
%
% 12/11/15  dhb  Wrote it to summarize measurements of today

% Clear
clear; close all;

% Load measurement
S = [380 5 81];
load AOSLOOneLightRadianceSpd_151211
wls = SToWls(S);
figure;
plot(wls,OLSpd);
totalRadiance = sum(OLSpd);