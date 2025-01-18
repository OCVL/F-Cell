% RetinalIrradianceConversion
%
% Example calculation converting a known retinal irradiance to trolands (and other units).
% This example is set up for monochromatic light
%
% This relies on underlying routines in the Psychophysics toolbox.
%
% 5/5/15  dhb	Wrote it from earlier code.

%% Clear and close
clear; close all

%% Set wavelength sampling
%
% Using 1 nm is a good idea because then power per nm and power per wl band
% are the same.
S = [380 1 401];
wls = SToWls(S);

%% Pick a wavelength and retinal irradiance
%
% And represent the light as a spectrum
theMonochromaticWl = 550;
theMonochromaticRetIrradianceWattsPerM2 = 1;
index = find(wls == theMonochromaticWl);
if (length(index) ~= 1)
    error('This routine assumes monochromatic light');
end
theInputRetIrradianceWattsPerM2 = zeros(size(wls));
theInputRetIrradianceWattsPerM2(index) = theMonochromaticRetIrradianceWattsPerM2;

%% Specify pupil area and eye length
%
% This allows us to convert the irradiance to an equivalent radiance.
% Although we don't have to do this, it is convenient because of the way
% the underlying routines are written.  If everything is working correctly,
% trolands will be independent of pupil diameter, and retinal irradiance
% will be independent of both when expressed in area (rather than degrees
% units.
pupilDiamMm = 6;
pupilAreaMm2 = pi*((pupilDiamMm/2)^2);
eyeLengthMm = 17;
eyeLengthM = eyeLengthMm/1000;
pupilAreaM2 = pupilAreaMm2/(1e6);

% Deg to mm conversion. The call through the inverse routine doesn't quite invert
% so I force conversion factors that are exact inverses.
MmToDeg = RetinalMMToDegrees(1,eyeLengthMm);
Mm2ToDeg2 = MmToDeg^2;
DegToMm = 1/MmToDeg;
%DegToMm = DegreesToRetinalMM(1,eyeLengthMm);
Deg2ToMm2 = DegToMm^2;

theInputRetIrradianceWattsPerMm2 = theInputRetIrradianceWattsPerM2/(1e6);
theInputRetIrradianceWattsPerDeg2 = theInputRetIrradianceWattsPerMm2/Mm2ToDeg2;

%% Convert irradiance to radiance
radianceWattsPerM2Sr = RetIrradianceAndPupilAreaEyeLengthToRadiance(theInputRetIrradianceWattsPerM2,S,pupilAreaM2,eyeLengthM);

%% Unit coversion
radianceWattsPerCm2Sr = (10.^-4)*radianceWattsPerM2Sr;
radianceQuantaPerCm2SrSec = EnergyToQuanta(S,radianceWattsPerCm2Sr);

%% Load CIE functions.   
load T_xyz1931
T_xyz = SplineCmf(S_xyz1931,683*T_xyz1931,S);
photopicLuminanceCdM2 = T_xyz(2,:)*radianceWattsPerM2Sr;
chromaticityXY = T_xyz(1:2,:)*radianceWattsPerM2Sr/sum(T_xyz*radianceWattsPerM2Sr);

%% Load cone spectral sensitivities
load T_cones_ss2
T_cones = SplineCmf(S_cones_ss2,T_cones_ss2,S);

%% Compute irradiance, trolands, etc.
degPerMm = RetinalMMToDegrees(1,eyeLengthMm);
irradianceWattsPerUm2 = RadianceToRetIrradiance(radianceWattsPerM2Sr,S,pupilAreaMm2,eyeLengthMm);
irradianceScotTrolands = RetIrradianceToTrolands(irradianceWattsPerUm2, S, 'Scotopic', [], num2str(eyeLengthMm));
irradiancePhotTrolands = RetIrradianceToTrolands(irradianceWattsPerUm2, S, 'Photopic', [], num2str(eyeLengthMm));
irradianceQuantaPerUm2Sec = EnergyToQuanta(S,irradianceWattsPerUm2);
irradianceWattsPerCm2 = (10.^8)*irradianceWattsPerUm2;
irradianceQuantaPerCm2Sec = (10.^8)*irradianceQuantaPerUm2Sec;
irradianceQuantaPerDeg2Sec = (degPerMm^2)*(10.^-2)*irradianceQuantaPerCm2Sec;

%% Get trolands another way.  For scotopic trolands, this just uses scotopic vlambda (in PTB as T_rods)
% and the magic factor of 1700 scotopic lumens per Watt from Wyszecki & Stiles (2cd edition),
% p. 257.  (This is the analog of 683 photopic lumens per Watt.  Then apply formula from
% page 103 of same book.
%
% Same idea for photopic trolands, although we already have luminance in cd/m2 from above so
% we can short cut a little.
%
% The agreement is good to integer scotopic trolands and I'm will to write off the rest
% as round off error.
%
% There must be an eye length implicit in this calculation.  What is it?
load T_rods
T_scotopicVlambda = SplineCmf(S_rods,T_rods,S);
irradianceScotTrolands_check = pupilAreaMm2*1700*(T_scotopicVlambda*radianceWattsPerM2Sr);
irradiancePhotTrolands_check = pupilAreaMm2*photopicLuminanceCdM2;

%% Get cone coordinates from radiance, and also adjust by pupil area.
% Useful for comparing to light levels produced by monochromatic lights
% in other papers.
%
% Should really get these in isomerizations per cone per second.
theLMS = T_cones*radianceWattsPerM2Sr;
theLMSTimesPupilArea = pupilAreaMm2*theLMS;

%% Compute irradiance arriving at cornea
%
% According to OSA Handbook of Optics, 2cd Edition, Chaper 24 (vol 2), pp. 24.13-24.15, the
% conversion is (assuming some approximations), irradiance = radiance*stimulusArea/distance^2.
% This is implemented in RadianceAndDistanceAreaToCornIrradiance
stimulusRadiusMm = 6;
stimulusDistanceMm = 25;
stimulusRadiusM = stimulusRadiusMm/1000;
stimulusAreaM2 = pi*(stimulusRadiusM^2);
stimulusDistanceM = stimulusDistanceMm/1000;
stimulusRadiusDeg = rad2deg(stimulusRadiusMm/stimulusDistanceMm);
stimulusAreaDegrees2 = pi*(stimulusRadiusDeg^2);
cornealIrradianceWattsPerM2 = RadianceAndDistanceAreaToCornIrradiance(radianceWattsPerM2Sr,stimulusDistanceM,stimulusAreaM2);
cornealIrradianceWattsPerCm2 = (10.^-4)*cornealIrradianceWattsPerM2;
cornealIrradianceQuantaPerCm2Sec = EnergyToQuanta(S,cornealIrradianceWattsPerCm2);

%% Report on stimulus
fprintf('\n');
fprintf('  * Assuming pupil diameter %d mm and eye length %d mm\n',pupilDiamMm,eyeLengthMm);
fprintf('  * One mm is %0.2f degrees\n',MmToDeg);
fprintf('  * Input monochromatic stimulus retinal irradiance %0.1f log10 watts/m2 at %d nm\n', ...
    log10(theMonochromaticRetIrradianceWattsPerM2),theMonochromaticWl);
fprintf('  * Input monochromatic stimulus retinal irradiance %0.1f log10 watts/cm2 at %d nm\n', ...
    log10(theMonochromaticRetIrradianceWattsPerM2/1e4),theMonochromaticWl);
fprintf('  * Input monochromatic stimulus retinal irradiance %0.1f log10 watts/mm2 at %d nm\n', ...
    log10(sum(theInputRetIrradianceWattsPerMm2)),theMonochromaticWl);
fprintf('  * Input monochromatic stimulus retinal irradiance %0.1f log10 watts/deg2 at %d nm\n', ...
    log10(sum(theInputRetIrradianceWattsPerDeg2)),theMonochromaticWl);
fprintf('  * Stimulus diameter mm %0.1f, degrees %0.1f\n',2*stimulusRadiusMm,2*stimulusRadiusDeg);
fprintf('  * Stimulus radiance %0.1f log10 watts/[m2-sr], %0.1f log10 watts/[cm2-sr]\n',log10(sum(radianceWattsPerM2Sr)),log10(sum(radianceWattsPerCm2Sr)));
fprintf('  * Stimulus luminance %0.1f candelas/m2\n',photopicLuminanceCdM2);
fprintf('  * Stimulus chromaticity x=%0.4f, y=%0.4f\n',chromaticityXY(1), chromaticityXY(2));
fprintf('  * Stimulus %0.0f (check val %0.0f) scotopic trolands, %0.0f photopic trolands (check val %0.0f)\n',irradianceScotTrolands,irradianceScotTrolands_check,...
    irradiancePhotTrolands,irradiancePhotTrolands_check);
fprintf('  * Stimulus %0.1f log10 scotopic trolands, %0.1f log10 photopic trolands\n',log10(irradianceScotTrolands),log10(irradiancePhotTrolands));
fprintf('  * Stimulus retinal irradiance %0.1f log10 watts/cm2\n',log10(sum(irradianceWattsPerCm2)));
fprintf('  * Stimulus retinal irradiance %0.1f log10 quanta/[cm2-sec]\n',log10(sum(irradianceQuantaPerCm2Sec)));
fprintf('  * Stimulus retinal irradiance %0.1f log10 quanta/[deg2-sec]\n',log10(sum(irradianceQuantaPerDeg2Sec)));
fprintf('  * Stimulus corneal irradiance %0.1f log10 watts/cm2\n',log10(sum(cornealIrradianceWattsPerCm2)));
fprintf('  * Stimulus corneal irradiance %0.1f log10 quanta/[cm2-sec]\n',log10(sum(cornealIrradianceQuantaPerCm2Sec)));
fprintf('  * Pupil area times LMS: %0.2f, %0.2f, %0.2f\n',...
        theLMSTimesPupilArea(1),theLMSTimesPupilArea(2),theLMSTimesPupilArea(3));


