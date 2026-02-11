function [allLum, allPhotTd, allScotTd] = AOLightLevelConversions_Func(stim_side, wavelengths, um_power_to_eye,verbose)
%
% Take monochromatic retinal irradiance and convert it to many other
% equivalent units.
%
% 10/29/15  dhb  Wrote (again, since I lost the first version) from OLLightLevelCheck


%% Define the wavelength spacing that we will work with
S = [380 1 700];
if (S(2) ~= 1)
    error('We are assuming 1 nm spacing');
end

if length(wavelengths) ~= length(um_power_to_eye)
    error('List of wavelengths and power must be the same!');
end

for w=1:length(wavelengths)
    
    theWavelength = wavelengths(w);
    pupilDiamMm = 7;  % 7mm typical
    eyeLengthMm = 17; % 17mm - lens to retina typical
    
    rawStimulusSideDegIn = stim_side;
    stimulusAreaDegrees2 = rawStimulusSideDegIn^2;
    
    % Utility calculations
    pupilAreaMm2 = pi*((pupilDiamMm/2)^2);
    pupilAreaCm2 = pupilAreaMm2*(10^-2);
    eyeLengthCm = eyeLengthMm*(10^-1);
    degPerMm = RetinalMMToDegrees(1,eyeLengthMm);
    
    rawPowerIntoEyeIn = um_power_to_eye(w);
    rawCornIrradianceMicrowattsPerCm2In = rawPowerIntoEyeIn./pupilAreaCm2;
    rawRadianceMicrowattsPerCm2Sr = CornIrradianceAndDegrees2ToRadiance(rawCornIrradianceMicrowattsPerCm2In,stimulusAreaDegrees2);
    rawRetIrradianceMicrowattsPerCm2In = RadianceAndPupilAreaEyeLengthToRetIrradiance(rawRadianceMicrowattsPerCm2Sr,S,pupilAreaCm2,eyeLengthCm);
    
    %% Given that we have retinal irradiance, corneal irradiance, the pupil area, and the stimulus area ...
    % it is easy to check the retinal irradiance.
    retIrradianceMicrowattsPerDeg2Check = rawCornIrradianceMicrowattsPerCm2In*pupilAreaCm2/stimulusAreaDegrees2;
    retIrradianceMicrowattsPerCm2Check = retIrradianceMicrowattsPerDeg2Check/(0.01*DegreesToRetinalMM(1,eyeLengthCm*10)^2);
    if (abs(rawRetIrradianceMicrowattsPerCm2In-retIrradianceMicrowattsPerCm2Check)/retIrradianceMicrowattsPerCm2Check > 0.05)
        error('Do not get same retinal irradiance two different ways');
    end
    
    %% Turn it into a spectral function
    wls = SToWls(S);
    retIrradianceMicrowattsPerCm2In = zeros(size(wls));
    index = find(wls == theWavelength);
    if (length(index) ~= 1)
        error('Specified wavelength is not in sampled wavelengths');
    end
    retIrradianceMicrowattsPerCm2In(index) = rawRetIrradianceMicrowattsPerCm2In;
    
    %% Convert to some other units for convenience
    retIrradianceMilliwattsPerCm2In = retIrradianceMicrowattsPerCm2In*(10^-3);
    retIrradianceWattsPerCm2In = retIrradianceMilliwattsPerCm2In*(10^-3);
    retIrradianceWattsPerUm2In = retIrradianceWattsPerCm2In*(10^-8);
    
    %% Load CIE functions.
    load T_xyz1931
    T_xyz = SplineCmf(S_xyz1931,683*T_xyz1931,S);
    
    %% Load cone spectral sensitivities
    load T_cones_ss2
    T_cones = SplineCmf(S_cones_ss2,T_cones_ss2,S);
    
    %% Load in a directly measured sunlight through window
    % and off piece of white paper towel on floor for comparison.
    % Surely that is safe to look at.
    load spd_phillybright
    spd_phillybright = SplineSpd(S_phillybright,spd_phillybright,S);
    photopicLuminancePhillyBrightCdM2 = T_xyz(2,:)*spd_phillybright;
    
    %% Compute irradiance to equivalent radiance
    %
    % This conversion depends on pupil area and eye length
    
    % Do the conversion
    radianceMilliwattsPerCm2Sr = RetIrradianceAndPupilAreaEyeLengthToRadiance(retIrradianceMilliwattsPerCm2In,S,pupilAreaCm2,eyeLengthCm);
    radianceMicrowattsPerCm2Sr = radianceMilliwattsPerCm2Sr*(10^3);
    radianceWattsPerCm2Sr = radianceMilliwattsPerCm2Sr*(10^-3);
    radianceWattsPerM2Sr = radianceWattsPerCm2Sr*(10^4);
    
    % Check by converting back using a different routine
    retIrradianceWattsPerUm2 = RadianceToRetIrradiance(radianceWattsPerM2Sr,S,pupilAreaMm2,eyeLengthMm);
    retIrradianceWattsPerM2 = retIrradianceWattsPerUm2*(10^12);
    retIrradianceWattsPerCm2 = retIrradianceWattsPerUm2*(10.^8);
    retIrradianceMilliwattsPerCm2 = retIrradianceWattsPerCm2*(10^3);
    retIrradianceMicrowattsPerCm2 = retIrradianceMilliwattsPerCm2*(10^3);
    tolerance = 1e-8;
    if (max(abs(retIrradianceMicrowattsPerCm2(:)-retIrradianceMicrowattsPerCm2In(:))) > tolerance)
        error('Failure to go there and back');
    end
    
    %% Conver retinal irradiance to trolands, etc.
    irradianceScotTrolands = RetIrradianceToTrolands(retIrradianceWattsPerUm2, S, 'Scotopic', [], num2str(eyeLengthMm));
    irradiancePhotTrolands = RetIrradianceToTrolands(retIrradianceWattsPerUm2, S, 'Photopic', [], num2str(eyeLengthMm));
    retIrradianceQuantaPerUm2Sec = EnergyToQuanta(S,retIrradianceWattsPerUm2);
    retIrradianceQuantaPerCm2Sec = (10.^8)*retIrradianceQuantaPerUm2Sec;
    retIrradianceQuantaPerDeg2Sec = (degPerMm^2)*(10.^-2)*retIrradianceQuantaPerCm2Sec;
    
    % Yet more units
    photopicLuminanceCdM2 = T_xyz(2,:)*radianceWattsPerM2Sr;
    
    %% Pupil adjustment factor for Ansi MPE
    mpePupilDiamMm = 3;
    % mpePupilDiamMm  = GetWithDefault('Enter ANSI 2007 MPE caclulations assumed pupil diameter in mm',mpePupilDiamMm );
    pupilAdjustFactor = (pupilDiamMm/mpePupilDiamMm).^2;
    
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
    load T_rods
    T_scotopicVlambda = SplineCmf(S_rods,T_rods,S);
    irradianceScotTrolands_check = pupilAreaMm2*1700*(T_scotopicVlambda*radianceWattsPerM2Sr);
    irradiancePhotTrolands_check = pupilAreaMm2*photopicLuminanceCdM2;
    
    %% Compute corneal irradiance
    cornealIrradianceMicrowattsPerCm2 = RadianceAndDegrees2ToCornIrradiance(radianceMicrowattsPerCm2Sr,stimulusAreaDegrees2);
    
    cornealIrradianceWattsPerM2 = cornealIrradianceMicrowattsPerCm2*(10^-6)*(10^4);
    cornealIrradianceWattsPerCm2 = (10.^-4)*cornealIrradianceWattsPerM2;
    cornealIrradianceQuantaPerCm2Sec = EnergyToQuanta(S,cornealIrradianceWattsPerCm2);
    
    
    %% Report on stimulus
    if verbose
        fprintf('\n\n');
        fprintf('  *** Analyzing light levels for wavelength: %0.1fnm and power: %0.1f uW/degree2: \n',wavelengths(w),um_power_to_eye(w));
        fprintf('  * Stimulus radiance %0.1f log10 watts/[m2-sr], %0.1f log10 watts/[cm2-sr]\n',log10(sum(radianceWattsPerM2Sr)),log10(sum(radianceWattsPerCm2Sr)));
        fprintf('  * Stimulus luminance %0.1f candelas/m2\n',photopicLuminanceCdM2);
        fprintf('    * For comparison, sunlight in Philly: %0.1f cd/m2\n',photopicLuminancePhillyBrightCdM2);
        fprintf('  * Stimulus %0.0f (check val %0.0f) scotopic trolands, %0.0f photopic trolands (check val %0.0f)\n',irradianceScotTrolands,irradianceScotTrolands_check,...
            irradiancePhotTrolands,irradiancePhotTrolands_check);
        fprintf('  * Stimulus %0.1f log10 scotopic trolands, %0.1f log10 photopic trolands\n',log10(irradianceScotTrolands),log10(irradiancePhotTrolands));
        fprintf('  * Stimulus retinal irradiance %0.1f log10 watts/cm2\n',log10(sum(retIrradianceWattsPerCm2)));
        fprintf('  * Stimulus retinal irradiance %0.2e quanta/[um2-sec]\n',sum(retIrradianceQuantaPerCm2Sec)/(1000*1000));
        fprintf('  * Stimulus retinal irradiance %0.1f log10 quanta/[deg2-sec]\n',log10(sum(retIrradianceQuantaPerDeg2Sec)));
        fprintf('  * Stimulus corneal irradiance %0.1f log10 watts/cm2\n',log10(sum(cornealIrradianceWattsPerCm2)));
        fprintf('  * Stimulus corneal irradiance %0.1f log10 quanta/[cm2-sec]\n',log10(sum(cornealIrradianceQuantaPerCm2Sec)));
    end
    allLum(w) = photopicLuminanceCdM2;
    allPhotTd(w) = irradiancePhotTrolands;
    allScotTd(w) = irradianceScotTrolands;
    
end

end

