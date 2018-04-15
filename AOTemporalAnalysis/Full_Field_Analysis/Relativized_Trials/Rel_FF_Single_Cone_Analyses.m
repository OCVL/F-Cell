% function [fitCharacteristics]=Rel_FF_Single_Cone_Analyses(stimRootDir, controlRootDir)
% [fitCharacteristics]=Rel_FF_Single_Cone_Analyses(stimRootDir, controlRootDir)
%
%   Calculates pooled variance across a set of pre-analyzed 
%   signals from a single cone's stimulus and control trials, performs 
%   the subtraction between its standard deviations, and performs a
%   piecewise fit of the subtraction.
%
%   This script is designed to work with FULL FIELD datasets- that is, each
%   dataset (mat file) contains *only* control or stimulus data.
%
%   Normally, the user doesn't need to select a stimulus or control root
%   directory (that will be found automatically by
%   "FF_Aggregate_Multi_Trial_Run.m"), but if the software is run by
%   itself it will prompt the user for the folders containing the
%   pre-analyzed mat files generated by Rel_FF_Temporal_Reflectivity_Analysis.m.
%
% Inputs:
%       stimRootDir: The folder path of the pre-analyzed (.mat) stimulus
%       trials. Each mat file must contain valid stimulus signals.
%
%       controlRootDir: The folder path of the pre-analyzed (.mat) control
%       trials. Each mat file must contain valid control signals.
%
%
% Outputs:
%       fitCharacteristics: Information extracted from the mat files and
%       the fitted subtracted signal.
%
% Created by Robert F Cooper 2017-10-31
%

clear;


% load('lowest_responders.mat');

CUTOFF = 26;
NUMTRIALS=20;
CRITICAL_REGION = 66:100;

CELL_OF_INTEREST = [];

if isempty(CELL_OF_INTEREST)
    close all force;
end

if ~exist('stimRootDir','var')
    stimRootDir = uigetdir(pwd, 'Select the directory containing the stimulus profiles');
    controlRootDir = uigetdir(pwd, 'Select the directory containing the control profiles');
end

profileSDataNames = read_folder_contents(stimRootDir,'mat');
profileCDataNames = read_folder_contents(controlRootDir,'mat');


% For structure:
% /stuff/id/date/wavelength/time/intensity/location/data/Profile_Data

[remain kid] = getparent(stimRootDir); % data
[remain stim_loc] = getparent(remain); % location 
[remain stim_intensity] = getparent(remain); % intensity 
[remain stim_time] = getparent(remain); % time
[remain stimwave] = getparent(remain); % wavelength
% [remain sessiondate] = getparent(remain); % date
[~, id] = getparent(remain); % id


%% Code for determining variance across all signals at given timepoint

THEwaitbar = waitbar(0,'Loading stimulus profiles...');

max_index=0;

load(fullfile(stimRootDir, profileSDataNames{1}));
stim_coords = ref_coords;

if ~isempty(CELL_OF_INTEREST)
    stim_cell_reflectance = cell(length(profileSDataNames),1);
end
stim_cell_reflectance_nonorm = cell(length(profileSDataNames),1);
stim_time_indexes = cell(length(profileSDataNames),1);
stim_cell_prestim_mean = cell(length(profileSDataNames),1);

for j=1:length(profileSDataNames)

    waitbar(j/length(profileSDataNames),THEwaitbar,'Loading stimulus profiles...');
    
    ref_coords=[];
    profileSDataNames{j}
    load(fullfile(stimRootDir,profileSDataNames{j}));
    
    if ~isempty(CELL_OF_INTEREST)
        stim_cell_reflectance_nonorm{j} = cell_reflectance;
    end
    stim_cell_reflectance{j} = norm_cell_reflectance;
    stim_time_indexes{j} = cell_times;
    stim_cell_prestim_mean{j} = cell_prestim_mean;
    
    thesecoords = union(stim_coords, ref_coords,'rows');
    
    % These all must be the same length! (Same coordinate set)
    if size(ref_coords,1) ~= size(thesecoords,1)
        error('Coordinate lists different between mat files in this directory. Unable to perform analysis.')
    end
    
    for k=1:length(cell_times)
        max_index = max([max_index max(cell_times{k})]);
    end
    
end

%%
if ~isempty(CELL_OF_INTEREST)
    control_cell_reflectance = cell(length(profileCDataNames),1);
end
control_cell_reflectance_nonorm = cell(length(profileCDataNames),1);
control_time_indexes = cell(length(profileCDataNames),1);
control_cell_prestim_mean = cell(length(profileCDataNames),1);

load(fullfile(controlRootDir, profileCDataNames{1}));
control_coords = ref_coords;

for j=1:length(profileCDataNames)

    waitbar(j/length(profileCDataNames),THEwaitbar,'Loading control profiles...');
    
    ref_coords=[];
    profileCDataNames{j}
    load(fullfile(controlRootDir,profileCDataNames{j}));
    
    if ~isempty(CELL_OF_INTEREST)
        control_cell_reflectance_nonorm{j} = cell_reflectance;
    end
    control_cell_reflectance{j} = norm_cell_reflectance;
    control_time_indexes{j} = cell_times;
    control_cell_prestim_mean{j} = cell_prestim_mean;

    thesecoords = union(control_coords, ref_coords,'rows');
    
    % The length of the cell reflectance lists *must* be the same, because the
    % coordinate lists *must* be the same in each mat file.
    if size(ref_coords,1) ~= size(thesecoords,1)
        error('Coordinate lists different between mat files in this directory. Unable to perform analysis.')
    end
    
    for k=1:length(cell_times)
        max_index = max([max_index max(cell_times{k})]);
    end
    
end

%% The coordinate lists must the same length,
% otherwise it's not likely they're from the same set.

if size(stim_coords,1) ~= size(control_coords,1)
    error('Coordinate lists different between control and stimulus directories. Unable to perform analysis.')
end

allcoords = stim_coords;


%% Aggregation of all trials

percentparula = parula(101);

numstimcoords = size(stim_coords,1);

stim_cell_var = nan(numstimcoords, max_index);
stim_cell_median = nan(numstimcoords, max_index);
stim_cell_pca_std = nan(numstimcoords, 1);
stim_trial_count = zeros(numstimcoords,1);
stim_posnegratio = nan(numstimcoords,max_index);
stim_prestim_means=[];

i=1;


        
for i=1:numstimcoords
    waitbar(i/size(stim_coords,1),THEwaitbar,'Processing stimulus signals...');

    numtrials = 0;
    all_times_ref = nan(length(profileSDataNames), max_index);
    if ~isempty(CELL_OF_INTEREST)
        nonorm_ref = nan(length(profileSDataNames), max_index);
    end
    for j=1:length(profileSDataNames)
        
        if ~isempty(stim_cell_reflectance{j}{i}) && ...
           sum(stim_time_indexes{j}{i} >= 67 & stim_time_indexes{j}{i} <=99) >= CUTOFF

            stim_prestim_means = [stim_prestim_means; stim_cell_prestim_mean{j}(i)];

            numtrials = numtrials+1;
            if ~isempty(CELL_OF_INTEREST)
                nonorm_ref(j, stim_time_indexes{j}{i} ) = stim_cell_reflectance_nonorm{j}{i}(~isnan(stim_cell_reflectance_nonorm{j}{i}));
            end
            all_times_ref(j, stim_time_indexes{j}{i} ) = stim_cell_reflectance{j}{i};
        end
    end 
    stim_trial_count(i) = numtrials;
    
    
    nonan_ref = all_times_ref(~all(isnan(all_times_ref),2), :);
    
    for j=1:max_index
        nonan_ref = all_times_ref(~isnan(all_times_ref(:,j)), j);
        refcount = sum(~isnan(all_times_ref(:,j)));
        refmedian = median(nonan_ref);
        if ~isnan(refmedian)
            stim_cell_median(i,j) = double(median(nonan_ref));
            stim_cell_var(i,j) = ( sum((nonan_ref-mean(nonan_ref)).^2)./ (refcount-1) );
        end
    end
    
    if i==CELL_OF_INTEREST 
        figure(1); clf;
        subplot(3,1,1); plot( bsxfun(@minus,nonorm_ref, nonorm_ref(:,2))');axis([2 134 -120 120]);
        subplot(3,1,2); plot(all_times_ref');  axis([2 134 -8 8]);       
        subplot(3,1,3); plot(stim_cell_median(i,:)); hold on;
                        plot(sqrt(stim_cell_var(i,:))); hold off; axis([2 134 -3 5]);
        title(['Cell #:' num2str(i)]);
        drawnow;
        saveas(gcf, ['Cell_' num2str(i) '_stimulus.svg']);
        
        figure(2); imagesc(ref_image); colormap gray; axis image;hold on; 
        plot(ref_coords(i,1),ref_coords(i,2),'r*'); hold off;
        saveas(gcf, ['Cell_' num2str(i) '_location.svg']);
        drawnow;
        
        
%         pause;
    end

end


%%
numcontrolcoords = size(control_coords,1);

control_cell_var = nan(size(control_coords,1), max_index);
control_cell_median = nan(size(control_coords,1), max_index);
control_cell_pca_std = nan(size(stim_coords,1), 1);
control_trial_count = zeros(size(control_coords,1),1);
control_posnegratio = nan(size(control_coords,1),max_index);
cont_prestim_means=[];


for i=1:numcontrolcoords
    waitbar(i/size(control_coords,1),THEwaitbar,'Processing control signals...');

    numtrials = 0;
    all_times_ref = nan(length(profileCDataNames), max_index);
    if ~isempty(CELL_OF_INTEREST)
        nonorm_ref = nan(length(profileCDataNames), max_index);
    end
    for j=1:length(profileCDataNames)
                        
        if ~isempty(control_cell_reflectance{j}{i}) && ...
           sum(control_time_indexes{j}{i} >= 67 & control_time_indexes{j}{i} <=99) >=  CUTOFF
       
            cont_prestim_means = [cont_prestim_means; control_cell_prestim_mean{j}(i)];
            numtrials = numtrials+1;
            if ~isempty(CELL_OF_INTEREST)
                nonorm_ref(j, control_time_indexes{j}{i} ) = control_cell_reflectance_nonorm{j}{i}(~isnan(control_cell_reflectance_nonorm{j}{i}));
            end
            all_times_ref(j, control_time_indexes{j}{i} ) = control_cell_reflectance{j}{i};
        end
    end
    control_trial_count(i) = numtrials;
    

    for j=1:max_index
        nonan_ref = all_times_ref(~isnan(all_times_ref(:,j)), j);
        refcount = sum(~isnan(all_times_ref(:,j)));
        refmedian = median(nonan_ref);
        if ~isnan(refmedian)
            control_cell_median(i,j) = refmedian;
            control_cell_var(i,j) = ( sum((nonan_ref-mean(nonan_ref)).^2)./ (refcount-1) );
        end
    end
    
    if i==CELL_OF_INTEREST 
        figure(1); clf;
        subplot(3,1,1); plot(bsxfun(@minus,nonorm_ref, nonorm_ref(:,2))'); axis([2 134 -120 120]);
        subplot(3,1,2); plot(all_times_ref'); axis([2 134 -8 8]);       
        subplot(3,1,3); plot(control_cell_median(i,:)); hold on;
                        plot(sqrt(control_cell_var(i,:))); hold off; axis([2 134 -3 5]);
        title(['Cell #:' num2str(i)]);                
        drawnow;
        saveas(gcf, ['Cell_' num2str(i) '_control.svg']);
%         pause;
    end
end


% figure;
% histogram(stim_prestim_means, 255); hold on; histogram(cont_prestim_means, 255);
% numover = sum(stim_prestim_means>200) + sum(cont_prestim_means>200);
% title(['Pre-stimulus means of all available trials (max 50) from ' num2str(size(control_coords,1)) ' cones. ' num2str(numover) ' trials >200 ']);


valid = (stim_trial_count >= NUMTRIALS) & (control_trial_count >= NUMTRIALS);

% Calculate the pooled std deviation
std_dev_sub = sqrt(stim_cell_var)-sqrt(control_cell_var);
median_sub = stim_cell_median-control_cell_median;
%%

if ~isempty(CELL_OF_INTEREST )
    figure(3);clf;
    plot(median_sub(CELL_OF_INTEREST,:));  hold on;
    plot(std_dev_sub(CELL_OF_INTEREST,:));
    axis([2 134 -3 3]);
    title(['Cell #:' num2str(CELL_OF_INTEREST)]);  
    drawnow;
    saveas(gcf, ['Cell_' num2str(CELL_OF_INTEREST) '_subs.svg']);
end
%% Calculate PCA on the crtiical area of the signals
critical_nonnan_ref = sqrt(stim_cell_var(:,CRITICAL_REGION));
[std_dev_coeff, std_dev_score, std_dev_latent, tquare, std_dev_explained, std_dev_mu]=pca(critical_nonnan_ref);


critical_nonnan_ref = stim_cell_median(:,CRITICAL_REGION);
[median_coeff, median_score, latent, tquare, explained]=pca(critical_nonnan_ref);

timeBase = ((1:max_index-1)/16.6)';

%% Analyze the signals

fitAmp = nan(size(std_dev_sub,1),1);
fitMedian = nan(size(std_dev_sub,1),1);


for i=1:size(std_dev_sub,1)
waitbar(i/size(std_dev_sub,1),THEwaitbar,'Analyzing subtracted signals...');

    std_dev_sig = std_dev_sub(i,:);
    median_sig = median_sub(i,:);

    if ~all( isnan(std_dev_sig) ) && (stim_trial_count(i) >= NUMTRIALS) && (control_trial_count(i) >= NUMTRIALS)

        % AUC        
        fitAmp(i) = sum(std_dev_sig(CRITICAL_REGION));
        fitMedian(i) = sum(median_sig(CRITICAL_REGION));

    end
end
close(THEwaitbar);
%%
save([ stim_intensity '.mat'],'fitAmp','fitMedian','std_dev_coeff','valid',...
     'median_coeff','std_dev_score','median_score','allcoords','ref_image','control_cell_median',...
     'control_cell_var','stim_cell_median','stim_cell_var');

 

%% Plot the pos/neg ratio of the mean vs the amplitude
posnegratio=nan(size(control_coords,1),1);


figure(101); clf; hold on;
for i=1:size(control_coords,1)
    if ~isnan(fitAmp(i))
        % Find out what percentage of time the signal spends negative
        % or positive after stimulus delivery (66th frame)
%         numposneg = sign(mean_sub(i,:));
%         pos = sum(numposneg == 1);
% 
%         posnegratio(i) = 100*pos/length(numposneg);

        plot( fitAmp(i), fitMedian(i),'k.');        
    end
end
ylabel('Median response amplitude');
xlabel('Reflectance response amplitude');
title('Median reflectance vs reflectance response amplitude')
hold off;
saveas(gcf,['posneg_vs_amp_' num2str(stim_intensity) '.png']);
%% Plot histograms of the amplitudes
figure(7);
histogram( fitAmp(~isnan(fitAmp)) ,'Binwidth',0.1);
title('Stim-Control per cone subtraction amplitudes');
xlabel('Amplitude difference from control');
ylabel('Number of cones');

%% TEMP to prove control equivalence!
% stim_resp = sum(sqrt(stim_cell_var(:,critical_region)),2) + abs(sum(stim_cell_median(:,critical_region),2));
% control_resp = sum(sqrt(control_cell_var(:,critical_region)),2) + abs(sum(control_cell_median(:,critical_region),2));
% 
% figure(8); plot(stim_resp,control_resp,'k.'); hold on;
% plot([25 90],[25 90],'k'); hold off; axis square;
% ylabel('450nW control response');
% xlabel('0nW control response');
