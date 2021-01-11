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
%   directory, but if the software is run by
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


CUTOFF = 10;
NUMTRIALS= 25;
CRITICAL_REGION = 72:92; 
window_size = 5;
half_window = floor(window_size/2);

pipeline_type = 'moving_rms'; % Legacy mode is 'stddev', moving_rms is new

CELL_OF_INTEREST = [5];%[31 83 198 599 746 790 794 824 842 1083 1255 1277];
% CELL_OF_INTEREST = [7 26 44 54 62 82 96 132 162 169 195 236 261 168 324 370 377 404 412 438 439 449 507 565 572 594 609 660 784 786 809 830 858 873 902];
% CELL_OF_INTEREST = [26,36,96,132,195,211,236,261,268,313,317,377,404,409,412,428,438,439,534,578,783,784,809,818,858,913,937,949,991,1014,1052,1158,1253,1338,1398,1432,1448];

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
outFname = [id '_' stimwave '_' stim_intensity '_' stim_time  '_' stim_loc '_' num2str(length(profileSDataNames)) '_single_cone_signals'];


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
    
    
    stim_cell_reflectance_nonorm{j} = cell_reflectance;
    
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


% Aggregation of all trials
%% Control trial aggregation

numcontrolcoords = size(control_coords,1);

control_cell_var = nan(size(control_coords,1), max_index);
control_cell_median = nan(size(control_coords,1), max_index);
control_cell_pca_std = nan(size(stim_coords,1), 1);
control_trial_count = zeros(size(control_coords,1),1);
control_posnegratio = nan(size(control_coords,1),max_index);
cont_prestim_means=[];

allcontrolsignals=cell(numcontrolcoords,1);


for i=1:numcontrolcoords
    waitbar(i/size(control_coords,1),THEwaitbar,'Processing control signals...');

    numtrials = 0;
    all_times_ref = nan(length(profileCDataNames), max_index);
    if ~isempty(CELL_OF_INTEREST)
        nonorm_ref = nan(length(profileCDataNames), max_index);
    end
    for j=1:length(profileCDataNames)
                        
        if ~isempty(control_cell_reflectance{j}{i}) && ...
           sum(control_time_indexes{j}{i} >= CRITICAL_REGION(1) & control_time_indexes{j}{i} <= CRITICAL_REGION(end)) >=  CUTOFF && ...
           control_cell_prestim_mean{j}(i) <= 240
       
            cont_prestim_means = [cont_prestim_means; control_cell_prestim_mean{j}(i)];
            numtrials = numtrials+1;
            if ~isempty(CELL_OF_INTEREST)
                nonorm_ref(j, control_time_indexes{j}{i} ) = control_cell_reflectance_nonorm{j}{i}(~isnan(control_cell_reflectance_nonorm{j}{i}));
            end
            all_times_ref(j, control_time_indexes{j}{i} ) = control_cell_reflectance{j}{i};
        end
    end
    control_trial_count(i) = numtrials;
    allcontrolsignals{i} = all_times_ref;
    
    if strcmp(pipeline_type,'stddev')
        % Standard form
        for j=1:max_index
            nonan_ref = all_times_ref(~isnan(all_times_ref(:,j)), j);

            refcount = sum(~isnan(all_times_ref(:,j)));
            refmedian = median(nonan_ref);
            if ~isnan(refmedian)
                control_cell_median(i,j) = refmedian;
                control_cell_var(i,j) = ( sum((nonan_ref-mean(nonan_ref)).^2)./ (refcount-1) );
            end
        end
    elseif strcmp(pipeline_type,'moving_rms')
        % Moving-RMS form
        padded_all_times_ref = padarray(all_times_ref, [0 half_window], 'symmetric','both');
        for j=1:size(padded_all_times_ref,2)-window_size
            window =  padded_all_times_ref(:, j:j+window_size-1);
            refcount = sum(~isnan(all_times_ref(:,j)));
            
            refmedian = median(window(:),'omitnan');
            if ~isnan(refmedian)
                control_cell_median(i,j) = refmedian;
                control_cell_var(i,j) = rms(window(~isnan(window)));
            end
        end
    end
    
    
%     if any(i==CELL_OF_INTEREST) %&& control_trial_count(i)>CUTOFF
%         figure(2); clf;
%         subplot(3,1,1); plot(bsxfun(@minus,nonorm_ref, nonorm_ref(:,2))');%axis([CRITICAL_REGION(1) CRITICAL_REGION(end) -150 150]);  xlabel('Time index'); ylabel('Raw Response');
%         subplot(3,1,2); plot(all_times_ref'); %axis([CRITICAL_REGION(1) CRITICAL_REGION(end) -10 10]); xlabel('Time index'); ylabel('Standardized Response');
%         subplot(3,1,3); plot(control_cell_median(i,:)); hold on;
%                         plot((control_cell_var(i,:))); hold off; %axis([CRITICAL_REGION(1) CRITICAL_REGION(end) -2 10]); xlabel('Time index'); ylabel('Response');
%         title(['Cell #:' num2str(i)]);   
%         drawnow;
% %         critreg=all_times_ref(:,CRITICAL_REGION);
% %         quantile(critreg(:),.95)
% %         quantile(critreg(:),.5)
% %         quantile(critreg(:),.05)
% %         
% % %         saveas(gcf, ['Cell_' num2str(i) '_control.png']);
%         pause;
%     end
end

%% Stimulus trial aggregation
percentparula = parula(101);

numstimcoords = size(stim_coords,1);

stim_cell_var = nan(numstimcoords, max_index);
stim_cell_median = nan(numstimcoords, max_index);
stim_resp_range = nan(numstimcoords, 1);
stim_trial_count = zeros(numstimcoords,1);
stim_posnegratio = nan(numstimcoords,max_index);
stim_prestim_means = nan(numstimcoords,length(profileSDataNames));


i=1;
allstimsignals=cell(numstimcoords,1);

fitops = fitoptions('Method','SmoothingSpline','SmoothingParam',0.999,'Normalize','on');
for i=28:numstimcoords
    waitbar(i/size(stim_coords,1),THEwaitbar,'Processing stimulus signals...');

    
    numtrials = 0;
    all_times_ref = nan(length(profileSDataNames), max_index);
    all_smooth_times_ref = nan(length(profileSDataNames), max_index);
    if ~isempty(CELL_OF_INTEREST)
        nonorm_ref = nan(length(profileSDataNames), max_index);
    end
    
    allnormsignals=[];
    allstims=[];
    

    for j=1:length(profileSDataNames)
        
        if ~isempty(stim_cell_reflectance{j}{i}) && ...
           sum(stim_time_indexes{j}{i} >= CRITICAL_REGION(1) & stim_time_indexes{j}{i} <= CRITICAL_REGION(end)) >= CUTOFF && ...
           stim_cell_prestim_mean{j}(i) <= 240

            stim_prestim_means(i,j) = stim_cell_prestim_mean{j}(i);
            numtrials = numtrials+1;
            
            nonorm_ref(j, stim_time_indexes{j}{i} ) = stim_cell_reflectance_nonorm{j}{i}(~isnan(stim_cell_reflectance_nonorm{j}{i}));
                
%                 figure(10); plot(stim_cell_reflectance{j}{i}); title(num2str(stim_prestim_means(i,j)));
            
            all_times_ref(j, stim_time_indexes{j}{i} ) = stim_cell_reflectance{j}{i};
            
%             f = fit(stim_time_indexes{j}{i}',stim_cell_reflectance{j}{i}','SmoothingSpline',fitops);            
%             all_smooth_times_ref(j, : ) = f(1:max_index);
%             allsignals = [allsignals [stim_time_indexes{j}{i}+(166*(j-1));
%                                       stim_cell_reflectance_nonorm{j}{i}(~isnan(stim_cell_reflectance_nonorm{j}{i}))]];
%             allnormsignals = [allnormsignals [stim_time_indexes{j}{i}+(166*(j-1));
%                                       stim_cell_reflectance{j}{i}]];
%             allstims = [allstims [(72:108)+(166*(j-1));
%                                       ones(1,37)*10]];
        end
        

    end
    allstimsignals{i} = all_times_ref;
    stim_trial_count(i) = numtrials;
    
    critical_region_ref =all_times_ref(:, CRITICAL_REGION);
%     quantrng = quantile(critical_region_ref(:), [0.05 0.95]);
    stim_resp_range(i) = rms( critical_region_ref(~isnan(critical_region_ref(:))) );
    
    if strcmp(pipeline_type,'stddev')
        % Standard form
        for j=1:max_index
            nonan_ref = all_times_ref(~isnan(all_times_ref(:,j)), j);
            refcount = sum(~isnan(all_times_ref(:,j)));
            refmedian = median(nonan_ref); 
            if ~isnan(refmedian)
                stim_cell_median(i,j) = double(median(nonan_ref));

                stim_cell_var(i,j) = ( sum((nonan_ref-mean(nonan_ref)).^2)./ (refcount-1) );
            end
        end
    elseif strcmp(pipeline_type,'moving_rms')
        % Moving-RMS form
        padded_all_times_ref = padarray(all_times_ref, [1 half_window], 'symmetric','both');
        for j=1:size(padded_all_times_ref,2)-window_size
            window =  padded_all_times_ref(:, j:j+window_size-1);
            refcount = sum(~isnan(all_times_ref(:,j)));

            refmedian = median(window(:),'omitnan');
            if ~isnan(refmedian)
                stim_cell_median(i,j) = refmedian;
                stim_cell_var(i,j) = rms(window(~isnan(window)));
            end
        end
    end
    
    if any(i==CELL_OF_INTEREST) %&& stim_trial_count(i)>=CUTOFF && (densitometry_fit_amplitude(i) < 0.1) && valid_densitometry(i)
        figure(1); clf;
        [~, firstinds] = max(~isnan(nonorm_ref),[],2); % Find all of the first non-Nan indexes in our rows.
        subplot(3,1,1); plot( bsxfun(@minus,nonorm_ref, nonorm_ref(sub2ind(size(nonorm_ref),(1:size(nonorm_ref,1))',firstinds)))');%axis([CRITICAL_REGION(1) CRITICAL_REGION(end) -150 150]); xlabel('Time index'); ylabel('Raw Response');        
        subplot(3,1,2); %hold on;
        for m=1:size(all_times_ref,1)
            profileSDataNames{m}
%             goodinds = find(~isnan(all_times_ref(m,:)));
            plot(all_times_ref(m,:));
        end
%         hold off;
        plot(all_times_ref');
        axis([0 180 -15 15]); xlabel('Time index'); ylabel('Standardized Response');
        
        subplot(3,1,3); %plot(stim_cell_median(i,:)); hold on;
                        plot((stim_cell_var(i,:))- (mean(control_cell_var,'omitnan')) ); hold off; %
%                         plot(stim_cell_stddev);  %axis([CRITICAL_REGION(1) CRITICAL_REGION(end) -2 10]); xlabel('Time index'); ylabel('Response');
%         subplot(4,1,4); plot(stim_prestim_means(i,:),'*'); xlabel('Trial #'); ylabel('Prestimulus mean (A.U.)'); axis([0 50 0 255]);
        title(['Cell #:' num2str(i) ' Resp Range: ', num2str(stim_resp_range(i))]);
        drawnow;
%         saveas(gcf, ['Cell_' num2str(i) '_stimulus_rawdata.svg']);
        
%         THEstimref = all_times_ref;
        
        figure(5); imagesc(ref_image); colormap gray; axis image;hold on; 
        plot(ref_coords(i,1),ref_coords(i,2),'r*'); hold off;
%         saveas(gcf, ['Cell_' num2str(i) '_location.png']);
        drawnow;


%         if (densitometry_fit_amplitude(i) < 0.1) && valid_densitometry(i)
           
%            subplot(4,1,4);
%            plot(CRITICAL_TIME/hz, criticalfit(i,:));hold on;
%            plot(densitometry_vect_times{i},densitometry_vect_ref{i},'.')           
%            axis([0 CRITICAL_TIME(end)/hz 0 1.5]); hold off;
%            axis([0 3 0 1.5]);
%            xlabel('Time (s)');
%            ylabel('Reflectance')
%            drawnow; %pause;
%         end
%         saveas(gcf, ['Cell_' num2str(i) '_stimulus_densitometry.png']);
        
%         pause;
    end

end


%%


valid = (stim_trial_count >= NUMTRIALS);
% valid = (stim_trial_count >= NUMTRIALS) & (control_trial_count >= NUMTRIALS);

% Calculate the pooled std deviation
if strcmp(pipeline_type,'stddev')
    std_dev_sub = sqrt(stim_cell_var)-sqrt(mean(control_cell_var,'omitnan'));
    control_std_dev_sub = sqrt(control_cell_var)-sqrt(mean(control_cell_var,'omitnan'));
elseif strcmp(pipeline_type,'moving_rms')
    std_dev_sub = stim_cell_var-mean(control_cell_var,'omitnan');
    control_std_dev_sub = control_cell_var-mean(control_cell_var,'omitnan');
end
median_sub = stim_cell_median-mean(control_cell_median,'omitnan');
control_median_sub = control_cell_median-mean(control_cell_median,'omitnan');


%% Analyze the signals

AmpResp = nan(size(std_dev_sub,1),1);
AUCResp = nan(size(std_dev_sub,1),1);
MedianResp = nan(size(std_dev_sub,1),1);
TTPResp = nan(size(std_dev_sub,1),1);
PrestimVal = nan(size(std_dev_sub,1),1);

ControlAmpResp = nan(size(std_dev_sub,1),1);
ControlMedianResp = nan(size(std_dev_sub,1),1);
ControlPrestimVal = nan(size(std_dev_sub,1),1);
hz=17.85;
allinds = 1:length(std_dev_sub);
for i=1:size(std_dev_sub,1)
    if ishandle(THEwaitbar)
        waitbar(i/size(std_dev_sub,1),THEwaitbar,'Analyzing subtracted signals...');
    end

    if ~all( isnan(std_dev_sub(i,2:end)) ) && valid(i)
        
        % Stimulus with control subtracted
        std_dev_sig = std_dev_sub(i,2:end);        
        nanners = ~isnan(std_dev_sig);
        firstind=find(cumsum(nanners)>0);
        [~, lastind] = max(cumsum(nanners)-sum(nanners));
        interpinds = firstind(1):lastind;
        goodinds = allinds(nanners);
        std_dev_sig = interp1(goodinds, std_dev_sig(nanners), interpinds, 'makima');
        filt_stddev_sig = std_dev_sig;

        median_sig = median_sub(i,2:end);
        nanners = ~isnan(median_sig);
        firstind=find(cumsum(nanners)>0);
        [~, lastind] = max(cumsum(nanners)-sum(nanners));
        interpinds = firstind(1):lastind;
        goodinds = allinds(nanners);
        median_sig = interp1(goodinds, median_sig(nanners), interpinds, 'makima');
        filt_median_sig = median_sig;

        critical_filt = filt_stddev_sig( CRITICAL_REGION );
        
        [~, TTPResp(i)] = max( abs(critical_filt) );    
        AmpResp(i) = quantile(critical_filt,0.95);
        MedianResp(i) = max(abs(filt_median_sig(CRITICAL_REGION))-mean(filt_median_sig(1:CRITICAL_REGION(1))) );        
        PrestimVal(i) = mean( stim_prestim_means(i,:),2, 'omitnan');
        AUCResp(i) = sum(critical_filt);
        
         if any(i==CELL_OF_INTEREST) %(log10(AmpResp(i)+1) > 0.3) && (log10(AmpResp(i)+1) <.45)
           figure(1); clf;
           subplot(3,1,1);
           plot( interpinds/hz, filt_stddev_sig ); hold on; %plot(interpinds/hz,std_dev_sig);
           plot( interpinds/hz,filt_median_sig ); %plot(interpinds/hz, median_sig);
           plot( interpinds/hz, repmat(quantile(critical_filt,0.95),[1 length(interpinds)]), 'r' );
            
%            plot(mean(sqrt(control_cell_var),'omitnan'));
           title(['#: ' num2str(i) ' A: ' num2str(AmpResp(i)) ', M: ' num2str(MedianResp(i)) ', LogResp: ' num2str(log10(AmpResp(i)+1)) ]);
           axis([0 9 -5 10]); hold off;
           
           subplot(3,1,2); 
           plot( allstimsignals{i}','k');hold on;
           plot(stim_cell_var(i,:),'r','LineWidth',3); 
           axis([1 163 -10 25])
           
           subplot(3,1,3);
           plot(allcontrolsignals{i}','k');
            axis([1 163 -10 25])

           drawnow; 
           pause;
%            saveas(gcf, ['Cell_' num2str(i) '_stimulus.png']);
         end
        
        % Control only - REMOVED as control responses are not of interest.
%         std_dev_sig = control_std_dev_sub(i,2:end);
%         nanners = ~isnan(std_dev_sig);
%         firstind=find(cumsum(nanners)>0);
%         [~, lastind] = max(cumsum(nanners)-sum(nanners));
%         interpinds = firstind(1):lastind;
%         goodinds = allinds(nanners);
%         std_dev_sig = interp1(goodinds, std_dev_sig(nanners), interpinds, 'makima');
%         filt_stddev_sig = std_dev_sig;
% 
%         
%         median_sig = control_median_sub(i,2:end);        
%         nanners = ~isnan(median_sig);        
%         firstind=find(cumsum(nanners)>0);
%         [~, lastind] = max(cumsum(nanners)-sum(nanners));
%         interpinds = firstind(1):lastind;
%         goodinds = allinds(nanners);        
%         median_sig = interp1(goodinds, median_sig(nanners), interpinds, 'makima');
%         filt_median_sig = median_sig;
% 
% 
%         critical_filt = filt_stddev_sig( CRITICAL_REGION );
% %         
%         ControlAmpResp(i) = quantile(critical_filt,0.95);
%         ControlMedianResp(i) = max(abs(filt_median_sig(CRITICAL_REGION))-mean(filt_median_sig(1:CRITICAL_REGION(1))) );        
%         ControlPrestimVal(i) = mean( cont_prestim_means(i,:),2, 'omitnan');


    end
end
%%
close(THEwaitbar);

save([ outFname '.mat'],'pipeline_type', 'AmpResp','MedianResp','TTPResp',...
    'valid','allcoords','ref_image','control_cell_median',...
     'control_cell_var','stim_cell_median','stim_cell_var','stim_prestim_means','stim_resp_range','allstimsignals');
%      'ControlAmpResp','ControlMedianResp','ControlPrestimVal',...
     

 

%% Plot the pos/neg ratio of the mean vs the amplitude


figure(101); clf; hold on;
for i=1:size(control_coords,1)
    if ~isnan(AmpResp(i))
        plot(log10( (AUCResp(i)./21)+1), log10(AmpResp(i)+1),'k.');
    end
end
% ylabel('Median response amplitude');
% xlabel('Reflectance response amplitude');
% title('Median reflectance vs reflectance response amplitude')
% hold off;
% saveas(gcf,['posneg_vs_amp_' num2str(stim_intensity) '.png']);
%% Plot histograms of the amplitudes
figure(7);
histogram( log10(AmpResp(~isnan(AmpResp))+1) ,'Binwidth',0.05);
title('Stim-Control per cone subtraction amplitudes');
xlabel('Amplitude difference from control');
ylabel('Number of cones');
%%
% goodtrials = ~isnan(AmpResp);
% 
% figure(8); clf; hold on;
% imagesc(ref_image); colormap gray; axis image;
% for i=1:length(goodtrials)
%     if goodtrials(i) && AmpResp(i)<=0
%         plot(allcoords(i,1), allcoords(i,2),'*');
%     end
% end
