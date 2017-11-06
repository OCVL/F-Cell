function [characteristics]=Aggregate_Analyses_TwoSource_bootstrap(stimRootDir, controlRootDir)
% Robert F Cooper
% 12-31-2015
% This script calculates pooled variance across a set of given signals.

if ~exist('stimRootDir','var')
    close all force;
    stimRootDir = uigetdir(pwd, 'Select the directory containing the stimulus profiles');
    controlRootDir = uigetdir(pwd, 'Select the directory containing the control profiles');
end

profileSDataNames = read_folder_contents(stimRootDir,'mat');
profileCDataNames = read_folder_contents(controlRootDir,'mat');

thatstimmax=0;
thatcontrolmax=0;
%% Code for determining variance across all signals at given timepoint

allmax=249;
num_control_cones = 0;
num_stim_cones =0;
max_cones = 0;
min_cones = 10000000000000;
mean_control_reflectance = zeros(500,1);
num_contributors_control_reflectance = zeros(500,1);

for j=1:length(profileCDataNames)

    profileCDataNames{j}
    load(fullfile(controlRootDir,profileCDataNames{j}));
    
    % Remove empty cells
    norm_control_cell_reflectance = norm_control_cell_reflectance( ~cellfun(@isempty,norm_control_cell_reflectance)  );
    control_cell_times            = control_cell_times( ~cellfun(@isempty,control_cell_times) );

    
    num_control_cones = num_control_cones+length(control_cell_times);
    
    thatcontrolmax = max( cellfun(@max,control_cell_times) );

    % Pooled variance of all cells before first stimulus
    [ ref_variance_control{j},ref_control_times{j}, ref_control_count{j} ] = reflectance_pooled_variance( control_cell_times, norm_control_cell_reflectance, allmax );    


    i=1;
    while i<= length( ref_control_times{j} )

        % Remove times from both stim and control that are NaN, or 0
        if  isnan(ref_control_times{j}(i)) || ref_control_times{j}(i) ==0 || isnan(ref_variance_control{j}(i)) || ref_variance_control{j}(i) ==0

            ref_control_count{j}(i) = [];
            ref_control_times{j}(i) = [];
            ref_variance_control{j}(i) = [];        
        else
            i = i+1;
        end

    end
    precontrol = ref_variance_control{j}(ref_control_times{j}<=66)./ref_control_count{j}(ref_control_times{j}<=66);
    figure(8); plot(ref_control_times{j}/16.6, sqrt(ref_variance_control{j}./ref_control_count{j})-sqrt(mean(precontrol)) ); hold on; drawnow;      
end


hold off;axis([0 16 0 4])

for j=1:length(profileSDataNames)

    profileSDataNames{j}
    load(fullfile(stimRootDir,profileSDataNames{j}));
    
    % Remove the empty cells
    norm_stim_cell_reflectance = norm_stim_cell_reflectance( ~cellfun(@isempty,norm_stim_cell_reflectance) );
    stim_cell_times            = stim_cell_times(  ~cellfun(@isempty,stim_cell_times) );

    num_stim_cones = num_stim_cones+length(stim_cell_times);
    
    thatstimmax = max( cellfun(@max,stim_cell_times) );    
    
    % Pooled variance of all cells before first stimulus
    [ ref_variance_stim{j}, ref_stim_times{j}, ref_stim_count{j} ]    = reflectance_pooled_variance( stim_cell_times, norm_stim_cell_reflectance, allmax );

    i=1;
    while i<= length( ref_stim_times{j} )

        % Remove times from both stim and control that are NaN, or 0
        if isnan(ref_stim_times{j}(i)) || ref_stim_times{j}(i) == 0 || isnan(ref_variance_stim{j}(i)) || ref_variance_stim{j}(i) == 0

            ref_stim_count{j}(i) = [];
            
            ref_stim_times{j}(i) = [];

            ref_variance_stim{j}(i) = [];
      
        else
            i = i+1;
        end

    end
    
    prestim = ref_variance_stim{j}(ref_stim_times{j}<=66)./ref_stim_count{j}(ref_stim_times{j}<=66);
    figure(9); plot(ref_stim_times{j}/16.6, sqrt( ref_variance_stim{j}./ref_stim_count{j})-sqrt(mean(prestim)) ); hold on; drawnow;

end
hold off;axis([0 16 0 4])

% if (length(stim_cell_times) + length(control_cell_times)) < min_cones
%     min_cones = (length(stim_cell_times) + length(control_cell_times));
% end
% if (length(stim_cell_times) + length(control_cell_times)) > max_cones
%     max_cones = (length(stim_cell_times) + length(control_cell_times));
% end


hz=16.66666666;
timeBase = ((1:allmax)/hz)';

[remain kid] = getparent(stimRootDir);
[remain region] = getparent(remain);
[remain stim_time] = getparent(remain);
[remain stim_intensity] = getparent(remain);
[remain stimwave] = getparent(remain);
[~, id] = getparent(remain);

stimlen = str2double( strrep(stim_time(1:3),'p','.') );

%% Bootstrap using the above signals
parfor b=1:1500

    rng('shuffle'); % Reshuffle the RNG after each loop to make sure we're getting a good mix.
    
    num_signals_stim = cell(allmax, 1);
    num_signals_control = cell(allmax, 1);
    
    % Get an array of which signals have data for each time point    
    for j=1:length(profileSDataNames)
        for i=1:length(ref_stim_times{j})
            num_signals_stim{ ref_stim_times{j}(i) } = [num_signals_stim{ ref_stim_times{j}(i) }; j ];
        end
    end
    
    for j=1:length(profileCDataNames) 
        for i=1:length(ref_control_times{j})
            num_signals_control { ref_control_times{j}(i) } = [num_signals_control{ ref_control_times{j}(i) } j];
        end
    end
    
    stim_signal_inds = cell(1,length(num_signals_stim));
    control_signal_inds = cell(1,length(num_signals_control));
    
    % Randomly pick the datapoints we'll average from the signals we have
    % at each time point
    for i=1:length(num_signals_stim)
        if ~isempty(num_signals_stim{i})
            signal_picks = randi( length(num_signals_stim{i}), length(num_signals_stim{i}),1);
            stim_signal_inds{i} = num_signals_stim{i}( signal_picks );
        end
    end
    
    for i=1:length(num_signals_control)
        if ~isempty(num_signals_control{i})
            signal_picks = randi( length(num_signals_control{i}), length(num_signals_control{i}),1);
            control_signal_inds{i} = num_signals_control{i}( signal_picks );
        end
    end       

    
    pooled_variance_stim = zeros(allmax, 1);
    pooled_variance_stim_count = zeros(allmax, 1);
    pooled_variance_control = zeros(allmax, 1);
    pooled_variance_control_count = zeros(allmax, 1);

    %% Create the pooled variance for each of these
    for i=1:length(pooled_variance_stim)
        for j=1:length(stim_signal_inds{i})
            which_ind = stim_signal_inds{i}(j);
            which_time = ref_stim_times{ which_ind };
            which_signal = ref_variance_stim{ which_ind };
            which_count = ref_stim_count{ which_ind };
            
            % Create the upper and lower halves of our pooled variance
            pooled_variance_stim( i ) = pooled_variance_stim( i ) + which_signal( which_time==i );
            pooled_variance_stim_count( i ) = pooled_variance_stim_count( i ) + which_count( which_time==i );
        end
    end

    for i=1:length(pooled_variance_control)
        for j=1:length(control_signal_inds{i})
            which_ind = control_signal_inds{i}(j);
            which_time = ref_control_times{ which_ind };
            which_signal = ref_variance_control{ which_ind };
            which_count = ref_control_count{ which_ind };
            
            % Create the upper and lower halves of our pooled variance
            pooled_variance_control( i ) = pooled_variance_control( i ) + which_signal( which_time==i );
            pooled_variance_control_count( i ) = pooled_variance_control_count( i ) + which_count( which_time==i );
        end
    end

    for i=1:length(pooled_variance_stim)    
        pooled_variance_stim(i) = pooled_variance_stim(i)/pooled_variance_stim_count(i);
    end
    for i=1:length(pooled_variance_control)    
        pooled_variance_control(i) = pooled_variance_control(i)/pooled_variance_control_count(i);
    end

%     figure(10); 
%     plot( timeBase,sqrt(pooled_variance_stim),'r'); hold on;
%     plot( timeBase,sqrt(pooled_variance_control),'b');

    pooled_std_stim = pooled_variance_stim-pooled_variance_control;
    pooled_std_stim = sqrt(pooled_std_stim-min(pooled_std_stim));
    pooled_std_stim(imag(pooled_std_stim)>0) = pooled_std_stim(imag(pooled_std_stim)>0)*sqrt(-1);

%     plot( timeBase(~isnan(pooled_std_stim)), pooled_std_stim(~isnan(pooled_std_stim)),'k'); hold on;
%     legend('Stimulus cones','Control cones','Subtraction');

    [fitCharacteristics, residuals] = modelFit(timeBase, pooled_std_stim);    

    if (residuals < 10) && (fitCharacteristics.amplitude < 3)
       all_amps(b) = fitCharacteristics.amplitude;
       all_res(b) = residuals; 
    else
       residuals
       fitCharacteristics.amplitude
       all_amps(b) = -1;
       all_res(b) = -1;
    end
end
sum(all_amps>=0)
all_amps = all_amps(all_amps>=0);
all_res = all_res(all_res>=0);

all_amps = all_amps(1:1000);
all_res = all_res(1:1000);

characteristics.avg_num_control_cones = (num_control_cones)/(length(profileCDataNames));
characteristics.avg_num_stim_cones = num_stim_cones/length(profileSDataNames);
characteristics.num_control_pooled = length(profileSDataNames);
characteristics.num_stim_pooled = length(profileCDataNames);
characteristics.subject = id;
characteristics.stim_intensity = stim_intensity;
characteristics.stim_length = stimlen;
characteristics.stim_wavelength = stimwave;

characteristics.all_amps = all_amps;
characteristics.mean_amp = mean(all_amps);
characteristics.std_amp = std(all_amps);

characteristics.mean_mse = mean(all_res);
characteristics.std_mse =std(all_res);
characteristics.min_mse = min(all_res);
characteristics.max_mse = max(all_res)

