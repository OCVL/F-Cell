

clear;
close all force;


profileDataNames = read_folder_contents(pwd,'mat');

if length(profileDataNames) < 2
    error('Requires more than one dataset to do conservation analysis.');
end


load(profileDataNames{1});

control_inds = cell(length(profileDataNames),1);
stimulus_inds = cell(length(profileDataNames),1);

control_times = cell(length(profileDataNames),1);
control_reflectance = cell(length(profileDataNames),1);

stim_times = cell(length(profileDataNames),1);
stim_reflectance = cell(length(profileDataNames),1);

for j=1:length(profileDataNames)

    load(profileDataNames{j});
    
    control_inds{j} = contcellinds;       
    control_times{j} = control_cell_times;
    control_reflectance{j} = norm_control_cell_reflectance;
    
    stimulus_inds{j} = stimcellinds;
    stim_times{j} = stim_cell_times;
    stim_reflectance{j} = norm_stim_cell_reflectance;
        
end


[cons_control_inds]= intersect(control_inds{1}, control_inds{2});
[cons_stimulus_inds]= intersect(stimulus_inds{1}, stimulus_inds{2});

for i=3:1:length(profileDataNames)
   
    [cons_control_inds] = intersect(cons_control_inds, control_inds{i});
    [cons_stimulus_inds] = intersect(cons_stimulus_inds, stimulus_inds{i});
    
end

for j=1:length(profileDataNames)

    control_times{j} = control_times{j}(cons_control_inds);
    control_reflectance{j} = control_reflectance{j}(cons_control_inds);
    
    stim_times{j} = stim_times{j}(cons_stimulus_inds);
    stim_reflectance{j} = stim_reflectance{j}(cons_stimulus_inds);
    
end

hz = 16.666666;
stim_locs = 55:88;

avg_order = 5;
filt_coeff = ones(avg_order,1)/avg_order;


% profile_vid = VideoWriter('NC_11049_0ND_control_cell_profiles.avi','Uncompressed AVI');
% open(profile_vid);

stim_response_detected = zeros( length(cons_stimulus_inds), length(profileDataNames) );

for i=1: length(cons_stimulus_inds)
    
    maxresp = 0;
    maxtime = 0;
    power_spect=[];
    
    for j=1:length(profileDataNames)
%         conv(stim_reflectance{j}{i}, filt_coeff,'same')

        reflectance = conv(stim_reflectance{j}{i}, filt_coeff,'same');

        % Determine if the cell responded in this trial.
        num_prestim_pts = length( reflectance(stim_times{j}{i}<stim_locs(1) & ~isnan( reflectance )) );
        prestim_mean = mean( ( reflectance(stim_times{j}{i}<stim_locs(1) & ~isnan( reflectance )) ) );
        prestim_std =   std( ( reflectance(stim_times{j}{i}<stim_locs(1) & ~isnan( reflectance )) ) );
        
        stim_max = max( ( reflectance(stim_times{j}{i}>=stim_locs(1) & stim_times{j}{i}<=stim_locs(end) & ~isnan( reflectance )) ) );
        stim_min = min( ( reflectance(stim_times{j}{i}>=stim_locs(1) & stim_times{j}{i}<=stim_locs(end) & ~isnan( reflectance )) ) );
        
%         interval_bound = prestim_mean + 1.96*prestim_std/sqrt(num_prestim_pts);                               % Confidence interval
        pos_interval_bound = prestim_mean + tinv(0.99, num_prestim_pts-1)*prestim_std*sqrt(1+(1/num_prestim_pts));  % Prediction interval
        neg_interval_bound = prestim_mean - tinv(0.99, num_prestim_pts-1)*prestim_std*sqrt(1+(1/num_prestim_pts));
        
        if stim_max > pos_interval_bound
            stim_response_detected(i,j) = 1;
        elseif stim_min < neg_interval_bound
            stim_response_detected(i,j) = 1;
        end

        figure(2); plot( stim_times{j}{i},  reflectance ); hold on;
        
        maxresp = max(maxresp, max(reflectance) );
        maxtime = max(maxtime, length(stim_times{j}{i}) );
        
%         plot( control_times{j}{i}, control_reflectance{j}{i}); hold on;
%         maxresp = max(maxresp, max(control_reflectance{j}{i}) );
%         maxtime = max(maxtime, length(control_times{j}{i}) );
    end
    
    per_oer_thresh  = 100*( sum(stim_response_detected(i,:))/length(profileDataNames) );
    
    plot(stim_locs, maxresp*ones(34),'r*'); title([num2str(per_oer_thresh) '% over threshold.']);  hold off;
    frame = getframe(gcf);
%     writeVideo(profile_vid,frame);
    pause;
end

% integrated area under the curve (absolute)

% close(profile_vid);
