function [ pooled_std_control, pooled_std_stim, signallen, subid ] = pooled_stddev_from_dir( rootDir )

% Robert F Cooper
% 12-31-2015
% This script calculates pooled variance across a set of given signals.

if ~exist('rootDir','var')
    rootDir = uigetdir(pwd);
end

profileDataNames = read_folder_contents(rootDir,'mat');

thatstimmax=0;
thatcontrolmax=0;
%% Code for determining variance across all signals at given timepoint

allmax=0;

mean_control_reflectance = zeros(500,1);

for j=1:length(profileDataNames)
    profileDataNames{j}
    load(fullfile(rootDir,profileDataNames{j}));
    
    % Remove the empty cells
    norm_stim_cell_reflectance = norm_stim_cell_reflectance( ~cellfun(@isempty,norm_stim_cell_reflectance) );
    stim_cell_times            = stim_cell_times(  ~cellfun(@isempty,stim_cell_times) );
    norm_control_cell_reflectance = norm_control_cell_reflectance( ~cellfun(@isempty,norm_control_cell_reflectance)  );
    control_cell_times            = control_cell_times( ~cellfun(@isempty,control_cell_times) );
    
    
    thatstimmax = max( cellfun(@max,stim_cell_times) );    
    
    thatcontrolmax = max( cellfun(@max,control_cell_times) );

    if thatstimmax ~= thatcontrolmax
        error('The control and stimulus number of frames do not match! Cannot perform analysis...');
    end
    if thatstimmax > allmax
        allmax=thatstimmax;
    end

    
    % Pooled variance of all cells before first stimulus
    [ ref_variance_stim{j}, ref_stim_times{j}, ref_stim_count{j} ]    = reflectance_pooled_variance( stim_cell_times, norm_stim_cell_reflectance, allmax );
    [ ref_variance_control{j},ref_control_times{j}, ref_control_count{j} ] = reflectance_pooled_variance( control_cell_times, norm_control_cell_reflectance, allmax );    


    i=1;
    while i<= length( ref_control_times{j} )

        % Remove times from both stim and control that are NaN
        if isnan(ref_stim_times{j}(i)) || isnan(ref_control_times{j}(i))

            ref_stim_count{j}(i) = [];
            ref_control_count{j}(i) = [];
            
            ref_stim_times{j}(i) = [];
            ref_control_times{j}(i) = [];

            ref_variance_stim{j}(i) = [];
            ref_variance_control{j}(i) = [];        
        else
%             ref_times = [ref_times; ref_stim_times(i)];
            i = i+1;
        end

    end
    
%     figure(1); plot(ref_stim_times{j}, sqrt(ref_variance_stim{j})-sqrt(ref_variance_stim{j}(1)) ); hold on; drawnow;
    
    for i=1 : length(norm_control_cell_reflectance)
        for k=1 : length( norm_control_cell_reflectance{i} )

            if ~isnan( norm_control_cell_reflectance{i}(k) )
                if mean_control_reflectance(k) == 0
                    mean_control_reflectance(k) = norm_control_cell_reflectance{i}(k);
                else
                    mean_control_reflectance(k) = (mean_control_reflectance(k) + norm_control_cell_reflectance{i}( (k) ) )/2;
                end
            end
        end
    end
    

end
% hold off;

pooled_variance_stim = zeros(allmax,1);
pooled_variance_stim_count = zeros(allmax,1);

pooled_variance_control = zeros(allmax,1);
pooled_variance_control_count = zeros(allmax,1);

% Create the pooled variance for each of these

for j=1:length(profileDataNames)
    
    for i=1:length(ref_stim_times{j})
    
        % Create the upper and lower halves of our pooled variance
        pooled_variance_stim( ref_stim_times{j}(i) ) = pooled_variance_stim( ref_stim_times{j}(i) ) + ref_variance_stim{j}(i);
        pooled_variance_stim_count( ref_stim_times{j}(i) ) = pooled_variance_stim_count( ref_stim_times{j}(i) ) + (ref_stim_count{j}(i)-1);

    end
    
    for i=1:length(ref_control_times{j})
    
        % Create the upper and lower halves of our pooled variance
        pooled_variance_control( ref_control_times{j}(i) ) = pooled_variance_control( ref_control_times{j}(i) ) + ref_variance_control{j}(i);
        pooled_variance_control_count( ref_control_times{j}(i) ) = pooled_variance_control_count( ref_control_times{j}(i) ) + (ref_control_count{j}(i)-1);

    end
end

for i=1:length(pooled_variance_stim)    
    pooled_variance_stim(i) = pooled_variance_stim(i)/pooled_variance_stim_count(i);
end
for i=1:length(pooled_variance_control)    
    pooled_variance_control(i) = pooled_variance_control(i)/pooled_variance_control_count(i);
end


pooled_std_stim    = (pooled_variance_stim);
pooled_std_control = (pooled_variance_control);
signallen = allmax;
subid = profileDataNames{1}(1:8);

end
