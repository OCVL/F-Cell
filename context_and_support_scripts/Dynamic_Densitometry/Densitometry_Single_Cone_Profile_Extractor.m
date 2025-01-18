
% Created by Robert F Cooper 2018-09-06
%

clear;
close all;


NUMTRIALS=4;
CRITICAL_TIME = 0:69;
START_IND=5;

CELL_OF_INTEREST = [211 404 476 783 991 1045 1398]; % High intrinsic, low densitometry.

if isempty(CELL_OF_INTEREST)
    close all force;
    
end

if ~exist('stimRootDir','var')
    stimRootDir = uigetdir(pwd, 'Select the directory containing the densitometry profiles');
end

profileSDataNames = read_folder_contents(stimRootDir,'mat');

% For structure:
%/stuff/id/date/wavelength/time/intensity/location/data/Profile_Data

[remain kid] = getparent(stimRootDir); % data
[remain stim_loc] = getparent(remain); % location 
[remain stim_intensity] = getparent(remain); % intensity 
[remain stim_time] = getparent(remain); % time
[remain stimwave] = getparent(remain); % wavelength
[remain sessiondate] = getparent(remain); % date
[~, id] = getparent(remain); % id

outPath = fullfile(remain,sessiondate);
outFname = [id '_' stimwave '_' stim_intensity '_' stim_time  '_' stim_loc '_' num2str(length(profileSDataNames)) '_single_cone_signals'];



%% Code for determining variance across all signals at given timepoint

THEwaitbar = waitbar(0,'Loading profiles...');

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

    waitbar(j/length(profileSDataNames),THEwaitbar,'Loading profiles...');
    
    ref_coords=[];
    profileSDataNames{j}
    load(fullfile(stimRootDir,profileSDataNames{j}));
    
    if ~isempty(CELL_OF_INTEREST)
        stim_cell_reflectance_nonorm{j} = cell_reflectance;
    end
    stim_cell_reflectance{j} = norm_cell_reflectance;
    stim_time_indexes{j} = cell_times;
    stim_cell_prestim_mean{j} = cell_prestim_mean;
    
%     thesecoords = union(stim_coords, ref_coords,'rows');
%     
%     % These all must be the same length! (Same coordinate set)
%     if size(ref_coords,1) ~= size(stim_coords,1)
%         error('Coordinate lists different between mat files in this directory. Unable to perform analysis.')
%     end
    
    for k=1:length(cell_times)
        max_index = max([max_index max(cell_times{k})]);
    end
    
end

% The coordinate lists must the same length,
% otherwise it's not likely they're from the same set.

allcoords = stim_coords;


%% Aggregation of all trials

percentparula = parula(101);

numstimcoords = size(stim_coords,1);

criticalfit = nan(numstimcoords,length(CRITICAL_TIME));
densitometry_fit_amplitude = nan(numstimcoords,1);
densitometry_init_val = nan(numstimcoords,1);
densitometry_b_amplitude  = nan(numstimcoords,1);
densitometry_trial_count = zeros(numstimcoords,1);
valid_densitometry = false(numstimcoords,1);

hz = 17.75;
i=1;

% Create the fit we're going to use. This is what is outlined in Ram's
% paper.
ft = fittype( 'c-a*exp(-x/b)', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares');
opts.Display = 'Off';
opts.Lower = [-0.2 0 0];
opts.Upper = [2 1 2];


START_TIME = START_IND/hz;
        
for i=1:numstimcoords
    waitbar(i/size(stim_coords,1),THEwaitbar,'Processing signals...');

    numtrials = 0;
    all_times_ref = nan(length(profileSDataNames), max_index);
    all_times = nan(length(profileSDataNames), max_index);
    for j=1:length(profileSDataNames)
        
        if ~isempty(stim_cell_reflectance{j}{i})
           
            numtrials = numtrials+1;

            all_times_ref(j, stim_time_indexes{j}{i} ) = stim_cell_reflectance{j}{i};
            all_times(j, stim_time_indexes{j}{i}) = stim_time_indexes{j}{i}/hz;
        end              
    end
    densitometry_trial_count(i) = numtrials;
    valid_densitometry(i) = densitometry_trial_count(i)>=NUMTRIALS;
    
    if valid_densitometry(i)


        vect_ref = all_times_ref(~isnan(all_times) & all_times<3);
        vect_times = all_times(~isnan(all_times)& all_times<3);
        vect_times = vect_times-START_TIME;

        opts.StartPoint = [0 1 mean(vect_ref)];
        [fitresult, gof] = fit( vect_times, vect_ref, ft, opts );

        densitometry_vect_times{i} = vect_times;
        densitometry_vect_ref{i} = vect_ref;
        criticalfit(i,:) = fitresult(CRITICAL_TIME/hz);
        densitometry_fit_amplitude(i) = ((criticalfit(i,end))-(criticalfit(i,1)));
        densitometry_b_amplitude(i) = fitresult.b;
        initvals = all_times_ref(:,1:2);
        densitometry_init_val(i) = mean(initvals(:),'omitnan');

        
        if any(i==CELL_OF_INTEREST) || (densitometry_fit_amplitude(i) <=-0.1)
            %%
            figure(1); clf; hold on;
            
%             datapresent={};
%             for j=1:length(profileSDataNames)                
%                 if any(~isnan(all_times_ref(j,:)))
%                      plot(all_times(j,:)-START_TIME, all_times_ref(j,:),'.');
%                     datapresent = [datapresent; {num2str(j)}];
%                 end
%             end
%             legend(datapresent);
%             plot(vect_times,vect_ref,'r*')
plot(densitometry_vect_times{i}, densitometry_vect_ref{i},'.');
            plot(CRITICAL_TIME/hz, criticalfit(i,:));
            xlabel('Time index'); ylabel('Standardized Response');
            title(['Cell #:' num2str(i) ', Amplitude: ' num2str(densitometry_fit_amplitude(i))]);
            axis([0 3 0 1.5]);
            hold off;
            drawnow;
%             fitresult
%             saveas(gcf, [outFname '_cell_' num2str(i) '.png']);
%             pause;
        end
        
    end

end
close(THEwaitbar);
%%
save(fullfile(outPath,[outFname '.mat']),'densitometry_fit_amplitude','valid_densitometry',...
                                        'densitometry_trial_count','criticalfit','NUMTRIALS',...
                                        'CRITICAL_TIME','START_IND','hz',...
                                        'densitometry_vect_times','densitometry_vect_ref')

%% Analyze the fitted amplitudes

histogram(densitometry_fit_amplitude,30,'BinEdges',-0.1:0.02:0.6);
xlabel('Fitted amplitude'); ylabel('Number of cones');
drawnow;
saveas(gcf,'amp_hisotogram_dens.fig')

% figure;
% histogram2(densitometry_fit_amplitude,densitometry_b_amplitude,30);
% xlabel('Fitted Amplitude'); ylabel('Fit b value');%, 'XBinEdges',0:0.02:0.6, 'YBinEdges',0:1:30);
% drawnow;
% saveas(gcf,'b_vs_amp_hisotogram_dens.fig')

counts = histcounts(densitometry_fit_amplitude,-0.1:0.02:1);

%%
figure; clf;
reghist = histogram(densitometry_fit_amplitude,'BinWidth',.015);


mainpeakind = mode(discretize(densitometry_fit_amplitude, reghist.BinEdges));
mainpeakval = reghist.Values(mainpeakind);
mainpeak = reghist.BinEdges(mainpeakind);

% Fit a gaussian mixture model to determine which is which.
sigmar = std(densitometry_fit_amplitude,'omitnan')
sigmar =repmat(sigmar,[1 1 2]);
startstruct = struct('mu',[min(reghist.BinEdges); mainpeak],'Sigma', sigmar,'ComponentProportion',[.15; .85]);

gausfits=fitgmdist(densitometry_fit_amplitude,2,'Start', startstruct)

range = min(densitometry_fit_amplitude):.001:max(densitometry_fit_amplitude);
leftmix=pdf('Normal',range',gausfits.mu(1),sqrt(gausfits.Sigma(1)));
rightmix=pdf('Normal',range',gausfits.mu(2),sqrt(gausfits.Sigma(2)));
mewomix=pdf(gausfits,range');

normleftmix = mainpeakval.*gausfits.ComponentProportion(1).*leftmix./max(leftmix);
normrightmix = mainpeakval.*gausfits.ComponentProportion(2).*rightmix./max(rightmix);
normmewomix = mainpeakval.*mewomix./max(mewomix);


hold on;
plot(range,normleftmix); 
plot(range,normrightmix);
plot(range,normmewomix);


DENSTOMETRY_THRESHOLD = max(range(normleftmix>normrightmix));


figure; clf;
lessthanvalid = (densitometry_fit_amplitude<=DENSTOMETRY_THRESHOLD) & valid_densitometry;
[V,C] = voronoin(allcoords,{'QJ'});
numlowresp=0;
for i=1:size(allcoords,1)
        
    vertices = V(C{i},:);

    if all(vertices(:,1)<max(allcoords(:,1))) && all(vertices(:,2)<max(allcoords(:,1))) ... % [xmin xmax ymin ymax] 
                            && all(vertices(:,1)>0) && all(vertices(:,2)>0) 

        if lessthanvalid(i)
            patch(V(C{i},1),V(C{i},2),ones(size(V(C{i},1))),'FaceColor', 'b');
            numlowresp = numlowresp + 1;
        elseif valid_densitometry(i)
            patch(V(C{i},1),V(C{i},2),ones(size(V(C{i},1))),'FaceColor', 'r');
        end
    end
end

% For when you want to fit all of the data, not just each trial
% vect_ref = all_times_ref(~isnan(all_times));
% vect_times = all_times(~isnan(all_times));
% vect_times = vect_times-START_TIME;
% 
% opts.StartPoint = [.1 1 mean(vect_ref)];
% [fitresult, gof] = fit( vect_times, vect_ref, ft, opts );
% 
% densitometry_vect_times{i} = vect_times;
% densitometry_vect_ref{i} = vect_ref;
% criticalfit(i,:) = fitresult(CRITICAL_TIME/hz);
% densitometry_fit_amplitude(i) = (max(criticalfit(i,end))-min(criticalfit(i,2)));
% densitometry_b_amplitude(i) = fitresult.b;
% 
% if any(i==CELL_OF_INTEREST) && (densitometry_fit_amplitude(i) <=0.2 && densitometry_fit_amplitude(i) >=.1)
%     figure(1); clf; hold on;
% 
%     datapresent={};
%     for j=1:length(profileSDataNames)                
%         if any(~isnan(all_times_ref(j,:)))
%             plot(all_times(j,:)-START_TIME, all_times_ref(j,:),'.');
%             datapresent = [datapresent; {num2str(j)}];
%         end
%     end
%     legend(datapresent);            
%     plot(CRITICAL_TIME/hz, criticalfit(i,:));
%     xlabel('Time index'); ylabel('Standardized Response');
%     title(['Cell #:' num2str(i) ', Amplitude: ' num2str(densitometry_fit_amplitude(i))]);
%     axis([0 CRITICAL_TIME(end)/hz 0 1.5]);
%     hold off;
%     drawnow;
%     fitresult
%     pause;
% end

% For looking at the fits to individual traces
%         all_times = all_times(:,START_IND:end)-START_TIME;
%         all_times_ref = all_times_ref(:,START_IND:end);
% 
%         % Fit the model from Ram's paper to the data.
% %         vect_ref = all_times_ref(~isnan(all_times));
% %         vect_times = all_times(~isnan(all_times));
% %         vect_times = vect_times-START_TIME;
%         figure(1); clf; hold on;
%         for j=1:length(profileSDataNames)
%             if any(~isnan(all_times_ref(j,:)))
%                 vect_ref = all_times_ref(j,:);
%                 vect_ref = vect_ref(~isnan(vect_ref))';
%                 vect_times = all_times(j,:);
%                 vect_times = vect_times(~isnan(vect_times))';
% 
%                 opts.StartPoint = [.1 1 mean(vect_ref)];
%                 [fitresult, gof] = fit( vect_times, vect_ref, ft, opts );
% 
%                 criticalfit(i,:) = fitresult(CRITICAL_TIME/hz);
%                 densitometry_fit_amplitude(i) = (max(criticalfit(i,end))-min(criticalfit(i,2)));
%                 densitometry_b_amplitude(i) = fitresult.b;
% 
%                 plot(vect_times, vect_ref,'.');
% 
%                 legend(num2str(j));            
%                 plot(CRITICAL_TIME/hz, criticalfit(i,:));
%                 xlabel('Time index'); ylabel('Standardized Response');
%                 title(['Cell #:' num2str(i) ', Amplitude: ' num2str(densitometry_fit_amplitude(i))]);
%                 axis([0 CRITICAL_TIME(end)/hz 0 1.5]);
% %                 hold off;
%                 drawnow;
%                 fitresult
%                 
%             end
%         end
%         pause;
