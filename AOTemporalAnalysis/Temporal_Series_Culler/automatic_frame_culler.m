% Robert F Cooper 16-19-2017
%
% This script enables a user to remove poor frames from a temporal confocal
% and
% series, while simultaneously updating the acceptable_frames
% file. With the inclusion of the shutter, there is no need to track
% visible frames.

clear
close all force
clc

%% Filename determination and handling
[confocal_fname, pathname] = uigetfile('*.avi', 'Select the confocal temporal video', 'MultiSelect','on');

if ~iscell(confocal_fname)
    confocal_fname={confocal_fname};
end

for f=1:length(confocal_fname)
    
confind = strfind(confocal_fname{f},'confocal');

if isempty(confind)
   error('Could not find confocal in the filename. Needed for proper function of this script!'); 
end

split_fname = strrep(confocal_fname{f}, 'confocal', 'split_det');

if exist(fullfile(pathname,split_fname),'file')
   loadsplit = 1;
else
   loadsplit = 0;
end

% Find where the filename should be cut off in the confocal videos, and
% determine our acceptable frame filename.
i=1;
[comb_str remain] = strtok(confocal_fname{f}(confind:end), '_');
acceptable_frame_fname = [];
while ~isempty(remain)
    [tok remain] = strtok( remain, '_');

    if i==4
        confocal_fname_out = comb_str;
    elseif i==9
        acceptable_frame_fname = comb_str;
        break;
    end
    
    comb_str = [comb_str '_' tok];
    
    i=i+1;
end
% Create our expected acceptable frame filenames
acceptable_frame_fname = [confocal_fname{f}(1:confind-1) acceptable_frame_fname '_acceptable_frames.csv'];

if ~exist(fullfile(pathname, acceptable_frame_fname),'file')
    reply = input('Unable to find acceptable frames csv! Search for it? Y/N [Y]:','s');
    if isempty(reply)
       reply = 'Y';
    end
    
    if strcmpi(reply,'Y')
        [acceptable_frame_fname, af_pathname] = uigetfile(fullfile(pathname, '*.csv'), 'Select the acceptable frames csv.');
    else
        error('Unable to find acceptable frames csv!');
    end
end

acceptable_frame_fname_out = [confocal_fname{f}(1:confind-1) confocal_fname_out '_crop_affine_acceptable_frames.csv'];


if loadsplit
    split_fname_out = [confocal_fname{f}(1:confind-1) confocal_fname_out '_crop.avi'];
    split_fname_out = strrep(split_fname_out, 'confocal', 'split_det');
end

% Create our confocal output filename - affine will need to be done outside
% MATLAB.
confocal_fname_out = [confocal_fname{f}(1:confind-1) confocal_fname_out '_crop.avi'];

confocal_vidobj = VideoReader( fullfile(pathname, confocal_fname{f}) );

if loadsplit
    split_vidobj = VideoReader( fullfile(pathname, split_fname) );
end

%% File loading
vid_length = round(confocal_vidobj.Duration*confocal_vidobj.FrameRate);

confocal_vid = cell(1, vid_length);
confocal_f_mean = zeros(vid_length,1);

frame_nums = cell(1, vid_length);

if loadsplit
    split_vid = cell(1, vid_length);
    split_f_mean = zeros(vid_length,1);
end

i=1;
while hasFrame(confocal_vidobj)
    
    confocal_vid{i} = readFrame(confocal_vidobj);
    confocal_f_mean(i) = mean(double(confocal_vid{i}(confocal_vid{i}~=0)));
    if loadsplit
        split_vid{i} = readFrame(split_vidobj);
        split_f_mean(i) = mean(double(split_vid{i}(split_vid{i}~=0)));
    end
    frame_nums{i} = ['Frame ' num2str(i) ' of: ' num2str(size(confocal_vid,2))];
    i=i+1;
end


acc_frame_list = dlmread( fullfile(pathname, acceptable_frame_fname) );

acc_frame_list = sort(acc_frame_list);

if length(acc_frame_list) < length(confocal_vid)
   error(['Acceptable frames and confocal video list lengths do not match! (' num2str(length(acc_frame_list)) ' vs ' num2str(length(confocal_vid)) ')']);
elseif length(acc_frame_list) > length(confocal_vid)
    acc_frame_list = acc_frame_list(1:length(confocal_vid));
end

%% Filter by image mean
confocal_mean = mean(confocal_f_mean);
confocal_dev = std(confocal_f_mean);
if loadsplit
    split_mean = mean(split_f_mean);
    split_dev = std(split_f_mean);
end

contenders = false(1,length(frame_nums));
for f=1:length(frame_nums)        
    contenders(f) =  (confocal_f_mean(f) > confocal_mean-2*confocal_dev);
    
    if loadsplit
       contenders(f) = contenders(f) & (split_f_mean(f) > split_mean-2*split_dev); 
    end
end

% Remove frames from contention.
confocal_vid = confocal_vid(contenders);
acc_frame_list = acc_frame_list(contenders);
if loadsplit
    split_vid = split_vid(contenders);
end


%% Determine which frames have divisions
frm_rp = cell(length(confocal_vid),1);
cc_areas = cell(length(confocal_vid),1);
contenders = false(length(confocal_vid),1);
div_frms = false(length(confocal_vid),1);

for f=1:length(confocal_vid)
    frm_nonzeros = (confocal_vid{f}>0);
    
    frm_nonzeros = imclose(frm_nonzeros, ones(5)); % There will always be noise in a simple threshold- get rid of it.
    
    frm_cc = bwconncomp(frm_nonzeros);    
    frm_rp{f} = regionprops(frm_cc,'Area','BoundingBox');
    
    cc_areas{f} = [frm_rp{f}.Area];
    
    % Assuming components that aren't more than an area of a few lines are
    % noise.
    big_comps = cc_areas{f} > size(confocal_vid{f},2)*6;
    
    small_comps = frm_rp{f}(~big_comps);
    
    % Mask out any small noisy areas
    for c=1:length(small_comps)
        maskbox = small_comps(c).BoundingBox;
        maskbox = round(maskbox);
        confocal_vid{f}(maskbox(2):(maskbox(2)+maskbox(4)), maskbox(1):(maskbox(1)+maskbox(3)) ) = 0;
    end
    
    % Remove the small areas from consideration.
    frm_rp{f} = frm_rp{f}(big_comps);
    cc_areas{f} = cc_areas{f}(big_comps);
end

mean_cc_area = mean([cc_areas{:}]);
std_cc_area = std([cc_areas{:}]);

for f=1:length(cc_areas)

%     imagesc( imclose(confocal_vid{f}>0, ones(5)) ); colormap gray;
    % Remove components that aren't more than mu-2std dev pixels in area    
    big_enough_comps = cc_areas{f} > mean_cc_area-2*std_cc_area;
    
    % Put these in a list to track if we have to we can use them, but
    % dropping the smaller of the two components
    if sum(big_enough_comps) == 1 && length(big_enough_comps) == 1
        contenders(f) = 1;
        
    % If it has breaks in it (and has at least one piece that's big enough)
    % flag it to determine if it is worth keeping.
    elseif sum(big_enough_comps) > 0 && sum(big_enough_comps) <= length(big_enough_comps)
        div_frms(f) = 1;        
    end
end

% Pull out the divided frames and see how workable they are.
div_confocal_vid = confocal_vid(div_frms);
div_acc_frame_list = acc_frame_list(div_frms);
if loadsplit
    div_split_vid = split_vid(div_frms);
end

% Remove the divided frames from contention (for now).
confocal_vid = confocal_vid(contenders);
acc_frame_list = acc_frame_list(contenders);
if loadsplit
    split_vid = split_vid(contenders);
end

%% Make the ideal area mask from the unregistered videos.

contenders = false(length(confocal_vid),1);

% Make a sum map of all of the undivided frames to determine the ideal
% cropping area.
crop_mask = zeros(size(confocal_vid{1}));
sum_map = zeros(size(confocal_vid{1}));

for f=1:length(confocal_vid)
    frm_nonzeros = imclose((confocal_vid{f}>0), ones(5)); 
    sum_map = sum_map+frm_nonzeros;
end
% imagesc(sum_map); axis image;

max_frm_mask = sum_map==max(sum_map(:));
average_frm_mask = sum_map > ceil(mean(sum_map(:)));
% Find the largest incribed rectangle in this mask.
[C, h, w, largest_rect] =FindLargestRectangles(average_frm_mask,[1 1 0], [300 150]);

% Find the coordinates for each corner of the rectangle, and
% return them
cropregion = regionprops(largest_rect,'BoundingBox');
cropregion = floor(cropregion.BoundingBox);

cropregion = [cropregion(1:2), cropregion(1)+cropregion(3), cropregion(2)+cropregion(4)];

sum_map_crop = sum_map(cropregion(2):cropregion(4),cropregion(1):cropregion(3));

figure(1); imagesc( sum_map_crop ); axis image; title('Ideal cropped sum map');


%% Register these frames together, removing residual rotation.

im_only_vid = cell(length(confocal_vid),1);
im_only_vid_ref = cell(length(confocal_vid),1);

for f=1:length(confocal_vid)
    
    frm_nonzeros = imclose((confocal_vid{f}>0), ones(11));
    
    masked_frm = frm_nonzeros.*largest_rect;
    cropped_masked_frm = masked_frm( cropregion(2):cropregion(4),cropregion(1):cropregion(3) );

%     if any(~cropped_masked_frm)        
        im_only_vid{f} = confocal_vid{f}( cropregion(2):cropregion(4),cropregion(1):cropregion(3) );
%     end
end

[optimizer, metric]  = imregconfig('monomodal');
% optimizer.GradientMagnitudeTolerance = 1e-4;
% optimizer.MinimumStepLength = 1e-5;
% optimizer.MaximumStepLength = 0.06;
% optimizer.MaximumIterations = 100;

tic;
disp('Forward')
cumu_tform=affine2d();
% Register the image stack forward.
parfor f=2:length(im_only_vid)

    % Register using the cropped frame,
    forward_reg_tform{f}=imregtform(im_only_vid{f}, im_only_vid{f-1},'affine',...
                            optimizer, metric,'PyramidLevels',1, 'InitialTransformation', affine2d());%,'DisplayOptimization',true);

end
toc;


reg_confocal_vid = cell(length(confocal_vid),1);
reg_confocal_vid{1} = im_only_vid{1};
cumu_tform=affine2d();
for f=2:length(im_only_vid)    
    
    cumu_tform.T = forward_reg_tform{f}.T*cumu_tform.T;
%     cumu_tform.T = forward_reg_tform{f}.T;
    
    reg_confocal_vid{f}= imwarp(im_only_vid{f}, cumu_tform,'OutputView', imref2d(size(reg_confocal_vid{1})) ); 
    figure(2); imagesc(reg_confocal_vid{f}); axis image; colormap gray;
    drawnow;
end


%% Remake the sum map with our rotated data.

% Make a sum map of all of the undivided frames to determine the ideal
% cropping area.
crop_mask = zeros(size(reg_confocal_vid{1}));
sum_map = zeros(size(reg_confocal_vid{1}));

for f=1:length(reg_confocal_vid)
    frm_nonzeros = imclose((reg_confocal_vid{f}>0), ones(5)); 
    sum_map = sum_map+frm_nonzeros;
end
imagesc(sum_map); axis image;

max_frm_mask = sum_map==max(sum_map(:));
average_frm_mask = sum_map>mean2(sum_map);
% Find the largest incribed rectangle in this mask.
[C, h, w, largest_rect] =FindLargestRectangles(max_frm_mask,[1 1 0], [300 150]);

% Find the coordinates for each corner of the rectangle, and
% return them
cropregion = regionprops(largest_rect,'ConvexHull');
cropregion = cropregion.ConvexHull;
% TL TR BR BL
cropregion = [floor(min(cropregion)); [ceil(max(cropregion(:,1))) floor( min(cropregion(:,2))) ]; ...
              ceil(max(cropregion));  [floor(min(cropregion(:,1))) ceil( max(cropregion(:,2)))] ];
cropregion(cropregion==0) = 1;
cropregion( cropregion(:,1)>size(reg_confocal_vid{1},2),1 ) = size(reg_confocal_vid{1},2);
cropregion( cropregion(:,2)>size(reg_confocal_vid{1},1),2 ) = size(reg_confocal_vid{1},1);

figure(2); imagesc(sum_map); axis image; title('Ideal cropped sum map- registered data');




%% Determine the divided files' degree of overlap with the ideal cropping region, add them to the stack if possible.
% crop_mask( cropregion(2,2):cropregion(3,2), cropregion(1,1):cropregion(2,1) ) = 1;
% 
% div_confocal_vid = confocal_vid(div_frms);
% div_acc_frame_list = acc_frame_list(div_frms);
% if loadsplit
%     div_split_vid = split_vid(div_frms);
% end
% 
% for f=1:length(div_confocal_vid)
%     frm_nonzeros = (confocal_vid{f}>0);
%     
%     frm_nonzeros = imclose(frm_nonzeros, ones(5)); % There will always be noise in a simple threshold- get rid of it.
%     
%     frm_cc = bwconncomp(frm_nonzeros);    
%     frm_rp{f} = regionprops(frm_cc,'Area','BoundingBox');
%     
% end


%% Output the cropped frames
confocal_vid_out = uint8( zeros( size(reg_confocal_vid{1},1), size(reg_confocal_vid{1},2), length(confocal_vid) ));

if loadsplit
    split_vid_out = uint8( zeros( cropregion(3,2)-cropregion(2,2)+1, cropregion(2,1)-cropregion(1,1)+1, length(confocal_vid) ));
end

for i=1:length(confocal_vid)

    confocal_vid_out(:,:,i) = uint8(reg_confocal_vid{i});

    if loadsplit
        split_vid_out(:,:,i) = uint8(split_vid{i}( cropregion(2,2):cropregion(3,2), cropregion(1,1):cropregion(2,1)));
    end

end

outfolder = 'region_cropped';
mkdir(pathname, outfolder);

dlmwrite( fullfile(pathname, outfolder,acceptable_frame_fname_out),acc_frame_list);
confocal_vidobj = VideoWriter( fullfile(pathname, outfolder, confocal_fname_out), 'Grayscale AVI' );

if loadsplit
    split_vidobj = VideoWriter( fullfile(pathname, outfolder, split_fname_out), 'Grayscale AVI' );
end


open(confocal_vidobj);
writeVideo(confocal_vidobj,confocal_vid_out);
close(confocal_vidobj);

if loadsplit
    open(split_vidobj);
    writeVideo(split_vidobj,split_vid_out);
    close(split_vidobj);
end

close all;
end
