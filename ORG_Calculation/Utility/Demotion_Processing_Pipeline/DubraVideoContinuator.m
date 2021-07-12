% DubraVideoContinuator
% 
% Thsi script takes a video output from Alf Dubra's Savior and makes it
% temporally continuous, by filling in the gaps it finds with blank frames
% and cutting off the video at the specified frame number.


%% Filename determination and handling
[fname, pathname] = uigetfile('*.avi', 'Select the temporal videos you wish to fill', 'MultiSelect','on');

clip=NaN;
while isnan(clip)
    answer = inputdlg('Which frame are the videos supposed to end at (inclusive)?');
    clip = str2double(answer);
    if clip < 1
        clip = NaN;
    end
end

if ~iscell(fname)
    fname={fname};
end

thebar = waitbar(0,'Filling and clipping videos...');
for k=1:length(fname)
    waitbar(k/length(fname),thebar,['Filling and clipping video: ' fname{k}])
        

    if ~exist(fullfile(pathname, [fname{k}(1:end-4) '.mat']),'file')
        load(fullfile(pathname, strrep([fname{k}(1:end-4) '.mat'], 'split_det', 'confocal')));
    else
        load(fullfile(pathname, [fname{k}(1:end-4) '.mat']));
    end
    vidobj = VideoReader( fullfile(pathname, fname{k}) );

    frame_numbers = frame_numbers - min(frame_numbers) + 1;
    
    %% Video loading
    vid_length = round(vidobj.Duration*vidobj.FrameRate);

    if clip > vid_length
       vid_length = clip;
    end
    
    vid = cell(1, vid_length);
    frame_nums = cell(1, vid_length);

    full_length_vid  = zeros(vidobj.Height, vidobj.Width, vid_length);
    
    i=1;
    while hasFrame(vidobj) 
        if any(i==frame_numbers)
            full_length_vid(:,:,i) = readFrame(vidobj);
        end

        frame_nums{i} = ['Frame ' num2str(i) ' of: ' num2str(size(vid,2))];
        i=i+1;
    end
    % Clip it to the length it is supposed to be.
    full_length_vid = full_length_vid(:,:,1:clip);
    
    %% Output the videos
    fname_out = [fname{k}(1:end-4) '_temp_cont.avi'];
    
    vidobj_1 = VideoWriter( fullfile(pathname, fname_out), 'Grayscale AVI');
    vidobj_1.FrameRate=17.85;

    open(vidobj_1);    
    writeVideo(vidobj_1,uint8(full_length_vid));
    close(vidobj_1);
    
end