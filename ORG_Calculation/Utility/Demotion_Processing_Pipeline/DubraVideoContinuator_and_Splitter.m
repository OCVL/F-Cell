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

split=NaN;
while isnan(split)
    answer = inputdlg('Which frame do you want to split the videos at (inclusive)?');
    split = str2double(answer);
    if split < 1
        split = NaN;
    end
end

if ~iscell(fname)
    fname={fname};
end

thebar = waitbar(0,'Filling, clipping, and splitting videos...');
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
    %% File loading

    vid = cell(1, vid_length);
    frame_nums = cell(1, vid_length);

    first_piece = full_length_vid(:,:,1:split);
    second_piece = full_length_vid(:,:,split+1:end);
        
    % Output the split videos
    % Create our confocal output filename 
    fname_out_first_piece = [fname{k}(1:end-4) '_1_' num2str(split) '.avi'];
    fname_out_second_piece = [fname{k}(1:end-4) '_' num2str(split+1) '_' num2str(vid_length) '.avi'];
    
    vidobj_1 = VideoWriter( fullfile(pathname, fname_out_first_piece), 'Grayscale AVI');
    vidobj_1.FrameRate=17.85;

    open(vidobj_1);
    % **** Add a dummy frame so that the bug in Alf's software is ignored.
    first_piece = cat(3, zeros(vidobj.Height,vidobj.Width), first_piece);
    writeVideo(vidobj_1,uint8(first_piece));
    close(vidobj_1);
%     vidobj_2 = VideoWriter( fullfile(pathname, fname_out_second_piece), 'Grayscale AVI');
%     vidobj_2.FrameRate=17.85;
%     
%     open(vidobj_2);
%     % **** Add a dummy frame so that the bug in Alf's software is ignored.
%     second_piece = cat(3, zeros(vidobj.Height,vidobj.Width), second_piece);
%     writeVideo(vidobj_2,uint8(second_piece));
%     close(vidobj_2);
    
    
end
close(thebar);