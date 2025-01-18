% function [written_file, written_path]=Automatic_Frame_Culler_Pipl(confocal_fname, pathname)
% Robert F Cooper -9-2017
%
% This script enables a user to remove poor frames from a temporal confocal
% and
% series, while simultaneously updating the acceptable_frames
% file. With the inclusion of the shutter, there is no need to track
% visible frames.



%% Filename determination and handling
[fname, pathname] = uigetfile('*.avi', 'Select the temporal videos you wish to split', 'MultiSelect','on');

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

thebar = waitbar(0,'Splitting videos...');
for k=1:length(fname)
    waitbar(k/length(fname),thebar,['Splitting video: ' fname{k}])

    vidobj = VideoReader( fullfile(pathname, fname{k}) );

    %% File loading
    vid_length = round(vidobj.Duration*vidobj.FrameRate);

    if split > vid_length
       error('Chosen split index exceeds the length of the selected video!')
    end
    
    vid = cell(1, vid_length);
    frame_nums = cell(1, vid_length);

    first_piece  = zeros(vidobj.Height,vidobj.Width, split);
    second_piece  = zeros(vidobj.Height,vidobj.Width, vid_length-(split));
    
    i=1;
    while hasFrame(vidobj)
        if i <=split
            first_piece(:,:,i) = readFrame(vidobj);
        else
            second_piece(:,:,i-split) = readFrame(vidobj);
        end

        frame_nums{i} = ['Frame ' num2str(i) ' of: ' num2str(size(vid,2))];
        i=i+1;
    end
    
    % TO FORCE EQUIVALENCE
    second_piece = second_piece(:,:,1+abs(size(second_piece,3)-size(first_piece,3)):end);
        
    
    %% Output the split videos
    % Create our confocal output filename 
    fname_out_first_piece = [fname{k}(1:end-4) '_1_' num2str(split) '.avi'];
    fname_out_second_piece = [fname{k}(1:end-4) '_' num2str(split+1) '_' num2str(vid_length) '.avi'];

    
    vidobj_1 = VideoWriter( fullfile(pathname, fname_out_first_piece), 'Grayscale AVI');
    vidobj_1.FrameRate=16.6666;

    open(vidobj_1);
    % **** Add a dummy frame so that the bug in Alf's software is ignored.
    first_piece = cat(3, zeros(vidobj.Height,vidobj.Width), first_piece);
    writeVideo(vidobj_1,uint8(first_piece));
    close(vidobj_1);
    
    vidobj_2 = VideoWriter( fullfile(pathname, fname_out_second_piece), 'Grayscale AVI');
    vidobj_2.FrameRate=16.6666;
    
    open(vidobj_2);
    % **** Add a dummy frame so that the bug in Alf's software is ignored.
    second_piece = cat(3, zeros(vidobj.Height,vidobj.Width), second_piece);
    writeVideo(vidobj_2,uint8(second_piece));
    close(vidobj_2);

end
close(thebar);
