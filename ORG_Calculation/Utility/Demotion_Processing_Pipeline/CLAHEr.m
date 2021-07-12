% CLAHEr
%
% I hardly know her!
%

%% Filename determination and handling
[fname, pathname] = uigetfile('*.avi', 'Select the temporal videos you wish to fill', 'MultiSelect','on');

if ~iscell(fname)
    fname={fname};
end


for k=1:length(fname)
    disp(['CLAHEizing video: ' fname{k}])
    
    vidobj = VideoReader( fullfile(pathname, fname{k}) );

    %% Video loading
    vid_length = round(vidobj.Duration*vidobj.FrameRate);

    vid = cell(1, vid_length);
    frame_nums = cell(1, vid_length);

    full_length_vid  = zeros(vidobj.Height, vidobj.Width, vid_length);
    
    windowsize = 5;
    numtiles = ceil([vidobj.Height/windowsize vidobj.Width/windowsize]);
    
    i=1;
    while hasFrame(vidobj) 

%         full_length_vid(:,:,i) = adapthisteq(readFrame(vidobj), 'ClipLimit', 0.9 , 'NumTiles', numtiles );
        logim = log10(double(readFrame(vidobj)+1));
        
        full_length_vid(:,:,i) = uint8(255*(logim./max(logim(:))));

        i=i+1;
    end

    
    %% Output the videos
    fname_out = [fname{k}(1:end-4) '_CLAHE.avi'];
    
    vidobj_1 = VideoWriter( fullfile(pathname, fname_out), 'Grayscale AVI');
    vidobj_1.FrameRate=17.85;

    open(vidobj_1);    
    writeVideo(vidobj_1,uint8(full_length_vid));
    close(vidobj_1);
    
end