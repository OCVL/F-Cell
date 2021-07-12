function [written_file, written_path]=Automatic_Frame_Culler_Pipl(video_fname, pathname)
% Robert F Cooper 16-19-2017
%
% This script enables a user to remove poor frames from a temporal confocal
% and
% series, while simultaneously updating the acceptable_frames
% file. With the inclusion of the shutter, there is no need to track
% visible frames.


if ~exist('video_fname','var')
    %% Filename determination and handling
    [video_fname, pathname] = uigetfile('*.avi', 'Select the temporal video', 'MultiSelect','on');

end

if ~iscell(video_fname)
    video_fname={video_fname};
end

for k=1:length(video_fname)

    confind = strfind(video_fname{k},'confocal');

    avg_fname = strrep(video_fname{k}, 'confocal', 'avg');

    if exist(fullfile(pathname,avg_fname),'file')
       loadavg = 1;       
    else
       loadavg = 0;
    end
    
    split_fname = strrep(video_fname{k}, 'confocal', 'split_det');

    if exist(fullfile(pathname,split_fname),'file')
       loadsplit = 1;       
    else
       loadsplit = 0;
    end
    
    visible_fname = strrep(video_fname{k}, 'confocal', 'visible');

    if exist(fullfile(pathname,visible_fname),'file')
       loadvisible = 1;
    else
       loadvisible = 0;
    end
    
    
    
    if ~isempty(confind)
%        error('Could not find confocal in the filename. Needed for proper function of this script!'); 
%         end
        % Find where the filename should be cut off in the confocal videos, and
        % determine our acceptable frame filename.
        i=1;
        [comb_str remain] = strtok(video_fname{k}(confind:end), '_');
        acceptable_frame_fname = [];
        while ~isempty(remain)
            [tok remain] = strtok( remain, '_');

            if i==5
                confocal_fname_out = comb_str;
            elseif i>=8
                acceptable_frame_fname = comb_str;
                if 2==exist(fullfile(pathname, [video_fname{k}(1:confind-1) acceptable_frame_fname '_repaired_acceptable_frames.csv']),'file')
                    break;
                end
            end

            comb_str = [comb_str '_' tok];

            i=i+1;
        end

        % Create our expected acceptable frame filenames
        acceptable_frame_fname = [video_fname{k}(1:confind-1) acceptable_frame_fname '_repaired_acceptable_frames.csv'];
        acceptable_frame_fname_out = [video_fname{k}(1:confind-1) confocal_fname_out '_crop_affine'];
    end
    
    if ~exist(fullfile(pathname, acceptable_frame_fname),'file') && loadsplit
            
        % Find where the filename should be cut off in the split videos, and
        % determine our acceptable frame filename.
        splitind = strfind(split_fname,'split_det');
        i=1;
        [comb_str remain] = strtok(split_fname(splitind:end), '_');
        acceptable_frame_fname = [];
        while ~isempty(remain)
            [tok remain] = strtok( remain, '_');

            if i==6
                split_fname_out = comb_str;
            elseif i>=8
                acceptable_frame_fname = comb_str;
                if 2==exist(fullfile(pathname, [split_fname(1:splitind-1) acceptable_frame_fname '_repaired_acceptable_frames.csv']),'file')
                    break;
                end
            end

            comb_str = [comb_str '_' tok];

            i=i+1;
        end

        % Create our expected acceptable frame filenames
        acceptable_frame_fname = [split_fname(1:splitind-1) acceptable_frame_fname '_repaired_acceptable_frames.csv'];
        acceptable_frame_fname_out = [split_fname(1:splitind-1) split_fname_out '_crop_affine'];
        split_fname_out = [split_fname(1:splitind-1) split_fname_out '_crop_affine'];
    end

    if ~exist(fullfile(pathname, acceptable_frame_fname),'file')
        error(['Unable to find acceptable frames csv: ' fullfile(pathname, acceptable_frame_fname)]);
    end

    if loadsplit && ~exist('split_fname_out','var')
        split_fname_out = [video_fname{k}(1:confind-1) confocal_fname_out '_crop_affine'];
        split_fname_out = strrep(split_fname_out, 'confocal', 'split_det');
    end

    if loadavg
        avg_fname_out = [video_fname{k}(1:confind-1) confocal_fname_out '_crop_affine'];
        avg_fname_out = strrep(avg_fname_out, 'confocal', 'avg');
    end
    
    if loadvisible
        vis_fname_out = [video_fname{k}(1:confind-1) confocal_fname_out '_crop_affine'];
        vis_fname_out = strrep(vis_fname_out, 'confocal', 'visible');
    end

    % Create our confocal output filename
    confocal_fname_out = [video_fname{k}(1:confind-1) confocal_fname_out '_crop_affine'];

    confocal_vidobj = VideoReader( fullfile(pathname, video_fname{k}) );

    if loadsplit
        split_vidobj = VideoReader( fullfile(pathname, split_fname) );
    end
    if loadavg
        avg_vidobj = VideoReader( fullfile(pathname, avg_fname) );
    end
    if loadvisible
        vis_vidobj = VideoReader( fullfile(pathname, visible_fname) );
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
    if loadavg
        avg_vid = cell(1, vid_length);
        avg_f_mean = zeros(vid_length,1);
    end
    if loadvisible
        vis_vid = cell(1, vid_length);
        vis_f_mean = zeros(vid_length,1);
    end

    i=1;
    while hasFrame(confocal_vidobj)

        confocal_vid{i} = readFrame(confocal_vidobj);
        confocal_f_mean(i) = mean(double(confocal_vid{i}(confocal_vid{i}~=0)));
        if loadsplit
            split_vid{i} = readFrame(split_vidobj);
            split_f_mean(i) = mean(double(split_vid{i}(split_vid{i}~=0)));
        end
        if loadavg
            avg_vid{i} = readFrame(avg_vidobj);
            avg_f_mean(i) = mean(double(avg_vid{i}(avg_vid{i}~=0)));
        end
        if loadvisible
            vis_vid{i} = readFrame(vis_vidobj);
            vis_f_mean(i) = mean(double(vis_vid{i}(vis_vid{i}~=0)));
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
    confocal_mean = mean(confocal_f_mean,'omitnan');
    confocal_dev = std(confocal_f_mean,'omitnan');
    if loadsplit
        split_mean = mean(split_f_mean,'omitnan');
        split_dev = std(split_f_mean,'omitnan');
    end
    if loadavg
        avg_mean = mean(avg_f_mean,'omitnan');
        avg_dev = std(avg_f_mean,'omitnan');
    end
    if loadvisible
        vis_mean = mean(vis_f_mean,'omitnan');
        vis_dev = std(vis_f_mean,'omitnan');
    end

    confocal_mean(isnan(confocal_mean)) = 0;
    
    contenders = false(1,length(frame_nums));
    for n=1:length(frame_nums)        
        contenders(n) =  (confocal_f_mean(n) > confocal_mean-2*confocal_dev);
    end

    % Remove frames from contention.
    confocal_vid = confocal_vid(contenders);
    acc_frame_list = acc_frame_list(contenders);
    if loadsplit
        split_vid = split_vid(contenders);
    end
    if loadavg
        avg_vid = avg_vid(contenders);
    end
    if loadvisible
        vis_vid = vis_vid(contenders);
    end


    %% Determine which frames have divisions
    frm_rp = cell(length(confocal_vid),1);
    cc_areas = cell(length(confocal_vid),1);
    contenders = false(length(confocal_vid),1);
    div_frms = false(length(confocal_vid),1);

    for n=1:length(confocal_vid)
        frm_nonzeros = (confocal_vid{n}>0);

        frm_nonzeros = imclose(frm_nonzeros, ones(5)); % There will always be noise in a simple threshold- get rid of it.

        frm_cc = bwconncomp(frm_nonzeros);    
        frm_rp{n} = regionprops(frm_cc,'Area','BoundingBox');

        cc_areas{n} = [frm_rp{n}.Area];

        % Assuming components that aren't more than an area of a few lines are
        % noise.
        big_comps = cc_areas{n} > size(confocal_vid{n},2)*6;

        small_comps = frm_rp{n}(~big_comps);

%         if n==110
%             size(confocal_vid{n})
%         end
        
        % Mask out any small noisy areas
        for c=1:length(small_comps)
            maskbox = small_comps(c).BoundingBox;
            maskbox = ceil(maskbox);
            % Bound our mask region to where the sum_map actually exists
            maskbox(maskbox<1) = 1;
            if (maskbox(2)+maskbox(4))>=size(confocal_vid{n},1)
                maskbox(4) = size(confocal_vid{n},1)-maskbox(2);
            end
            if (maskbox(1)+maskbox(3))>=size(confocal_vid{n},2)
                maskbox(3) = size(confocal_vid{n},2)-maskbox(1);
            end
                        
            confocal_vid{n}(maskbox(2):(maskbox(2)+maskbox(4)), maskbox(1):(maskbox(1)+maskbox(3)) ) = 0;
        end
        
        % Remove the small areas from consideration.
        frm_rp{n} = frm_rp{n}(big_comps);
        cc_areas{n} = cc_areas{n}(big_comps);
    end

    mean_cc_area = mean([cc_areas{:}]);
    std_cc_area = std([cc_areas{:}]);

    for n=1:length(cc_areas)

%         imagesc( imclose(confocal_vid{n}>0, ones(5)) ); colormap gray;
        % Remove components that aren't more than mu-2std dev pixels in area    
        big_enough_comps = cc_areas{n} > mean_cc_area-std_cc_area;

        % Put these in a list to track if we have to we can use them, but
        % dropping the smaller of the two components
        if sum(big_enough_comps) == 1 && length(big_enough_comps) == 1
            contenders(n) = 1;

        % If it has breaks in it (and has at least one piece that's big enough)
        % flag it to determine if it is worth keeping.
        elseif sum(big_enough_comps) > 0 && sum(big_enough_comps) <= length(big_enough_comps)
            div_frms(n) = 1;        
        end
    end

    % Pull out the divided frames and see how workable they are.
    div_confocal_vid = confocal_vid(div_frms);
    div_acc_frame_list = acc_frame_list(div_frms);
%     if loadsplit
%         div_split_vid = split_vid(div_frms);
%     end
%     if loadvisible
%         div_vis_vid = vis_vid(div_frms);
%     end

    % Remove the divided frames from contention (for now).
    confocal_vid = confocal_vid(contenders);
    acc_frame_list = acc_frame_list(contenders);
    if loadsplit
        split_vid = split_vid(contenders);
    end
    if loadavg
        avg_vid = avg_vid(contenders);
    end
    if loadvisible
        vis_vid = vis_vid(contenders);
    end

    %% Make the ideal area mask from the unregistered videos.

    contenders = false(length(confocal_vid),1);

    % Make a sum map of all of the undivided frames to determine the ideal
    % cropping area.
    crop_mask = zeros(size(confocal_vid{1}));
    sum_map = zeros(size(confocal_vid{1}));

    for n=1:length(confocal_vid)
        frm_nonzeros = imclose((confocal_vid{n}>0), ones(5));

        sum_map = sum_map+frm_nonzeros;
    end
    % imagesc(sum_map); axis image;

    max_frm_mask = sum_map >= ( max(sum_map(:))*0.8 );
    [C, h, w, max_largest_rect] =FindLargestRectangles(max_frm_mask,[1 1 0], [300 150]);
    cropregion = regionprops(max_largest_rect,'BoundingBox');
    cropregion = floor(cropregion.BoundingBox);
    
    maxcropregion = [cropregion(1:2), cropregion(1)+cropregion(3), cropregion(2)+cropregion(4)];
    
    % Bound our crop region to where the sum_map actually exists
    maxcropregion(maxcropregion<1) = 1;
    if maxcropregion(4)>size(sum_map,1)
        maxcropregion(4) = size(sum_map,1);
    end
    if maxcropregion(3)>size(sum_map,2)
        maxcropregion(3) = size(sum_map,2);
    end    
    
    average_frm_mask = sum_map >= ceil(mean(sum_map(:)));
    % Find the largest incribed rectangle in this mask.
    [C, h, w, largest_rect] =FindLargestRectangles(average_frm_mask,[1 1 0], [300 150]);

    % Find the coordinates for each corner of the rectangle, and
    % return them
    cropregion = regionprops(largest_rect,'BoundingBox');
    cropregion = ceil(cropregion.BoundingBox);

    cropregion = [cropregion(1:2), cropregion(1)+cropregion(3), cropregion(2)+cropregion(4)];

    % Bound our crop region to where the sum_map actually exists
    cropregion(cropregion<1) = 1;
    if cropregion(4)>=size(sum_map,1)
        cropregion(4) = size(sum_map,1);
    end
    if cropregion(3)>=size(sum_map,2)
        cropregion(3) = size(sum_map,2);
    end

    sum_map_crop = sum_map(cropregion(2):cropregion(4),cropregion(1):cropregion(3));

    figure(1); imagesc( sum_map_crop ); axis image; title('Ideal cropped sum map');


    %% Register these frames together, removing residual rotation.

    im_only_vid = cell(length(confocal_vid),1);
    reg_only_vid = cell(length(confocal_vid),1);
    split_im_only_vid = cell(length(confocal_vid),1);
    avg_im_only_vid = cell(length(confocal_vid),1);
    vis_im_only_vid = cell(length(confocal_vid),1);

    for n=1:length(confocal_vid)

        frm_nonzeros = imclose((confocal_vid{n}>0), ones(11));

        masked_frm = frm_nonzeros.*largest_rect;
        cropped_masked_frm = masked_frm( cropregion(2):cropregion(4),cropregion(1):cropregion(3) );


        im_only_vid{n} = confocal_vid{n}( cropregion(2):cropregion(4),cropregion(1):cropregion(3) );
        if loadsplit
            split_im_only_vid{n} = split_vid{n}( cropregion(2):cropregion(4),cropregion(1):cropregion(3) );
        end
        if loadavg
            avg_im_only_vid{n} = avg_vid{n}( cropregion(2):cropregion(4),cropregion(1):cropregion(3) );
        end
        if loadvisible
            vis_im_only_vid{n} = vis_vid{n}( cropregion(2):cropregion(4),cropregion(1):cropregion(3) );
        end
        
        reg_only_vid{n} = confocal_vid{n}( maxcropregion(2):maxcropregion(4),maxcropregion(1):maxcropregion(3) );
    end


    for n=1:length(im_only_vid)
        prc_filled(n) = sum( sum( imclose((im_only_vid{n}>0), ones(11)) ) )./ (size(im_only_vid{n},1)*size(im_only_vid{n},2));
    end

    max_fill = max(prc_filled);
    for n=1:length(im_only_vid) % Find the first frame that has the highest fill %.
        if prc_filled(n) == max_fill
            ref_ind =n;
            break;
        end
    end

    [optimizer, metric]  = imregconfig('monomodal');
    % optimizer.GradientMagnitudeTolerance = 1e-4;
    % optimizer.MinimumStepLength = 1e-5;
    % optimizer.MaximumStepLength = 0.04;
    % optimizer.MaximumIterations = 100;

    tic;

    forward_reg_tform = cell(length(reg_only_vid),1);
    % Register the image stack forward. It is more stable if we align to the 
    % frame with the largest percentage of the cropped region covered.
    parfor n=1:length(reg_only_vid)

        % Register using the cropped frame
        forward_reg_tform{n}=imregtform(reg_only_vid{n}, reg_only_vid{ref_ind},'affine',...
                                optimizer, metric,'PyramidLevels',1, 'InitialTransformation', affine2d());%,'DisplayOptimization',true);

        tforms(:,:,n) = forward_reg_tform{n}.T;
    end
    toc;

    mean_tforms = mean(tforms,3);

    % Determine the typical distance from the mean xform.
    for t=1:size(tforms,3)
        tfo = tforms(:,:,t);
        frob_dist(t) = abs(tfo(:)'*mean_tforms(:));
    end

    mean_frob_dist = mean(frob_dist);
    std_frob_dist = std(frob_dist);


    max_frob_dist = 3*std_frob_dist+mean_frob_dist;

    %%
    reg_confocal_vid = cell(length(confocal_vid),1);
    reg_split_vid = cell(length(confocal_vid),1);
    reg_avg_vid = cell(length(confocal_vid),1);
    reg_vis_vid = cell(length(confocal_vid),1);

    for f=1:length(im_only_vid)    

        if f~=ref_ind

            % We would NOT expect large changes here- so if there is a big
            % jump, use the surrounding transforms to force some stablility on
            % the registration.
            tfo = tforms(:,:,f);
            t=0;
            theend = false;
            while abs(tfo(:)'*mean_tforms(:)) > max_frob_dist
                tfo = tforms(:,:,f-t);
                t=t+1;

                if (f-t) == 0
                   theend = true;
                   break;
                end
            end

            if theend % If we reached the end in that direction, check the other direction
                t=1;
                while abs(tfo(:)'*mean_tforms(:)) > max_frob_dist  && (f+t) < length(im_only_vid)
                    tfo = tforms(:,:,f+t);
                    t=t+1;
                end
            end

            reg_confocal_vid{f}= imwarp(im_only_vid{f}, affine2d(tfo),'OutputView', imref2d(size(im_only_vid{1})) );
            
            if loadsplit
                reg_split_vid{f}= imwarp(split_im_only_vid{f}, affine2d(tfo),'OutputView', imref2d(size(im_only_vid{1})) );
            end
            if loadavg
                reg_avg_vid{f}= imwarp(avg_im_only_vid{f}, affine2d(tfo),'OutputView', imref2d(size(im_only_vid{1})) );
            end
            if loadvisible
                reg_vis_vid{f}= imwarp(vis_im_only_vid{f}, affine2d(tfo),'OutputView', imref2d(size(im_only_vid{1})) );
            end
     

        else
            reg_confocal_vid{f}= im_only_vid{f};
            if loadsplit
                reg_split_vid{f}= split_im_only_vid{f};
            end
            if loadavg
                reg_avg_vid{f}= avg_im_only_vid{f};
            end
            if loadvisible
                reg_vis_vid{f}= vis_im_only_vid{f};
            end
        end
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


    average_frm_mask = sum_map>mean2(sum_map);
    % Find the largest incribed rectangle in this mask.
    [C, h, w, largest_rect] =FindLargestRectangles(average_frm_mask,[1 1 0], [300 150]);

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
        split_vid_out = uint8( zeros( size(reg_split_vid{1},1), size(reg_split_vid{1},2), length(confocal_vid) ));
    end
    if loadavg
        avg_vid_out = uint8( zeros( size(reg_avg_vid{1},1), size(reg_avg_vid{1},2), length(confocal_vid) ));
    end
    if loadvisible
        vis_vid_out = uint8( zeros( size(reg_vis_vid{1},1), size(reg_vis_vid{1},2), length(confocal_vid) ));
    end

    for i=1:length(confocal_vid)

        confocal_vid_out(:,:,i) = uint8( reg_confocal_vid{i} );

        if loadsplit
            split_vid_out(:,:,i) = uint8( reg_split_vid{i} );
        end
        if loadavg
            avg_vid_out(:,:,i) = uint8( reg_avg_vid{i} );
        end
        if loadvisible
            vis_vid_out(:,:,i) = uint8( reg_vis_vid{i} );
        end

    end

    frmcount = ['_n' num2str(size(confocal_vid_out,3))];

    dlmwrite( fullfile(pathname, [acceptable_frame_fname_out frmcount '_piped_acceptable_frames.csv']),acc_frame_list);
    delete(fullfile(pathname,acceptable_frame_fname));
    
    confocal_vidobj = VideoWriter( fullfile(pathname, [confocal_fname_out frmcount '.avi']), 'Grayscale AVI' );

    if loadsplit
        split_vidobj = VideoWriter( fullfile(pathname, [split_fname_out frmcount '.avi']), 'Grayscale AVI' );
    end
    if loadavg
        avg_vidobj = VideoWriter( fullfile(pathname, [avg_fname_out frmcount '.avi']), 'Grayscale AVI' );
    end
    if loadvisible
        vis_vidobj = VideoWriter( fullfile(pathname, [vis_fname_out frmcount '.avi']), 'Grayscale AVI' );
    end


    open(confocal_vidobj);
    writeVideo(confocal_vidobj,confocal_vid_out);
    close(confocal_vidobj);

    if loadsplit
        open(split_vidobj);
        writeVideo(split_vidobj,split_vid_out);
        close(split_vidobj);
    end
    if loadavg
        open(avg_vidobj);
        writeVideo(avg_vidobj,avg_vid_out);
        close(avg_vidobj);
        delete(fullfile(pathname,avg_fname));
    end
    if loadvisible
        open(vis_vidobj);
        writeVideo(vis_vidobj,vis_vid_out);
        close(vis_vidobj);
        delete(fullfile(pathname,visible_fname));
    end

    % Write the average images.
    imwrite(uint8(sum(confocal_vid_out,3)./sum_map), fullfile(pathname, [confocal_fname_out frmcount '_AVG.tif']) );

    written_file = [confocal_fname_out frmcount '_AVG.tif'];
    written_path = pathname;
    
    if loadsplit        
        imwrite(uint8(sum(split_vid_out,3)./sum_map), fullfile(pathname, [split_fname_out frmcount '_AVG.tif']) );
        %Delete data from the last step of the pipeline.
        delete(fullfile(pathname,split_fname));
    end
    %Delete data from the last step of the pipeline.
    delete(fullfile(pathname,video_fname{k}));
    
    
    close all;
end
