function []=Relativize_Trials_Pipl(confocal_fname, pathname)

if ~exist('confocal_fname','var')
    %% Filename determination and handling
    [confocal_fname, pathname] = uigetfile('*.tif', 'Select the trial average images you wish to relativize:', 'MultiSelect','on');

    if ~iscell(confocal_fname)
        confocal_fname={confocal_fname};
    end

end

for i=1:length(confocal_fname)
    
    trial_im{i} = imread(fullfile(pathname,confocal_fname{i}));
    imsize(i,:) = size(trial_im{i});
end

%%
cropsize = min(imsize,[],1);

cropped_ims = zeros(cropsize(1),cropsize(2),length(confocal_fname));

for i=1:length(confocal_fname)
    
    residualleft = ceil((imsize(i,:)-cropsize)/2);
    residualright = floor((imsize(i,:)-cropsize)/2);
    
    world_ref{i} = imref2d(size(cropped_ims), [1+residualleft(2), (imsize(i,2)-residualright(2))], ...
                                              [1+residualleft(1), (imsize(i,1)-residualright(1))] );
    cropped_ims(:,:,i) = trial_im{i}( 1+residualleft(1):(imsize(i,1)-residualright(1)),...
                                      1+residualleft(2):(imsize(i,2)-residualright(2)) );
end

[optimizer, metric]  = imregconfig('multimodal');
optimizer.InitialRadius = 1.5e-03;
metric.NumberOfHistogramBins = 200;

[monooptimizer, monometric]  = imregconfig('monomodal');
monooptimizer.MaximumIterations = 250;
monooptimizer.RelaxationFactor = 0.6;
monooptimizer.GradientMagnitudeTolerance = 1e-5;

tforms = cell(length(trial_im),length(trial_im));

% Find the transform from each image to every other

thearea = prod(imsize,2);

[~, ref_im] = max(thearea);
confocal_fname{ref_im}

i=ref_im;
    for j=1:length(trial_im)
        if i~=j
            %%
            tic;
            confocal_fname{j}
%             tforms{i,j} = imregtform(trial_im{j},trial_im{ref_im},... 
%                                      'rigid',optimizer,metric, 'PyramidLevels',3);
            % First get close via ncc
            [xcorr_map , ~] = normxcorr2_general(trial_im{j}, trial_im{ref_im}, prod(mean([imsize(j,:);imsize(ref_im,:)])/2) );
            
            [~, ncc_ind] = max(xcorr_map(:));
            [roff, coff]= ind2sub(size(xcorr_map), ncc_ind );
            roff = roff-size(trial_im{j},1);
            coff = coff-size(trial_im{j},2);
            
            tforms{i,j} = affine2d([1 0 0; 0 1 0; coff roff 1]);
            
            tforms{i,j} = imregtform( trial_im{j}, trial_im{ref_im},... % Then tweak for affine
                                     'affine',monooptimizer,monometric, 'PyramidLevels',1,'InitialTransformation',tforms{i,j});
                                 
                                 
            toc;


%             reg = imwarp(trial_im{j}, imref2d(size(trial_im{j})), tforms{ref_im,j},'OutputView', imref2d(size(trial_im{ref_im})) );
%             figure(1);imshowpair(trial_im{ref_im},reg);
 
        end
    end
% end

%% Confocal Relativized stack
stk_name = 'Relativized_Trial_Stack.tif';
reg_ims = repmat(trial_im{ref_im},[1 1 length(trial_im)]);
imwrite(reg_ims(:,:,ref_im), fullfile(pathname, stk_name) );

for j=1:length(trial_im)
    if j ~= ref_im
        reg_ims(:,:,j) = imwarp(trial_im{j}, imref2d(size(trial_im{j})), tforms{ref_im,j}, 'OutputView', imref2d(size(trial_im{ref_im})) );
        imwrite(reg_ims(:,:,j),fullfile(pathname, stk_name),'WriteMode','append','Description',confocal_fname{j});                
%         figure(1); imshowpair(trial_im{ref_im}, reg_ims(:,:,j)); pause(.5)

        
    end
    imwrite(reg_ims(:,:,j), fullfile(pathname,[confocal_fname{j}(1:end-8) '_piped_AVG.tif']) );
%     delete(confocal_fname{j});
end


% Make a sum map, and create an average image.
sum_map = zeros(size(reg_ims(:,:,1)));

for f=1:size(reg_ims,3)
    frm_nonzeros = (reg_ims(:,:,f)>0); 
    sum_map = sum_map+frm_nonzeros;
end

% matchexp = '_\d*_ref_\d*_'; %String match that is expected to show up in the filename of each image. E.g. '_0018_ref_7_'
% avgname = regexprep(confocal_fname{1},matchexp,'_ALL_TRIALS_');
% matchexp = '_n\d*_';
% avgname = regexprep(avgname,matchexp,'_');

imwrite(uint8(sum(double(reg_ims),3)./sum_map), fullfile(pathname,[confocal_fname{i}(1:end-8) '_ALL_TRIALS.tif']));

%% Split Relativized stack
split_name_in = fullfile(pathname, strrep(confocal_fname{ref_im}, 'confocal', 'split_det'));

if exist(split_name_in,'file')
    stk_name = 'Split_Relativized_Trial_Stack.tif';
    
    split_im = imread(split_name_in);    
    
    split_reg_ims = repmat(split_im,[1 1 length(trial_im)]);
    imwrite(split_im, fullfile(pathname, stk_name));

    for j=1:length(trial_im)
        
        split_name_in = fullfile(pathname, strrep(confocal_fname{j}, 'confocal', 'split_det'));
        split_im = imread(split_name_in);
        
        if j ~= ref_im            
            
            split_reg_ims(:,:,j) = imwarp(split_im, imref2d(size(split_im)), tforms{ref_im,j}, 'OutputView', imref2d(size(trial_im{ref_im})) );
            imwrite(split_reg_ims(:,:,j),fullfile(pathname, stk_name),'WriteMode','append','Description',split_name_in);                
    %         figure(1); imshowpair(trial_im{ref_im}, reg_ims(:,:,j)); pause(.5)


        end
        imwrite(split_reg_ims(:,:,j), [split_name_in(1:end-8) '_piped_AVG.tif'] );
    %     delete(confocal_fname{j});
    end


    % Make a sum map, and create an average image.
    sum_map = zeros(size(split_reg_ims(:,:,1)));

    for f=1:size(split_reg_ims,3)
        frm_nonzeros = (split_reg_ims(:,:,f)>0); 
        sum_map = sum_map+frm_nonzeros;
    end

    % matchexp = '_\d*_ref_\d*_'; %String match that is expected to show up in the filename of each image. E.g. '_0018_ref_7_'
    % avgname = regexprep(confocal_fname{1},matchexp,'_ALL_TRIALS_');
    % matchexp = '_n\d*_';
    % avgname = regexprep(avgname,matchexp,'_');

    imwrite(uint8(sum(double(split_reg_ims),3)./sum_map), fullfile(pathname, [strrep(confocal_fname{ref_im}(1:end-8), 'confocal', 'split_det') '_ALL_TRIALS.tif']) );
end

%% Transform all of the videos associated with each avg image
for i=1:length(confocal_fname)
    mov_name_in = fullfile(pathname,[confocal_fname{i}(1:end-8) '.avi']);
    mov_name_out = fullfile(pathname,[confocal_fname{i}(1:end-8) '_piped.avi']);

    if exist(mov_name_in,'file')
        confocal_vidin = VideoReader( mov_name_in );
        confocal_vidout = VideoWriter( mov_name_out, 'Grayscale AVI' );
        confocal_vidout.FrameRate = 16.6;
        open(confocal_vidout);    
        while hasFrame(confocal_vidin)
            frm_in = readFrame(confocal_vidin);

            if ~isempty(tforms{ref_im,i})
                writeVideo( confocal_vidout, imwarp(frm_in, imref2d(size(frm_in)), tforms{ref_im,i},...
                                                    'OutputView', imref2d(size(trial_im{ref_im}))) );
            else
                writeVideo( confocal_vidout, frm_in );
            end
        end
        close(confocal_vidout);

        avg_name_in = [confocal_fname{i}(1:end-8) '.avi'];    
        avg_name_in = fullfile(pathname, strrep(avg_name_in, 'confocal', 'avg'));

        if exist(avg_name_in,'file')
            avg_name_out = [confocal_fname{i}(1:end-8) '_piped.avi'];
            avg_name_out = fullfile(pathname, strrep(avg_name_out, 'confocal', 'avg'));

            avg_vidin = VideoReader( avg_name_in );
            avg_vidout = VideoWriter( avg_name_out, 'Grayscale AVI' );
            avg_vidout.FrameRate = 16.6;
            open(avg_vidout);    
            while hasFrame(avg_vidin)
                frm_in = readFrame(avg_vidin);

                if ~isempty(tforms{ref_im,i})
                    writeVideo( avg_vidout, imwarp(frm_in, imref2d(size(frm_in)), tforms{ref_im,i},...
                                                        'OutputView', imref2d(size(trial_im{ref_im}))) );
                else
                    writeVideo( avg_vidout, frm_in );
                end
            end
            close(avg_vidout);

        end

        split_name_in = [confocal_fname{i}(1:end-8) '.avi'];    
        split_name_in = fullfile(pathname, strrep(split_name_in, 'confocal', 'split_det'));

        if exist(split_name_in,'file')
            split_name_out = [confocal_fname{i}(1:end-8) '_piped.avi'];
            split_name_out = fullfile(pathname, strrep(split_name_out, 'confocal', 'split_det'));

            split_vidin = VideoReader( split_name_in );
            split_vidout = VideoWriter( split_name_out, 'Grayscale AVI' );
            split_vidout.FrameRate = 16.6;
            open(split_vidout);    
            while hasFrame(split_vidin)
                frm_in = readFrame(split_vidin);

                if ~isempty(tforms{ref_im,i})
                    writeVideo( split_vidout, imwarp(frm_in, imref2d(size(frm_in)), tforms{ref_im,i},...
                                                        'OutputView', imref2d(size(trial_im{ref_im}))) );
                else
                    writeVideo( split_vidout, frm_in );
                end
            end
            close(split_vidout);

        end

        vis_name_in = [confocal_fname{i}(1:end-8) '.avi'];    
        vis_name_in = fullfile(pathname, strrep(vis_name_in, 'confocal', 'visible'));

        if exist(vis_name_in,'file')
            vis_name_out = [confocal_fname{i}(1:end-8) '_piped.avi'];
            vis_name_out = fullfile(pathname, strrep(vis_name_out, 'confocal', 'visible'));

            vis_vidin = VideoReader( vis_name_in );
            vis_vidout = VideoWriter( vis_name_out, 'Grayscale AVI' );
            vis_vidout.FrameRate = 16.6;
            open(vis_vidout);    
            while hasFrame(vis_vidin)
                frm_in = readFrame(vis_vidin);

                if ~isempty(tforms{ref_im,i})
                    writeVideo( vis_vidout, imwarp(frm_in, imref2d(size(frm_in)), tforms{ref_im,i},...
                                                        'OutputView', imref2d(size(trial_im{ref_im}))) );
                else
                    writeVideo( vis_vidout, frm_in );
                end
            end
            close(vis_vidout);

        end
    else
        warning(['Unable to find matching confocal video: ' [confocal_fname{i}(1:end-8) '.avi']]);
    end
%     delete(mov_name_in);
    
end


