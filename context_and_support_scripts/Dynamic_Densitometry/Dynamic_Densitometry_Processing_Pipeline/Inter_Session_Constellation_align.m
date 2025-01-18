% addpath(genpath('C:\Users\dontm\Dropbox (Personal)\Research\Projects\AOAutomontaging\AOAutomontaging\SupportFunctions'))

if ~exist('vl_sift')
    error('Didn''t detect vl_sift on the path. Make sure to add SupportFunctions to the path!');
end

[reference_fname, reference_path] = uigetfile(fullfile(pwd, '*.tif') , 'Select the ALL_TRIALs image you''d like to use a reference.');

if exist(fullfile(reference_path, [reference_fname(1:end-4) '_coords.csv']), 'file')
    ref_coords = dlmread( fullfile(reference_path, [reference_fname(1:end-4) '_coords.csv']) );
else
    error('Unable to find coords for the reference image, unable to complete constellation alignment!');
end

% Determine the date that we're aligning to from the reference fname
% SHOULD BE AFTER MCW-style ID.
[~, remain]=strtok(reference_fname, '_');
[~, remain]=strtok(remain, '_');
[aligndate, remain]=strtok(remain, '_');


locInd=1; % Location index

align_fnames = {};
align_path={};
align_fullfile={};
addanother=1;

sel_path=pwd;
while addanother ~= 0

    [sel_fname, sel_path] = uigetfile(fullfile(sel_path, '*.tif') ,'Select the image(s) you''d like to align to the reference frame.', 'MultiSelect', 'on');

    if sel_path == 0
       break; 
    end
    
    if ~iscell(sel_fname)
       sel_fname = {sel_fname};
       sel_path = {sel_path};
    end
    
    align_fnames = [align_fnames; sel_fname];
    align_path = [align_path; sel_path];
    align_fullfile = [align_fullfile fullfile(sel_path, sel_fname)];
        
    choice = questdlg('Would you like to select another set of images?','','Yes!','No, thank you.','Yes!');
    
    addanother = strcmp(choice,'Yes!');
end
clear names;
%%
num_a_im = size(align_fullfile,2);

% Attempt to align the reference image to all of the other images.
methodname = 'Grid';
featureType = 1;
modalities = 1;

pixelScale = ones(1,num_a_im);
parallelFlag = 0;
[f_all_A, d_all_A, h] = calculateFeatures({fullfile(reference_path,reference_fname)}, parallelFlag, pixelScale, featureType, modalities, 1);
close(h)
[f_all_B, d_all_B, h] = calculateFeatures(align_fullfile, parallelFlag, pixelScale, featureType, modalities, num_a_im);
close(h)

TransType = 3;
saveFlag = 1;
%% Determine the offset between each of the datasets
ref_image = imread( fullfile(reference_path,reference_fname) );

tforms = cell(1,num_a_im);
align_image = cell(1,num_a_im);
removelist = false(size(ref_coords,1),1);

for n = 1:num_a_im
    
    align_image{n} = imread(align_fullfile{n});
    

    [bestH, numOkMatches, numMatches, bestScale]= sift_mosaic_fast_MultiModal({ref_image}, {align_image{n}},...
                                                                              pwd, saveFlag,...
                                                                              f_all_A, d_all_A,...
                                                                              f_all_B(n),d_all_B(n),TransType,[],featureType);
    tforms{n} = affine2d(inv(bestH)');

    
    tif_name_out = fullfile(align_path{n},[align_fnames{n}(1:end-4) '_to_' num2str(aligndate) '.tif']);
    
    warped_im = imwarp(align_image{n}, imref2d(size(align_image{n})), tforms{n}, 'OutputView', imref2d(size(ref_image)) );
    
    imwrite(uint8(warped_im), tif_name_out);
    
    warped_coords{n} = transformPointsForward(affine2d(bestH'),ref_coords);
    
    removelist = removelist | (warped_coords{n}(:,1) < 1 | warped_coords{n}(:,2) < 1 ...
                            | warped_coords{n}(:,1) > size(align_image{n},2)-1 ...
                            | warped_coords{n}(:,2) > size(align_image{n},1)-1);
end


%% Write the moved coords.

dlmwrite(fullfile(reference_path, [reference_fname(1:end-4) '_coords.csv']), ref_coords(~removelist,:), 'delimiter',',')
for n = 1:num_a_im
    
    these_coords = warped_coords{n}(~removelist,:);
%     this_image = align_image{n};
% %     [fx,fy]=gradient(double(this_image));
% %     lapacie = del2(double(this_image));
%     for c=1:size(these_coords,1)
%         
%         thiscoord = round(these_coords(c,:));
%         offc = -10;
%         offr = -10;
%         if all(thiscoord>1) && (thiscoord(1) < size(this_image,2)-1) && (thiscoord(2) < size(this_image,1)-1) 
% %               (offc~=0 || offr~=0)
%             
%         % 1st iteration
%         thisroi = this_image(thiscoord(2)-1:thiscoord(2)+1, thiscoord(1)-1:thiscoord(1)+1);
% 
%         [val, ind]=max(thisroi(:));
%         [offr, offc]=ind2sub([3 3],ind);
% 
%         offc=(offc-2);
%         offr=(offr-2);
% 
%         thiscoord(1) = thiscoord(1)+offc;
%         thiscoord(2) = thiscoord(2)+offr;
%             
%             if all(thiscoord>1) && (thiscoord(1) < size(this_image,2)-1) && (thiscoord(2) < size(this_image,1)-1) 
%                 % 2nd iteration
%                 thisroi = this_image(thiscoord(2)-1:thiscoord(2)+1, thiscoord(1)-1:thiscoord(1)+1);
% 
%                 [val, ind]=max(thisroi(:));
%                 [offr, offc]=ind2sub([3 3],ind);
% 
%                 offc=(offc-2);
%                 offr=(offr-2);
% 
%                 thiscoord(1) = thiscoord(1)+offc;
%                 thiscoord(2) = thiscoord(2)+offr;
%             end
%         end
%         these_coords(c,:) = thiscoord;
%     end
    
    
    dlmwrite([align_fullfile{n}(1:end-4) '_coords.csv'], these_coords, 'delimiter',',')
    
end


%% If we need to grab the images, use this
%%% Transform all of the videos associated with each avg image


% Loop through all of the files we aligned to the reference
% f=1;

% %% Grab all of the tifs, transform them
% confocal_fname = read_folder_contents(align_path{f},'tif');
% matchexp = '_piped_AVG.tif';
% confocal_fname = confocal_fname( ~cellfun(@isempty,regexp(confocal_fname,matchexp)) );
% 
% for i=1:length(confocal_fname)
%     tif_name_in = fullfile(align_path{f},confocal_fname{i});
%     tif_name_out = fullfile(align_path{f},[confocal_fname{i}(1:end-8) '_to_' num2str(aligndate) '_AVG.tif']);
%     
%     tif_im = imread(tif_name_in);
%     
%     warped_im = imwarp(tif_im, imref2d(size(tif_im)), tforms{f}, 'OutputView', imref2d(size(ref_image)) );
%     
%     imwrite(uint8(warped_im), tif_name_out);
% end
% 
% %% Grab all of the avis, transform them
% confocal_fname = read_folder_contents(align_path{f},'avi');
% matchexp = '_piped.avi';
% confocal_fname = confocal_fname( ~cellfun(@isempty,regexp(confocal_fname,matchexp)) );
% 
% for i=1:length(confocal_fname)
%     mov_name_in = fullfile(align_path{f},confocal_fname{i});
%     mov_name_out = fullfile(align_path{f},[confocal_fname{i}(1:end-4) '_to_' num2str(aligndate) '.avi']);
% 
%     confocal_vidin = VideoReader( mov_name_in );
%     confocal_vidout = VideoWriter( mov_name_out, 'Grayscale AVI' );
%     confocal_vidout.FrameRate = 16.6;
%     open(confocal_vidout);    
%     while hasFrame(confocal_vidin)
%         frm_in = readFrame(confocal_vidin);
%         
%         if ~isempty(tforms{f})
%             writeVideo( confocal_vidout, imwarp(frm_in, imref2d(size(frm_in)), tforms{f},...
%                                                 'OutputView', imref2d(size(ref_image))) );
%         else
%             writeVideo( confocal_vidout, frm_in );
%         end
%     end
%     close(confocal_vidout);
% 
% end


    


