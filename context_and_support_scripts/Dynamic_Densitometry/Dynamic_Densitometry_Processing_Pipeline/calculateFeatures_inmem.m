function [f_all, d_all] = calculateFeatures_inmem(imagestack, featureType)
%Calculate features for all images listed in imageFilename

%feature parameters
SiftLevel = 55; %The number of levels to use in SIFT, default
ROICropPct = 0; %Sets a percentage crop on the boundaries of the image, where SIFT features are

%stores features for each image
f_all = cell(1, length(imagestack));
d_all = cell(1, length(imagestack));

if (featureType == 0) 
FeatureName = 'SIFT';
elseif (featureType == 1)
FeatureName = 'Constellation';
end


h = waitbar(0,['Calculating ' FeatureName ' Features.']);
parfor n=1:length(imagestack)
    
    im = imagestack{n};
    if(featureType == 0)
        [f1,d1] = vl_sift(im(:,:,1),'Levels',SiftLevel);
    elseif(featureType == 1)
        [f1,d1] = gridFeatures(im(:,:,1));
    else
        [f1,d1] = vl_sift(im(:,:,1),'Levels',SiftLevel);
    end
    [f1_crop, d1_crop] = filterSiftFeaturesByROI(im, f1, d1, ROICropPct);
    f_all{n} = f1_crop;
    d_all{n} = d1_crop;    
    
%     waitbar(n/(length(imagestack)),h,['Calculating ' FeatureName ...
%                     ' Features (' num2str(100*n/length(imagestack),3) '%)']);
end

close(h)
end

