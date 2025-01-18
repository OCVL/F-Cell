function [f_all, d_all] = calculateGRIDFeatures(image, featureType)
%Calculate features for all images listed in imageFilename

    %feature parameters
    SiftLevel = 55; %The number of levels to use in SIFT, default
    ROICropPct = 0; %Sets a percentage crop on the boundaries of the image, where SIFT features are

    %stores features for each image
    f_all = {};
    d_all = {};

    if (featureType == 0) 
    FeatureName = 'SIFT';
    elseif (featureType == 1)
    FeatureName = 'Constellation';
    end

    im = im2single(image);
    if(featureType == 0)
        [f1,d1] = vl_sift(im(:,:,1),'Levels',SiftLevel);
    elseif(featureType == 1)
        [f1,d1] = gridFeatures(im(:,:,1));
    else
        [f1,d1] = vl_sift(im(:,:,1),'Levels',SiftLevel);
    end
    [f1_crop, d1_crop] = filterSiftFeaturesByROI(im, f1, d1, ROICropPct);
    f_all{1} = f1_crop;
    d_all{1} = d1_crop;

end

