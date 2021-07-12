function [xform]=constellation_align(ref_image, align_image)

    if ~exist('vl_sift')
        error('Didn''t detect vl_sift on the path. Make sure to add SupportFunctions to the path!');
    end

    % Attempt to align the reference image to all of the other images.
    methodname = 'Grid';
    featureType = 1;

    [f_all_A, d_all_A] = calculateGRIDFeatures(ref_image, featureType);
    [f_all_B, d_all_B] = calculateGRIDFeatures(align_image, featureType);

    TransType = 3;
    saveFlag = 0;

    [bestH, numOkMatches, numMatches, bestScale]= sift_mosaic_fast_MultiModal({ref_image}, {align_image},...
                                                                              [], saveFlag,...
                                                                              f_all_A,d_all_A,...
                                                                              f_all_B,d_all_B, TransType,[],featureType);

    xform = bestH;
    
    
end

