function [masked_temporal_data]=Residual_Torsion_Removal_Pipl(temporal_data, mask_data)
% Robert F Cooper 7-12-2021


    [optimizer, metric]  = imregconfig('monomodal');
    % optimizer.GradientMagnitudeTolerance = 1e-4;
    % optimizer.MinimumStepLength = 1e-5;
    % optimizer.MaximumStepLength = 0.04;
    % optimizer.MaximumIterations = 100;

    tic;
    if ~isempty(mask_data)
        masked_temporal_data = mask_data.*temporal_data;
    else
        warning(['No mask data supplied. Registration may be inaccurate.']);
        masked_temporal_data = temporal_data;
    end
    
    forward_reg_tform = cell(length(temporal_data),1);
    % Register the image stack forward. It is more stable if we align to the 
    % frame with the largest percentage of the cropped region covered.
    tforms(:,:,1)=affine2d().T;
    parfor n=2:size(masked_temporal_data,3)

        % Register using the cropped frame
        forward_reg_tform{n}=imregtform(masked_temporal_data(:,:,n), masked_temporal_data(:,:,1),'affine',...
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


    max_frob_dist = std_frob_dist+mean_frob_dist;

    %%
%     figure(1);
%     imagesc(masked_temporal_data(:,:,1));
    for f=2:size(temporal_data,3)    

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
                while abs(tfo(:)'*mean_tforms(:)) > max_frob_dist  && (f+t) < size(temporal_data,3)
                    tfo = tforms(:,:,f+t);
                    t=t+1;
                end
            end

            
            
            masked_temporal_data(:,:,f)= imwarp(masked_temporal_data(:,:,f), affine2d(tfo),'OutputView', imref2d(size(masked_temporal_data(:,:,f))) );
%             imshowpair(masked_temporal_data(:,:,f-1), masked_temporal_data(:,:,f) );
%             if f==107
%                 tfo
%                 disp('what')
%                 
%     end
%         imagesc(masked_temporal_data(:,:,f)); colormap gray; drawnow;pause(1/29.466)
    end


  
end
