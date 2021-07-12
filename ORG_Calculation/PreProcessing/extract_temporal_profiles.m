function [temporal_profiles]=extract_temporal_profiles(temporal_data, varargin)


p = inputParser;

dataset_size = size(temporal_data);

if length(dataset_size) > 3
   fourDeeData = true; 
end

[X_grid, Y_grid]= meshgrid(1:dataset_size(1), 1:dataset_size(2));

defaultcoords = [X_grid(:) Y_grid(:)];
checkcoords = @(x) size(x,2) == 2; % the coordinate list should be Nx2. 

defaultmethod = 'box';
validmethods = {'box', 'voronoi'};
checkMethods = @(x) any(validatestring(x,validmethods));

defaultextmethod = 'mean';
validextmethods = {'mean', 'median', 'glcm_ent', 'glcm_ent'};
checkExtMethods = @(x) any(validatestring(x,validextmethods));

defaultradius = 2;

addRequired(p,'temporal_data', @isnumeric)
addOptional(p,'Coordinates', defaultcoords, checkcoords);
addOptional(p,'SegmentationMethod', defaultmethod, checkMethods);
addOptional(p,'SegmentationRadius', defaultradius, @isnumeric); %Unused for voronoi.
addOptional(p,'ExtractionMethod', defaultextmethod, checkExtMethods);
addOptional(p,'ProgBarHandle', [], @ishandle);

% Parse our inputs.
parse(p,temporal_data,varargin{:})

segmentation_method = p.Results.SegmentationMethod;
extraction_method = p.Results.ExtractionMethod;
roiradius = p.Results.SegmentationRadius;
coordinates = p.Results.Coordinates;

if ~isempty(p.UsingDefaults.ProgBarHandle)
    wbh = p.Results.ProgBarHandle;
else
    wbh = waitbar(0,'Segmenting coordinate 0');
end


if strcmp(segmentation_method, 'voronoi')
    [V, C]=voronoin(coordinates);
    [X, Y ] = meshgrid(1:size(ref_image,2), 1:size(ref_image,1));
end
coordinates = round(coordinates);


cellseg_inds = cell(size(coordinates,1),1);

for i=1:size(coordinates,1)

    waitbar(i/size(coordinates,1),wbh, ['Segmenting coordinate ' num2str(i)]);

    switch segmentation_method
        case 'box'
            if (coordinates(i,1) - roiradius) > 1 && (coordinates(i,1) + roiradius) < size(mask_image,2) &&...
               (coordinates(i,2) - roiradius) > 1 && (coordinates(i,2) + roiradius) < size(mask_image,1)

                [R, C ] = meshgrid((coordinates(i,2) - roiradius) : (coordinates(i,2) + roiradius), ...
                                   (coordinates(i,1) - roiradius) : (coordinates(i,1) + roiradius));

                cellseg_inds{i} = sub2ind( size(mask_image), R, C );

                cellseg_inds{i} = cellseg_inds{i}(:);

            end
        case 'voronoi'
            vertices = V(C{i},:);

            if (all(C{i}~=1)  && all(vertices(:,1)<size(ref_image,2)) && all(vertices(:,2)<size(ref_image,1)) ... % [xmin xmax ymin ymax] 
                             && all(vertices(:,1)>=1) && all(vertices(:,2)>=1)) 

                [in, on] = inpolygon(X(:), Y(:), vertices(:,1), vertices(:,2));

                cellseg_inds{i} = sub2ind( size(mask_image), Y(in|on), X(in|on) );

                cellseg_inds{i} = cellseg_inds{i}(:);


            end
    end
end

temporal_profiles = cell( length(cellseg_inds), 1);

if ~isempty(p.UsingDefaults.ProgBarHandle)
    waitbar(0,wbh, 'Creating reflectance profile for cell: 0');
else
    wbh = waitbar(0,'Creating reflectance profile for cell: 0');
end


switch segmentation_method
    case 'box' 
        switch extraction_method
            case 'mean' % Shorthand, but way faster than the naive implementation, for box extraction only. (>2x speedup)
                for i=1:length(cellseg_inds) 

                    if ~isempty(cellseg_inds{i})

                        temporal_profiles{i} = zeros(1, dataset_size(end));

                        [m,n] = ind2sub(vid_size, cellseg_inds{i});

                        thisstack = temporal_data(min(m):max(m), min(n):max(n),:);        

                        thisstack(thisstack == 0) = NaN;

                        thisstack = sum(thisstack,1);
                        thisstack = squeeze(sum(thisstack,2));
                        thisstack = thisstack./ (length(cellseg_inds{i})*ones(size(thisstack)));


                        temporal_profiles{i} = thisstack';
                    end
                end
            case 'median'
                for i=1:length(cellseg_inds)
                    if ~isempty(cellseg_inds{i})
                    temporal_profiles{i} = zeros(1, dataset_size(end));

                    for t=1:size(temporal_data,3)

                        masked_timepoint = temporal_data(:,:,t); 

                        if all( masked_timepoint(cellseg_inds{i}) ~= 0 )
                            temporal_profiles{i}(t) = median( masked_timepoint(cellseg_inds{i}));
                        else            
                            temporal_profiles{i}(t) =  NaN;
                        end
                    end
                    end
                end
            case 'glcm'
                
        end
    case 'voronoi'
        switch extraction_method
            case 'mean'
                for i=1:length(cellseg_inds)
                    if ~isempty(cellseg_inds{i})
                    temporal_profiles{i} = zeros(1, dataset_size(end));

                    for t=1:size(temporal_data,3)

                        masked_timepoint = temporal_data(:,:,t); 

                        if all( masked_timepoint(cellseg_inds{i}) ~= 0 )
                            temporal_profiles{i}(t) = mean( masked_timepoint(cellseg_inds{i}));
                        else            
                            temporal_profiles{i}(t) =  NaN;
                        end
                    end
                    end
                end
            case 'median'
                for i=1:length(cellseg_inds)
                    if ~isempty(cellseg_inds{i})
                    temporal_profiles{i} = zeros(1, dataset_size(end));

                    for t=1:size(temporal_data,3)

                        masked_timepoint = temporal_data(:,:,t); 

                        if all( masked_timepoint(cellseg_inds{i}) ~= 0 )
                            temporal_profiles{i}(t) = median( masked_timepoint(cellseg_inds{i}));
                        else            
                            temporal_profiles{i}(t) =  NaN;
                        end
                    end
                    end
                end
            case 'glcm'
                
        end
end

% If we didn't supply a progress bar, close it at the end to be a good
% neighbor.
if isempty(p.UsingDefaults.ProgBarHandle)
   close(wbh) 
end

end