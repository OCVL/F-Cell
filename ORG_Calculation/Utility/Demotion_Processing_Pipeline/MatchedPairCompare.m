addpath(genpath('C:\Users\dontm\Dropbox (Personal)\Research\Projects\AOAutomontaging\AOAutomontaging\SupportFunctions'))

%Load Pair Information

pairsInfoFileLoc = 'C:\Users\dontm\Documents\ResearchResults\AOLongitudinal\ArvoExperiments\Data\MatchingPairs\Pairs_ID.xlsx';
ImageADir = 'C:\Users\dontm\Documents\ResearchResults\AOLongitudinal\ArvoExperiments\Data\MatchingPairs\A';
ImageBDir = 'C:\Users\dontm\Documents\ResearchResults\AOLongitudinal\ArvoExperiments\Data\MatchingPairs\B';

outputDir = 'C:\Users\dontm\Documents\ResearchResults\AOLongitudinal\ArvoExperiments\Results\MatchedPairCompare\test';
[temp,temp2,C] = xlsread(pairsInfoFileLoc);

N = size(C,1);
MN = 1;
ModalitiesSrchStrings = {'confocal'; 'split'; 'avg'};

FilenamesA = cell(MN,N);%stores all filenames
FilenamesB = cell(MN,N);%stores all filenames
SubID = cell(1,N);
ImageAID = cell(1,N);
ImageBID = cell(1,N);

%Create data structures
for n = 1:N
    SubID{n} = num2str(C{n,1});
    ImageAID{n} = C{n,2};
    ImageBID{n} = C{n,3};
    for m = 1:MN
        ImageA = dir(fullfile(ImageADir,['*' SubID{n} '*' ModalitiesSrchStrings{m} '*' ImageAID{n} '*']));
        ImageA = ImageA(1).name;
        ImageB = dir(fullfile(ImageBDir,['*' SubID{n} '*' ModalitiesSrchStrings{m} '*' ImageBID{n} '*']));
        ImageB = ImageB(1).name;
        %Calculate Features
        FilenamesA{m,n} = fullfile(ImageADir,ImageA);
        FilenamesB{m,n} = fullfile(ImageBDir,ImageB);
    end
end




for q = 3
    
    switch q
        case 1
            methodname = 'SIFT';
            featureType = 0;
            MN = 1;
        case 2
            methodname = 'SIFT_multimodal';
            featureType = 0;
            MN = 3;
        case 3
            methodname = 'Grid';
            featureType = 1;
            MN = 1;
        case 4

            methodname = 'CC';
            
    end
    
       
    if (q<4)
        pixelScale = ones(1,N);
        parallelFlag = 1;
        [f_all_A, d_all_A, h] = calculateFeatures(FilenamesA, parallelFlag, pixelScale, featureType, MN, N);
        [f_all_B, d_all_B, h] = calculateFeatures(FilenamesB, parallelFlag, pixelScale, featureType, MN, N);
        output=struct('bestH',cell(1,N), 'numOkMatches',cell(1,N), 'numMatches',cell(1,N), 'bestScale', cell(1,N));
        TransType = 2;
        rotLimit = [];
        saveFlag = 1;
        im1 = cell(MN,1);
        im2 = cell(MN,1);
        for n = 1:N
            for m= 1:MN
                im1{m} = imread(char(FilenamesA{m,n}));
                im2{m} = imread(char(FilenamesB{m,n}));
            end
            saveDir = fullfile(outputDir,methodname,['Sub_' SubID{n} '_timeAref' ImageAID{n} '_matchto_timeBref' ImageBID{n}]);
            mkdir(saveDir);
            [bestH, numOkMatches, numMatches, bestScale]= sift_mosaic_fast_MultiModal(im1, im2, saveDir,saveFlag,f_all_A(1:MN,n),d_all_A(1:MN,n),f_all_B(1:MN,n),d_all_B(1:MN,n),TransType,[],featureType);
            output(n).bestH = bestH;
            output(n).numOkMatches = numOkMatches;
            output(n).numMatches = numMatches;
            output(n).bestScale = bestScale;
        end
    else
        output=struct('bestH',cell(1,N), 'finalCC',cell(1,N));
                im1 = imread(char(FilenamesA{m,n}));
                im2 = imread(char(FilenamesB{m,n}));
                
                movingRegistered = imregister(moving, fixed, 'affine', optimizer, metric);


    end
    
    
    save(fullfile(outputDir,methodname,'ExperimentSave'),'output','SubID','ImageAID',...
    'ImageAID','FilenamesA','FilenamesB',...
    'f_all_A', 'd_all_A','f_all_B', 'd_all_B');

    
end
