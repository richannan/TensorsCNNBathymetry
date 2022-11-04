%% Brief Description and Warning
% This is a MATLAB demo of a deep learning (CNN) bathymetry recovery method 
% in a research currently under review. 
%% 
% * The inputs for the CNN are 1 arc-minute images constructed from full tensors 
% of gravity gradient.
% * This demo is performed in a region that encompasses longitudes -1 ~ 60 and 
% latitudes -31 ~ 30. 
% * The code relies heavily on the MATLAB wrapper for GMT (Generic Mapping Tools). 
% You only need to install GMT 6.3.0 or above, available at <https://github.com/GenericMappingTools/gmt/releases 
% GMT6> and introduced at <https://github.com/GenericMappingTools/gmtmex/wiki 
% GMT-MATLAB> .
% * This demo was done on a computer with 64 GB RAM, i7-12700H CPU and RTX 3050Ti 
% GPU. It has been successfully tested on MATLAB R2020b and above.
% * There is a possibility of running out of memory as the demo area is large. 
% This can crash MATLAB or even your computer.
% * You are advised to cut out an area smaller than, or preferably half of, 
% this demo area if your computer has 32 GB RAM.
% * The demo region has 1,649,538 ship-borne depths. Regions with much higher 
% number of ship-borne depths (like the Pacific Ocean) would require a computer 
% with RAM bigger than 64 GB.
% * This demo took about three days to fully run it. Therefore, access to a 
% GPU superior to the one used here is highly encouraged to speed up training 
% of the deep learning model.
% * You are advised to run it section-by-section.
% * There are further instructions as you run the code.

addpath(genpath("C:\programs\gmt6\bin"))

%%%%%%%%%%%%%%%%%%%%%%%%%%% Resample the gravity gradients and normalize them %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
Txx = gmt('grdsample -rg -I15s Txx.nc');      Txx.z = fNormalizeData(Txx.z);
Tyy = gmt('grdsample -rg -I15s Tyy.nc');      Tyy.z = fNormalizeData(Tyy.z);
Tzz = gmt('grdsample -rg -I15s Tzz.nc');      Tzz.z = fNormalizeData(Tzz.z);
Txy = gmt('grdsample -rg -I15s Txy.nc');      Txy.z = fNormalizeData(Txy.z);
Txz = gmt('grdsample -rg -I15s Txz.nc');      Txz.z = fNormalizeData(Txz.z);
Tyz = gmt('grdsample -rg -I15s Tyz.nc');      Tyz.z = fNormalizeData(Tyz.z);

%%%%%%%%%%%%%% Normalize the ship-borne depths and exclude 10% for performance assessment %%%%%%%%%%%%
load Sounding.mat
Deepest = min(Sounding(:,3));
Sounding(:,3) = fNormalizeData(Sounding (:,3)); 

n = length(Sounding);
Percent = 0.10;
rng('default')
IndependentIndex = randperm(n, round(Percent * n));
RemainingIndex = setdiff(1:n, IndependentIndex);
RemainingSet = Sounding(RemainingIndex,:);
IndependentSet = Sounding(IndependentIndex,:);                     
Sounding = RemainingSet;

%%%%%%%%%%%%%%%%%%%%%%%%%%% Reformat ship-borne into grid structure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Region = '-R-1/60/-31/30';     % lon_1/lon_2/lat_1/lat_2
xyz2gridString = strcat('-I15s', {' '}, Region);
FullShipGrid = gmt('xyz2grd', xyz2gridString{1}, Sounding);  

%%%%%%%%%%%%%%%%%%%%%%%%%%% Construct grids of lons and lats, and normalize them %%%%%%%%%%%%%%%%%%%%%
[Lons, Lats] = meshgrid(FullShipGrid.x, FullShipGrid.y);
LonsNormalized = single(fNormalizeData(Lons));
LatsNormalized = single(fNormalizeData(Lats));

%%%%%%%%%%%%%%%%%%%%%%%%%%% Extract 4x4 grids at locations of ship-borne depths only %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
window = 4;
sz = window - 1;
[M, N] = size(FullShipGrid.z);
FullData = cell((M-sz)*(N-sz), 11);
k = 1;
for i = 1:M-sz
    for j = 1:N-sz                
        QQQ = FullShipGrid.z(i:i+sz,j:j+sz);
        if all(isnan(QQQ(:))) 
            continue
        else
            FullData{k,1} = LonsNormalized(i:i+sz,j:j+sz);
            FullData{k,2} = LatsNormalized(i:i+sz,j:j+sz);
            FullData{k,3} = Txx.z(i:i+sz,j:j+sz);
            FullData{k,4} = Tyy.z(i:i+sz,j:j+sz);
            FullData{k,5} = Tzz.z(i:i+sz,j:j+sz);
            FullData{k,6} = Txy.z(i:i+sz,j:j+sz);
            FullData{k,7} = Txz.z(i:i+sz,j:j+sz);
            FullData{k,8} = Tyz.z(i:i+sz,j:j+sz);
            zz = FullShipGrid.z(i:i+sz,j:j+sz);
            xx = Lons(i:i+sz,j:j+sz);
            yy = Lats(i:i+sz,j:j+sz);
            FullData{k,9} = mean(zz(:),"omitnan");
            FullData{k,10} = mean(xx(:),"omitnan");
            FullData{k,11} = mean(yy(:),"omitnan");
        end            
        k = k + 1;        
    end
end
EmptyFull = cellfun('isempty', FullData(:,1));
FullData(EmptyFull,:) = []; 

%%%%%%%%%%%%%%%%%%%%%%%%%%% Save needed variables, and clear temporary ones to free up memory %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save GravityGradients.mat Sounding Txx Tyy Tzz Txy Txz Tyz IndependentSet Deepest Region 
clear GravityGradients.mat Sounding Tx* Ty* Tz* IndependentSet FullShipGrid Lats* Lons* Remaining* EmptyFull


%%%%%%%%%%%%%%%%%%%%%%%%%%% Construct the images %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m, n] = size(QQQ);
channels = 8;    
nn = length(FullData);
XFull = zeros(m, n, channels, nn);  
for k = 1:nn
    XFull(:,:,:,k) = cat(3, FullData{k,1}, FullData{k,2}, FullData{k,3}, FullData{k,4}, ...
                            FullData{k,5}, FullData{k,6}, FullData{k,7}, FullData{k,8});
end
YFull = cell2mat(FullData(:,9));
FullXY = cell2mat(FullData(:,10:11));       

%%%%%%%%%%%%%%%%%%%%%%%%%%% Save the variables, and clear them to free up memory %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save('GravityGradients.mat', 'XFull', 'YFull', 'FullXY', '-append')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Split them into 80% training and 20% validation sets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
clear 
load('GravityGradients.mat', 'XFull', 'YFull')
n = length(YFull);    
Percent = 0.80; % try 0.60, 0.65, 0.70, 0.75 or 0.80
rng('default')
TrainNum = round(Percent * n);
TrainIndex = randperm(n, TrainNum);
TestIndex = setdiff(1:n, TrainIndex);

%%%%%%%%%%%%%%%%%%%% Design and train the CNN architecture %%%%%%%%%%%%%%%%%%%%
% This take days depending on your computing hardware, such as: GPU or large number of cores.

[m, n, channels, ~] = size(XFull);
Layers = [
    imageInputLayer([m, n, channels])
    convolution2dLayer(3,512,'Padding','same')
    batchNormalizationLayer
    reluLayer          
    maxPooling2dLayer(2,'Stride',2) 
    convolution2dLayer(3,512,'Padding','same')
    batchNormalizationLayer
    reluLayer         
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,512,'Padding','same')
    batchNormalizationLayer
    reluLayer          
    convolution2dLayer(3,1024,'Padding','same')
    batchNormalizationLayer
    reluLayer 
    convolution2dLayer(3,1024,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(1024)    
    reluLayer
    dropoutLayer(0.30) 
    fullyConnectedLayer(1)      
    regressionLayer];

MiniBatchSize  = 128;
CheckPointPath = pwd;
ValidationFreq = floor(numel(YFull(TrainIndex)) / MiniBatchSize);
Options = trainingOptions('sgdm', ...           
                          'MiniBatchSize',MiniBatchSize, ...
                          'MaxEpochs',25, ...
                          'InitialLearnRate',1e-3, ...
                          'LearnRateSchedule','piecewise', ...
                          'LearnRateDropFactor',0.1, ...
                          'LearnRateDropPeriod',15, ...
                          'Shuffle','every-epoch', ...
                          'ValidationData',{XFull(:,:,:,TestIndex), YFull(TestIndex)}, ...
                          'ValidationFrequency',ValidationFreq, ...
                          'Plots','training-progress', ...
                          'Verbose',false, ...
                          'CheckpointPath',CheckPointPath, ...
                          'ExecutionEnvironment','auto');

% You can manually stop training if the differences in RMSEs are negligible, especially from epoch 15 onwards. 
% In this demonstration, I allowed it to run to epoch 25.
[netModel, netInfo] = trainNetwork(XFull(:,:,:,TrainIndex), YFull(TrainIndex), Layers, Options);
save('GravityGradients.mat', 'netModel', 'netInfo', '-append') % save the model

%%
%%%%%%%%%%%%%%%%%%%% Performance Assessment using Independent set %%%%%%%%%%%%%%%%%%%%
clear
load GravityGradients.mat netModel Txx Tyy Tzz Txy Txz Tyz IndependentSet Deepest Region

xyz2gridString = strcat('-I15s', {' '}, Region);
IndependentShipGrid = gmt('xyz2grd', xyz2gridString{1}, IndependentSet);
[Lons, Lats] = meshgrid(IndependentShipGrid.x, IndependentShipGrid.y);
LonsNormalized = single(fNormalizeData(Lons));
LatsNormalized = single(fNormalizeData(Lats));

[M, N] = size(Txx.z);
window = 4;
sz = window - 1;
IndependentData = cell((M-sz)*(N-sz), 11);
k = 1;
for i = 1:M-sz
    for j = 1:N-sz 
        QQQ = IndependentShipGrid.z(i:i+sz,j:j+sz);
        if all(isnan(QQQ(:))) 
            continue
        else
            IndependentData{k,1} = LonsNormalized(i:i+sz,j:j+sz);
            IndependentData{k,2} = LatsNormalized(i:i+sz,j:j+sz);
            IndependentData{k,3} = Txx.z(i:i+sz,j:j+sz); 
            IndependentData{k,4} = Tyy.z(i:i+sz,j:j+sz); 
            IndependentData{k,5} = Tzz.z(i:i+sz,j:j+sz);
            IndependentData{k,6} = Txy.z(i:i+sz,j:j+sz);
            IndependentData{k,7} = Txz.z(i:i+sz,j:j+sz);
            IndependentData{k,8} = Tyz.z(i:i+sz,j:j+sz);
            zz = IndependentShipGrid.z(i:i+sz,j:j+sz);
            xx = Lons(i:i+sz,j:j+sz);
            yy = Lats(i:i+sz,j:j+sz);
            IndependentData{k,9} = mean(zz(:),"omitnan");
            IndependentData{k,10} = mean(xx(:),"omitnan");
            IndependentData{k,11} = mean(yy(:),"omitnan");
        end                 
        k = k + 1;        
    end
end
EmptyIndependent = cellfun('isempty', IndependentData(:,1));
IndependentData(EmptyIndependent,:) = [];
clear Lons* Lats* Tx* Ty* Tz* EmptyIndependent IndependentSet

nn = length(IndependentData);
[m, n] = size(QQQ);
channels = 8;
XIndependent = zeros(m, n, channels, nn);
for k = 1:nn
    XIndependent(:,:,:,k) = cat(3, IndependentData{k,1}, IndependentData{k,2}, IndependentData{k,3}, IndependentData{k,4}, ...
                                   IndependentData{k,5}, IndependentData{k,6}, IndependentData{k,7}, IndependentData{k,8});
end
YIndependent = cell2mat(IndependentData(:,9));
IndependentXY = cell2mat(IndependentData(:,10:11));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Performance assessment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PredictIndependent = predict(netModel, XIndependent);
Assessment(:,1:2) = IndependentXY;
Assessment(:,3:4) = [PredictIndependent, YIndependent] * Deepest;

figure;
scatter(Assessment(:,3), Assessment(:,4), 'r.'); hold on
plot(min(Assessment(:,4)):0, min(Assessment(:,4)):0, 'k'); hold off
xlabel("Predicted Depths (m)")
ylabel("Independent Depths (m)")
xlim([min(Assessment(:,4)), 0]) 
ylim([min(Assessment(:,4)), 0])

ErrorIndependent = (Assessment(:,3) - Assessment(:,4));

figure; histogram(ErrorIndependent, 500, "FaceColor","red"); grid
xlabel("Deviations (m)"); ylabel("Counts"); xlim([-400, 400])
title("Errors at independent unseen depths")
%%
%%%%%%%%%%%%%%%%%%%% Performance Assessment using Full set %%%%%%%%%%%%%%%%%%%%
clear
load('GravityGradients.mat', 'netModel', 'Region', 'XFull', 'YFull', 'FullXY', ...
     'Txx', 'Tyy', 'Tzz', 'Txy', 'Txz', 'Tyz', 'Deepest')

PredictFull = predict(netModel, XFull);
save BATHYCNN.mat Compare

Compare(:,1:2) = FullXY;
Compare(:,3:4) = [PredictFull, YFull] * Deepest;

ErrorFull = Compare(:,3) - Compare(:,4);

figure; histogram(ErrorFull, 500, "FaceColor","red"); grid
xlabel("Deviations (m)"); ylabel("Counts"); xlim([-400, 400])
title("Errors at all depths")
%%
%%%%%%%%%%%%%%%%%%%% Juxtapose predicted bathymetry with ship-borne bathymetry %%%%%%%%%%%%%%%%%%%%    
grdlandmaskString = strcat(Region, {' '},'-I15s -N1/NaN -Df');
LandMask = gmt('grdlandmask', grdlandmaskString{1});

blockmedianString = strcat(Region, {' '}, '-I1m');
surfaceString = strcat(Region, {' '}, '-I1m -T0.35');

CNNBathy = gmt('blockmedian', blockmedianString{1}, Compare(:,1:3));
CNNBathy = gmt('surface', surfaceString{1}, CNNBathy);
CNNBathy = gmt('grdsample', '-I15s', CNNBathy);
CNNBathy.z(isnan(LandMask.z)) = LandMask.z(isnan(LandMask.z));
CNNBathy.z(CNNBathy.z > 0) = nan;

ShipBathy = gmt('blockmedian', blockmedianString{1}, Compare(:,[1,2,4]));
ShipBathy = gmt('surface', surfaceString{1}, ShipBathy);
ShipBathy = gmt('grdsample', '-I15s', ShipBathy);
ShipBathy.z(isnan(LandMask.z)) = LandMask.z(isnan(LandMask.z));
ShipBathy.z(ShipBathy.z > 0) = nan;

figure; fImagescn(CNNBathy.x, CNNBathy.y, CNNBathy.z); cc = colorbar; colormap jet;
title("BATHY_{CNN}"); Lim = caxis; cc.Label.String = 'Depth (m)';

figure; fImagescn(ShipBathy.x, ShipBathy.y, ShipBathy.z); cc = colorbar; colormap jet;
title("Ship-borne Bathymetry"); caxis([Lim(1), Lim(2)]); cc.Label.String = 'Depth (m)';
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Extract the learnt bathymetric features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
load('GravityGradients.mat', 'netModel', 'XFull', 'YFull', 'FullXY', 'Deepest', 'Region')

n = length(YFull); 
rng('default')
Percent = 0.70; % try 0.60, 0.65, 0.70, 0.75 or 0.80
TrainNum = round(Percent * n);
TrainIndex = randperm(n, TrainNum);
TestIndex = setdiff(1:n, TrainIndex);

FeaturesTrain = activations(netModel, XFull(:,:,:,TrainIndex), 'fc_1', 'OutputAs','rows');
FeaturesTest = activations(netModel, XFull(:,:,:,TestIndex), 'fc_1', 'OutputAs','rows');
clear XFull

YTrain = YFull(TrainIndex);
YTest = YFull(TestIndex);
clear YFull

%%%%%%%%%%%%%%%% Matrix of extracted features is so huge. We reduced it to top 50 features %%%%%%%%%%%%%%%%%%%%%%%%%
%%% We use the reconstruction independent component analysis (RICA) function. 
%%% Takes some time and consumes memory. Depending on your hardware, you can run into "out-of-memory" error.

tic
qqq = 50;    % No. of features you want. Try 50, 100, 150 or 200
RICAModel = rica(FeaturesTrain, qqq, 'IterationLimit',400);
FeaturesTrain = transform(RICAModel, FeaturesTrain);
FeaturesTest = transform(RICAModel, FeaturesTest);
toc
%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Feature extraction bathymetry using Decision tree %%%%%%%%%%%%%%%%%%%%%%%%%
MLRegression = fitrtree(FeaturesTrain, YTrain);
save MLRegressionModel.mat MLRegression

YPredicted = predict(MLRegression, FeaturesTest);
PredictTest(:,3:4) = [YPredicted, YTest] * Deepest;
PredictTest(:,1:2) = FullXY(TestIndex,:);

PredictionErrors = PredictTest(:,3) - PredictTest(:,4);

figure; 
scatter(PredictTest(:,3), PredictTest(:,4), 'r.'); hold on
plot(min(PredictTest(:,4)):0, min(PredictTest(:,4)):0, 'k'); hold off
xlabel("Predicted Depths (m)")
ylabel("Test Depths (m)"); 

figure; histogram(PredictionErrors, 500, "FaceColor","red"); hold on
xlabel('Error (m)'); ylabel('Count'); xlim([-400, 400]); grid
title("Errors at unseen depths")

TrainPredicted = predict(MLRegression, FeaturesTrain);
PredictTrain(:,3:4) = [TrainPredicted, YTrain] * Deepest;
PredictTrain(:,1:2) = FullXY(TrainIndex,:);

PredictFull = [PredictTrain; PredictTest];
save BATHYFE.mat PredictFull

%%%%%%%%%%%%%%%%%%%% Performance Assessment using Full set %%%%%%%%%%%%%%%%%%%%
ErrorFull = PredictFull(:,3) - PredictFull(:,4);

figure; histogram(ErrorFull, 500, "FaceColor","red"); grid
xlabel("Deviations (m)"); ylabel("Counts"); xlim([-400, 400])
title("Errors at all depths")

%%%%%%%%%%%%%%%%%%%% Juxtapose predicted bathymetry with ship-borne bathymetry %%%%%%%%%%%%%%%%%%%%    
grdlandmaskString = strcat(Region, {' '},'-I15s -N1/NaN -Df');
LandMask = gmt('grdlandmask', grdlandmaskString{1});

blockmedianString = strcat(Region, {' '}, '-I1m');
surfaceString = strcat(Region, {' '}, '-I1m -T0.35');

FEBathy = gmt('blockmedian', blockmedianString{1}, PredictFull(:,1:3));
FEBathy = gmt('surface', surfaceString{1}, FEBathy);
FEBathy = gmt('grdsample', '-I15s', FEBathy);
FEBathy.z(isnan(LandMask.z)) = LandMask.z(isnan(LandMask.z));
FEBathy.z(FEBathy.z > 0) = nan;

ShipBathy = gmt('blockmedian', blockmedianString{1}, PredictFull(:,[1:2,4]));
ShipBathy = gmt('surface', surfaceString{1}, ShipBathy);
ShipBathy = gmt('grdsample', '-I15s', ShipBathy);
ShipBathy.z(isnan(LandMask.z)) = LandMask.z(isnan(LandMask.z));
ShipBathy.z(ShipBathy.z > 0) = nan;

figure; fImagescn(FEBathy.x, FEBathy.y, FEBathy.z); cc = colorbar; colormap jet;
title("BATHY_{FE}"); Lim = caxis; caxis([Lim(1), Lim(2)]); cc.Label.String = 'Depth (m)';

figure; fImagescn(ShipBathy.x, ShipBathy.y, ShipBathy.z); cc = colorbar; colormap jet;
title("Ship-borne Bathymetry"); cc.Label.String = 'Depth (m)';
%% Conclusions
%% 
% * From the plot of predicted vs test, we can see better performance in the 
% feature extraction bathymetry (${\mathrm{BATHY}}_{\mathrm{FE}}$) than in the 
% CNN-derived bathymetry (${\mathrm{BATHY}}_{\mathrm{CNN}}$). There is more noise 
% in ${\mathrm{BATHY}}_{\mathrm{CNN}}$ than in ${\mathrm{BATHY}}_{\mathrm{FE}}$.
% * Also, the histograms show that error margin in ${\mathrm{BATHY}}_{\mathrm{FE}}$ 
% is smaller than in ${\mathrm{BATHY}}_{\mathrm{CNN}}$.
% * Deep learning can be used to combine gravity gradient tensors easily. In 
% classical methods, this is quite a challenge.
% * Deep learning, specifically CNN, is computationally efficient to extract 
% bathymetric features for geodetic bathymetry recovery.