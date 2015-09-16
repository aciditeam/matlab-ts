function deepFunction(nbLayers, typePretrain, typeTrain, layers, binaryLayers, layerType, paramsType, preNames, preValues, traNames, traValues)
% Transform inputs needed
nbLayers = str2num(nbLayers);
layers = str2num(layers);
binaryLayers = str2num(binaryLayers);
disp(nbLayers);
disp(layers);
disp(binaryLayers);
userStruct = struct;
userStruct.pretrain = struct;
eval(['userStruct.pretrain.names = {' preNames '}']);
userStruct.pretrain.values = str2num(preValues);
eval(['userStruct.train.names = {' traNames '}']);
userStruct.train.values = str2num(traValues);
if (isdeployed == false)
    addpath(genpath('.'));
end
% Datasets directory
mainDirectory = '/home/esling/deepLearn/datasets';
outDir = '/home/esling/deepLearn/results';
% Datasets used
datasets = {'50words','Adiac','ArrowHead','ARSim','Beef','BeetleFly',...
    'BirdChicken','Car','CBF','Coffee','Computers','Chlorine',...
    'CinECG','Cricket_X','Cricket_Y','Cricket_Z','DiatomSize','ECG200',...
    'DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect',...
    'ECGFiveDays','Earthquakes','FaceAll','FaceFour',...
    'FacesUCR','Fish','Gun_Point','HandOutlines', ...
    'DistalPhalanxTW','Herring','LargeKitchenAppliances',...
    'Haptics','InlineSkate','ItalyPower','Lighting2',...
    'Lighting7','MALLAT','MedicalImages','MoteStrain','NonInv_ECG1',...
    'MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect',...
    'MiddlePhalanxTW','PhalangesOutlinesCorrect','Plane',...
    'ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect',...
    'ProximalPhalanxTW','RefrigerationDevices',...
    'NonInv_ECG2','OliveOil','OSULeaf','SonyAIBO1','SonyAIBO2',...
    'ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances',...
    'SwedishLeaf','Symbols','Synthetic','Trace',...
    'Two_Patterns','TwoLeadECG','uWGestureX','uWGestureY','uWGestureZ',...
    'ToeSegmentation1','ToeSegmentation2','Wafer','WordsSynonyms','Yoga'};

% --------------
%
% Hyper-parameters
%
% --------------
% Perform individual training
individualTraining = 0;
% Factor of resampling for the time series
resampleFactor = 128;
% Use the densification of the manifold
densifyManifold = 1;
% Whiten the data in pre-processing
whitenData = 0;

% --------------
% 
% Import all datasets
%
% --------------
[fullTrainSeries, fullTrainLabels, trainLabels, trainSeries] = datasetImportTS(mainDirectory, datasets, 'TRAIN', resampleFactor, densifyManifold);
[~, ~, testLabels, testSeries] = datasetImportTS(mainDirectory, datasets, 'TEST', resampleFactor, 0);

% --------------
% 
% Pre-processing
%
% --------------
if ~individualTraining
    [fullTrainSeries, ~, Wsep, Wmix, mX] = datasetPreprocessTS(fullTrainSeries, fullTrainLabels, whitenData);
    if whitenData
        for d = 1:length(datasets)
            testSeries{d} = zca_whiten(testSeries{d}, Wsep, Wmix, mX);
        end
    end
end

% ---------------------------
%
% Final classification script
%
% ---------------------------
% Output file purposes
outputFile = [outDir '/classifRes_' typePretrain '_' typeTrain '_' layerType '_' paramsType];
outputFID = fopen([outputFile '.txt'], 'w');
% Initialize random numbers generator
rng('shuffle');
% Initialize the optimizer structure
optimize = hyperparameters_optimize(nbLayers);
% Create one example model
model = hyperparameters_init(optimize, nbLayers, resampleFactor, 10, typePretrain, typeTrain, outputFile, 'user', 1, layers, binaryLayers, userStruct);
% Optimization step
errorRates = ones(length(datasets), 1);
eFeat = ones(length(datasets), 1);
eSoft = ones(length(datasets), 1);
trainT = ones(length(datasets), 1);
testT = ones(length(datasets), 1);
if ~individualTraining
    model = deepPretrain(model, typePretrain, fullTrainSeries);
end
fprintf('Dataset \t Error \t Error(Feat) \t Error(Soft) \t Train \t Test\n');
for d = 1:length(datasets)
    fprintf('* Dataset %s.\n', datasets{d});
    if individualTraining
        [trainSeries{d}, trainLabels{d}, Wsep, Wmix, mX] = datasetPreprocessTS(trainSeries{d}, trainLabels{d}, whitenData);
        [testSeries{d}, testLabels{d}] = datasetPreprocessTS(testSeries{d}, trainSeries{d}, whitenData, Wsep, Wmix, mX);
        model = deepPretrain(model, typePretrain, trainSeries{d});
    end
    [errorRates(d), eFeat(d), eSoft(d), ~, trainT(d), testT(d)] = deepClassify(datasets{d}, typeTrain, model, trainSeries{d}, trainLabels{d}, testSeries{d}, testLabels{d}, 1);
    fprintf(outputFID, '%s \t %f \t %f \t %f \t %f \t %f\n', datasets{d}, errorRates(d), eFeat(d), eSoft(d), trainT(d), testT(d));
end
fclose(outputFID);
% Save the final results and model to Matlab file
save([outputFile '.mat'], 'errorRates', 'eFeat', 'eSoft', 'model', 'trainT', 'testT');
end