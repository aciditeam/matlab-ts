addpath(genpath('.'));

% Datasets directory
mainDirectory = '/home/esling/deepLearn/datasets';
outDir = '/home/esling/deepLearn';
% Datasets used
datasets = {'50words','Adiac','ArrowHead','ARSim','Beef','BeetleFly',...
    'BirdChicken','Car','CBF','Coffee','Computers','Chlorine',...
    'CinECG','Cricket_X','Cricket_Y','Cricket_Z','DiatomSize','ECG200',...
    'DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect',...
    'ECGFiveDays','Earthquakes','ElectricDevices','FaceAll','FaceFour',...
    'FacesUCR','Fish','FordA','FordB','Gun_Point','HandOutlines', ...
    'DistalPhalanxTW','Herring','LargeKitchenAppliances',...
    'Haptics','InlineSkate','ItalyPower','Lighting2',...
    'Lighting7','MALLAT','MedicalImages','MoteStrain','NonInv_ECG1',...
    'MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect',...
    'MiddlePhalanxTW','PhalangesOutlinesCorrect','Plane',...
    'ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect',...
    'ProximalPhalanxTW','RefrigerationDevices',...
    'NonInv_ECG2','OliveOil','OSULeaf','SonyAIBO1','SonyAIBO2',...
    'ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances',...
    'StarLight','SwedishLeaf','Symbols','Synthetic','Trace',...
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
resampleFactor = 256;
% Use the densification of the manifold
densifyManifold = 1;
% Whiten the data in pre-processing
whitenData = 1;
% Use the initialization of the weights as randomly-centered gaussians
gaussianInit = 0;
gaussianVariance = 0.5;
% Should we perform a random search
architectureRandomSearch = 0;

% --------------
% 
% Import all datasets
%
% --------------
[fullTrainSeries, fullTrainLabels, trainLabels, trainSeries] = datasetImportTS(mainDirectory, datasets, 'TRAIN', resampleFactor, densifyManifold);
[fullTestSeries, fullTestLabels, testLabels, testSeries] = datasetImportTS(mainDirectory, datasets, 'TEST', resampleFactor, 0);

% --------------
% 
% Pre-processing
%
% --------------
if ~individualTraining
    [fullTrainSeries, fullTrainLabels, Wsep, Wmix, mX] = datasetPreprocessTS(fullTrainSeries, fullTrainLabels, whitenData);
    [fullTestSeries, fullTestLabels] = datasetPreprocessTS(fullTestSeries, fullTestLabels, whitenData, Wsep, Wmix, mX);
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
    [errorRates(d), eFeat, eSoft, model, trainT, testT] = deepClassify(datasets{d}, typeTrain, model, trainSeries{d}, trainLabels{d}, testSeries{d}, testLabels{d}, 0);
    fprintf(outputFID, '%s \t %f \t %f \t %f \t %f \t %f\n', datasets{d}, errorRates(d), eFeat, eSoft, model, trainT, testT);
end
fclose(outputFID);

save([outputFile '.mat'], 'errorRates', 'eFeat', 'eSoft', 'model', 'trainT', 'testT');
