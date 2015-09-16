addpath(genpath('.'));

% Datasets directory
mainDirectory = '/Users/esling/Dropbox/TS_Datasets';
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
resampleFactor = 128;
% Use the densification of the manifold
densifyManifold = 0;
% Whiten the data in pre-processing
whitenData = 0;
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
[fullTrainSeries, fullTrainLabels] = datasetPreprocessTS(fullTrainSeries, fullTrainLabels, whitenData);

% --------------
%
% Random architecture search
%
% --------------
if architectureRandomSearch
    deepStructure
else
    layers = {[],[], ...
        [resampleFactor 4096 10], ...
        [resampleFactor 4000 3000 10], ...
        [resampleFactor 1000 2000 4000 10], ...
        [resampleFactor 1000 3000 2000 4000 10], ...
        [resampleFactor 1000 3000 2000 4000 500 10], ...
    };
end

% --------------
%
% Random parameters search
%
% --------------
repeat = 2;
nbSteps = 100;
nbBatch = 10;
nbNetworks = nbSteps * nbBatch;
% Initialize the optimizer structure
optimize = hyperparameters_optimize(nbLayers);
% Prepare the past structures
optimize = hyperparameters_past(optimize, typeTrain, typePretrain, length(datasets), nbNetworks);
curNetwork = 1;
nextBatch = {};
% Create one example model
model = hyperparameters_init(optimize, nbLayers, resampleFactor, 10, typePretrain, typeTrain, '/tmp', 'random', 1, layers{nbLayers});
% Optimization step
for steps = 1:nbSteps
    % Architecture batch
    for batch = 1:nbBatch
        if (isempty(nextBatch) || ((batch-1) > (steps / nbBatch)))
            model = hyperparameters_init(optimize, nbLayers, resampleFactor, 10, typePretrain, typeTrain, '/tmp', 'random', 1, layers{nbLayers});
        else
            model = nextBatch{batch};
        end
        errorRates = ones(length(datasets), repeat);
        try
            for r = 1:repeat
                if ~individualTraining
                    model = deepPretrain(model, typePretrain, fullTrainSeries);
                end
                for d = 1:length(datasets)
                    fprintf('* Dataset %s.\n', datasets{d});
                    if individualTraining
                        [trainSeries{d}, trainLabels{d}] = datasetPreprocessTS(trainSeries{d}, trainLabels{d}, whitenData);
                        model = deepPretrain(model, typePretrain, fullTrainSeries);
                    end
                    [errorRates(d, r), eFeat, eSoft, model, trainT, testT] = deepClassify(datasets{d}, typeTrain, model, trainSeries{d}, trainLabels{d}, testSeries{d}, testLabels{d}, 0);
                end
            end
        catch
            disp('Erroneous network configurations.');
        end
        optimize = hyperparameters_gather(optimize, model, curNetwork, mean(errorRates, 2));
        curNetwork = curNetwork + 1;
    end
    % Extract current error rates and architectures
    curError = optimize.pretrain.errors(1:(curNetwork-1), :);
    curValue = [optimize.pretrain.past(1:(curNetwork-1), :) optimize.train.past(1:(curNetwork-1), :)];
    % Rank different architecture against each other
    ranks = hyperparameters_criticaldifference(curError');
    % Generate a grid that will in fact be a very wide set of random models
    finalGrid = hyperparameters_grid(model, optimize, 'full', 1e6);
    % Find the next values of parameters to evaluate
    nextBatch = hyperparameters_fit(optimize, model, curValue, ranks, finalGrid, nbLayers, 10, 'full');
    % Save the current state of optimization (only errors and structures)
    save(['optimizedParameters_' typeTrain '_' typePretrain '_' num2str(nbLayers) '_' fileIDchar '.mat'], 'optimize');
end
