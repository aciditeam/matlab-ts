addpath(genpath('.'));

matlabpool('open');

% Datasets directory
mainDirectory = 'datasets';
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
repeat = 3;
nbSteps = 100;
nbBatch = matlabpool('size');
nbNetworks = nbSteps * nbBatch;
% Initialize the optimizer structure
optimize = hyperparameters_optimize(nbLayers);
optimize.structure.units.past = zeros(nbNetworks, nbLayers);
optimize.structure.units.errors = zeros(nbNetworks, length(datasets));
curNetwork = 1;
nextBatch = {};
% Create one example model
modelEx = hyperparameters_random(optimize, nbLayers, size(fullTrainSeries, 2), 10);
% Compute the structure grid beforehand
finalGrid = hyperparameters_grid(modelEx, optimize, 'structure', 0);
% Optimization step
for steps = 1:nbSteps
    % Local pasts values
    localPasts = zeros(nbBatch, size(optimize.structure.units.past, 2));
    localErrors = zeros(nbBatch, size(optimize.structure.units.errors, 2));
    % Architecture batch
    parfor batch = 1:nbBatch
        if (isempty(nextBatch) || ((batch-1) > (steps / nbBatch)))
            model = hyperparameters_random(optimize, nbLayers, size(fullTrainSeries, 2), 10);
        else
            model = nextBatch{batch};
        end
        % Keep track of current architecture
        localPasts(batch, :) = model.structure;
        errorRates = zeros(length(datasets), repeat);
        % Perform 10 random repetitions
        for r = 1:repeat
            localModel = model;
            % Generate random network weights
            for l = 1:nbLayers-2
                localModel.train.biases{l+1} = rand(size(model.train.biases{l+1})) - 0.5;
                localModel.train.W{l} = rand(size(model.train.W{l})) - 0.5;
            end
            % Softmax classifier on top
            for d = 1:length(datasets)
                errorRates(d, r) = deepClassifyRND(datasets{d}, localModel.train, trainSeries{d}, trainLabels{d}, testSeries{d}, testLabels{d});
            end
        end
        % Mean error of current architecture
        localErrors(batch, :) = mean(errorRates, 2);
    end
    % Retrieve parallel error rates and architectures
    optimize.structure.units.errors((curNetwork):(curNetwork+nbBatch-1), :) = localErrors;
    optimize.structure.units.past((curNetwork):(curNetwork+nbBatch-1), :) = localPasts;
	curNetwork = curNetwork + nbBatch;
    % Extract current error rates and architectures
    curError = optimize.structure.units.errors(1:(curNetwork-1), :);
    curValue = optimize.structure.units.past(1:(curNetwork-1), :);
    % Rank different architecture against each other
    ranks = hyperparameters_criticaldifference(curError');
    % Find the next values of parameters to evaluate
    nextBatch = hyperparameters_fit(optimize, modelEx, curValue, ranks, finalGrid, nbLayers, nbBatch);
    % Save the current state of optimization (only errors and structures)
    save(['optimizedStructure_' num2str(nbLayers) '_' fileIDchar '_layers.mat'], 'curError', 'curValue');
end

matlabpool('close');