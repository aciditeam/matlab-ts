function deepOptimizePretrain(nbLayers, typePretrain, typeTrain, fileIDchar)

fID = fopen(['/home/esling/deepLearn/results/log_' typePretrain '_' num2str(nbLayers) '_' fileIDchar '.txt'], 'w');
fprintf(fID, '%s\n', typePretrain);
fprintf(fID, '%s\n', typeTrain);
fprintf(fID, '%d\n', nbLayers);

addpath(genpath('/home/esling/deepLearn/'));
% Datasets directory
mainDirectory = '/home/esling/deepLearn/datasets';
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

saveFile = ['/home/esling/deepLearn/results/' typePretrain '_' typeTrain '_' fileIDchar];

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
[fullTestSeries, fullTestLabels] = datasetPreprocessTS(fullTestSeries, fullTestLabels, whitenData);

% --------------
%
% Random architecture search
%
% --------------
if architectureRandomSearch
    deepStructure
else
    layers = {[],[], ...
        [resampleFactor 500 10], ...
        [resampleFactor 500  250 10], ...
        [resampleFactor 500  1000 2000 10], ...
        [resampleFactor 500  1500 1000 2000 10], ...
        [resampleFactor 1000 3000 2000 4000 500 10], ...
    };
end

% --------------
%
% Random parameters search
%
% --------------
repeat = 1;
nbSteps = 100;
nbBatch = 10;
nbNetworks = nbSteps * nbBatch;
%matlabpool('open');
% Initialize the optimizer structure
optimize = hyperparameters_optimize(nbLayers);
% Initialize the random number generator
rng('shuffle');
% Prepare the past structures
optimize = hyperparameters_past(optimize, typeTrain, typePretrain, length(datasets), nbNetworks);
curNetwork = 0;
nextBatch = {};
% Create one example model
model = hyperparameters_init(optimize, nbLayers, resampleFactor, 10, typePretrain, typeTrain, saveFile, 'random', 1, layers{nbLayers});
% Optimization step
for steps = 1:nbSteps
	fprintf(fID, 'Starting step n.%d\n', steps);
    % Errors of the batch
    errorRates = ones(length(datasets), nbBatch);
    % Models of the batch
    modelBatch = cell(nbBatch, 1);
    % Architecture batch
    for batch = 1:nbBatch
%         t = getCurrentTask();
%         myLogFile = sprintf('/home/esling/deepLearn/results/log.%c.%d.txt', fileIDchar, t.ID);
%         tID = fopen(myLogFile, 'a+');
        fprintf(fID, 'Starting batch n.%d', batch);
%         fclose(tID);
        if (isempty(nextBatch) || ((batch-1) > (steps / nbBatch)))
            curModel = hyperparameters_init(optimize, nbLayers, resampleFactor, 10, typePretrain, typeTrain, '/tmp', 'random', 1, layers{nbLayers});
        else
            curModel = nextBatch{batch};
        end
%         tID = fopen(myLogFile, 'a+');
        hyperparameters_output(fID, optimize, curModel, typePretrain, typeTrain);
%         fclose(tID);
        try
            %g = gpuDevice();
            %reset(g);
            curErrors = ones(length(datasets), 1);
            %for r = 1:repeat
                if ~individualTraining
                    curModel = deepPretrain(curModel, typePretrain, fullTrainSeries);
                end
                fprintf(fID, 'Done pretraining.');
                parfor d = 1:length(datasets)
                    %fprintf('* Dataset %s.\n', datasets{d});
                    curErrors(d) = deepClassifyLayerSoftmax(typePretrain, curModel.pretrain, trainSeries{d}, trainLabels{d}, testSeries{d}, testLabels{d});
                end
            %end
            errorRates(:, batch) = curErrors;
%             tID = fopen(myLogFile, 'a+');
            fprintf(fID, 'Successful network configuration.\n');
%             fclose(tID);
        catch meExc
            strMess = getReport(meExc);
%             tID = fopen(myLogFile, 'a+');
            fprintf(fID, 'Erroneous network configurations : %s\n', strMess);
%             fclose(tID);
        end
        modelBatch{batch} = curModel;
    end
    for b = 1:nbBatch
        fprintf(fID, 'Gathering batch.%d\n', batch);
        optimize = hyperparameters_gather(optimize, modelBatch{b}, curNetwork + b, errorRates(:, b));
    end
    clear curModel;
    clear modelBatch;
    fprintf(fID, 'Step n.%d finished.\n', steps);
    curNetwork = curNetwork + nbBatch;
    % Extract current error rates and architectures
    curError = optimize.pretrain.errors(1:(curNetwork-1), :);
    curValue = optimize.pretrain.past(1:(curNetwork-1), :);
    % Rank different architecture against each other
    ranks = hyperparameters_criticaldifference(curError');
    % Generate a grid that will in fact be a very wide set of random models
    finalGrid = hyperparameters_grid(model, optimize, 'pretrain', 1e6);
    % Find the next values of parameters to evaluate
    nextBatch = hyperparameters_fit(optimize, model, curValue, ranks, finalGrid, nbLayers, 10, 'pretrain');
    % Save the current state of optimization (only errors and structures)
    save(['/home/esling/deepLearn/results/optimizedParameters_' typePretrain '_' num2str(nbLayers) '_' fileIDchar '.mat'], 'optimize');
end
fclose(fID);
