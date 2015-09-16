% ---------------------------
%
% Debugged parameters 
%
% ---------------------------
structures = {[128 250 500 10], [128 1500 1000 10], [128 1500 1500 10], ...
    [128 500 1000 1500 10],[128 1500 1000 500 10],[128 1000 1000 1000 10], ...
    [128 500 1000 1500 2000 10],[128 500 1500 1000 2000 10],[128 2000 1500 1000 500 10],[128 1000 1000 1000 1000 10], ...
    [128 250 500 1000 2000 250 10],[128 500 1500 1000 2000 250 10]};
nbLayers = [4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7];
layerTypes = {'4_inc', '4_dec', '4_lin', '5_inc', '5_dec', '5_lin', '6_inc', '6_bot', '6_dec', '6_lin', '7_inc', '7_bot'};
typePretrains = {'DAE', 'RBM'};
typeVariants = {{'DAE', 'RICA', 'CAE'}, {'CD1', 'PERS', 'PARAL'}};
variantsParamsPretrain = {'''iteration.n_epochs'', ''learning.lrate'', ''use_tanh'', ''noise.drop'', ''noise.level'', ''rica.cost'', ''cae.cost''',...
    '''iteration.n_epochs'', ''learning.lrate'', ''learning.cd_k'', ''learning.persistent_cd'', ''parallel_tempering.use'''};
curPreVals = {{{'200 1e-3 0 0.1 0.1 0 0','200 1e-3 1 0.1 0.1 0 0', '200 1e-3 2 0.1 0.1 0 0'},... %DAE_DAE
    {'200 1e-3 0 0 0 0.1 0','200 1e-3 1 0 0 0.1 0', '200 1e-3 2 0 0 0.1 0'},... %DAE_RICA
    {'200 1e-3 0 0 0 0.01 0','200 1e-3 1 0 0 0.01 0', '200 1e-3 2 0 0 0.01 0'}},... %DAE_CAE
    {{'200 1e-3 1 0 0'},... %RBM_CD1
    {'200 1e-3 1 1 0'},... %RBM_PERS
    {'200 1e-3 1 0 1'}}}; %RBM_PARAL
typeFunctions = {{'sig', 'tanh', 'relu'}, {'base'}};
typeTrains = {{'SDAE','MLP'}, {'MLP','DBM','DBN'}};
variantsParamsTrain = {{'''iteration.n_epochs'', ''use_tanh''','''iteration.n_epochs'', ''use_tanh'''}...
    {'''iteration.n_epochs''', '''iteration.n_epochs'', ''learning.persistent_cd''', '''iteration.n_epochs'', ''learning.persistent_cd'''}};
curTrainVals = {{{{'200 0','200 0'},{'200 1','200 1'},{'200 2','200 2'}},... %DAE_DAE
    {{'200 0','200 0'},{'200 1','200 1'},{'200 2','200 2'}},... %DAE_RICA
    {{'200 0','200 0'},{'200 1','200 1'},{'200 2','200 2'}}},... %DAE_CAE
    {{{'200 0', '200 1', '200 0'},{'200 0', '200 1', '200 0'},{'200 0', '200 1', '200 0'}},... %RBM_CD1
    {{'200 0', '200 1', '200 0'},{'200 0', '200 1', '200 0'},{'200 0', '200 1', '200 0'}},... %RBM_PERS
    {{'200 0', '200 1', '200 0'},{'200 0', '200 1', '200 0'},{'200 0', '200 1', '200 0'}}}};
binaryLayers = [0 1];

%% --------------
%
% Base parameters
%
% --------------

addpath(genpath('.'));
% Datasets directory
mainDirectory = '/Volumes/HDD/Users/esling/Dropbox/TS_Datasets';
outDir = '/Volumes/HDD/Users/esling/Research/Coding/deepLearning/results';
% Datasets used
datasets = {'50words','Adiac','Coffee','Lighting7','MiddlePhalanxTW',...
    'NonInv_ECG2','OliveOil','uWGestureX','WordsSynonyms','Yoga'};

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
[fullTrainSeries, fullTrainLabels, trainSeries, trainLabels] = datasetImportTS(mainDirectory, datasets, 'TRAIN', resampleFactor, densifyManifold);
[~, ~, testSeries, testLabels] = datasetImportTS(mainDirectory, datasets, 'TEST', resampleFactor, 0);

% --------------
% 
% Pre-processing
%
% --------------
if ~individualTraining
    [fullTrainSeries, ~, Wsep, Wmix, mX] = datasetPreprocessTS(fullTrainSeries, fullTrainLabels, whitenData);
    if whitenData
    fprintf('Whitening testing data.\n');
        for d = 1:length(datasets)
            testSeries{d} = zca_whiten(testSeries{d}, Wsep, Wmix, mX);
        end
    end
end

%% ---------------------------
%
% Final classification script
%
% ---------------------------
for st = 1:length(structures)
    curNbLayers = nbLayers(st);
    for binary = 0:1
        if binary
            layerType = [layerTypes{st} '_bin'];
            bLayer = [0 ones(1, curNbLayers-1)];
        else
            layerType = [layerTypes{st} '_real'];
            bLayer = [0 zeros(1, curNbLayers-1)];
        end
        for p = 1:length(typePretrains)
            typePretrain = typePretrains{p};
            curTypeTrain = typeTrains{p};
            curVariants = typeVariants{p};
            curFunction = typeFunctions{p};
            for v = 1:length(curVariants)
                for f = 1:length(curFunction)
                    paramsType = [curVariants{v} '_' curFunction{f}];
                    for t = 1:length(curTypeTrain)
                        curPreNames = variantsParamsPretrain{p};
                        curTrainNamesF = variantsParamsTrain{p};
                        curPreVal = curPreVals{p}{v}{f};
                        typeTrain = curTypeTrain{t};
                        curTrainNames = curTrainNamesF{t};
                        curTrainVal = curTrainVals{p}{v}{f}{t};
                        % Transform inputs needed
                        nbLayers = curNbLayers;
                        layers = structures{st};
                        binaryLayers = bLayer;
                        userStruct = struct;
                        userStruct.pretrain = struct;
                        eval(['userStruct.pretrain.names = {' curPreNames '};']);
                        eval(['curPreNames = {' curPreNames '};']);
                        userStruct.pretrain.values = str2num(curPreVal);
                        userStruct.pretrain.values(1) = 20;
                        curPreVal = str2num(curPreVal);
                        eval(['userStruct.train.names = {' curTrainNames '};']);
                        eval(['curTrainNames = {' curTrainNames '};']);
                        userStruct.train.values = str2num(curTrainVal);
                        userStruct.train.values(1) = 20;
                        curTrainVal = str2num(curTrainVal);
                        fprintf('Evaluating model configuration:\n')
                        fprintf('* %32s\t: %s.\n', 'Layers', layerType);
                        fprintf('* %32s\t: %s.\n', 'Pretrain', typePretrain);
                        fprintf('* %32s\t: %s.\n', 'Params', paramsType);
                        for i = 1:length(curPreNames)
                            fprintf(' - %32s\t: %f.\n', curPreNames{i}, curPreVal(i));
                        end
                        fprintf('* %32s\t: %s.\n', 'Train', typeTrain);
                        for i = 1:length(curTrainNames)
                            fprintf(' - %32s\t: %f.\n', curTrainNames{i}, curTrainVal(i));
                        end
                        %% ---------------------------
                        %
                        % Actual learning part
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
                        model = hyperparameters_init(optimize, nbLayers, resampleFactor, 10, typePretrain, typeTrain, '/tmp', 'user', 1, layers, binaryLayers, userStruct);
                        % Optimization step
                        errorRates = ones(length(datasets), 1);
                        eFeat = ones(length(datasets), 1);
                        eSoft = ones(length(datasets), 1);
                        trainT = ones(length(datasets), 1);
                        testT = ones(length(datasets), 1);
                        if ~individualTraining
                            %figure; deepTSmotifs(model.pretrain(1).W, fullTrainSeries, 1);
                            %title([typePretrain '_' paramsType ' - Before pre-train']);
                            model = deepPretrain(model, typePretrain, fullTrainSeries);
                            %figure; deepTSmotifs(model.pretrain(1).W, fullTrainSeries, 1);
                            %title([typePretrain '_' paramsType ' - After pre-train']);
                        end
                        fprintf('Dataset \t Error \t Error(Feat) \t Error(Soft) \t Train \t Test\n');
                        for d = 1:length(datasets)
                            fprintf('* Dataset %s.\n', datasets{d});
                            if individualTraining
                                [trainSeries{d}, trainLabels{d}, Wsep, Wmix, mX] = datasetPreprocessTS(trainSeries{d}, trainLabels{d}, whitenData);
                                [testSeries{d}, testLabels{d}] = datasetPreprocessTS(testSeries{d}, trainSeries{d}, whitenData, Wsep, Wmix, mX);
                                model = deepPretrain(model, typePretrain, trainSeries{d});
                            end
                            [errorRates(d), eFeat(d), eSoft(d), modelFT, trainT(d), testT(d)] = deepClassify(datasets{d}, typeTrain, model, trainSeries{d}, trainLabels{d}, testSeries{d}, testLabels{d}, 1);
                            %figure; deepTSmotifs(modelFT.train.W{1}, fullTrainSeries, 1);
                            %title([typePretrain '_' typeTrain '_' paramsType ' - After fine-tune']);
                            disp('Model error rate :');
                            disp(errorRates(d));
                            disp('Features error rate :');
                            disp(eFeat(d));
                            fprintf(outputFID, '%s \t %f \t %f \t %f \t %f \t %f\n', datasets{d}, errorRates(d), eFeat(d), eSoft(d), trainT(d), testT(d));
                            deepVisualCheckup('SDAE', model, trainSeries, trainLabels, testSeries, testLabels, 1)
                        end
                        fclose(outputFID);
                        % Save the final results and model to Matlab file
                        save([outputFile '.mat'], 'errorRates', 'eFeat', 'eSoft', 'model', 'trainT', 'testT');
                    end
                end
            end
        end
    end
end
%%
disp(unique(trainLabels{2}));
n = hist(trainLabels{2}, unique(trainLabels{2}));
disp(n);
4
     9
    12
    17
    27
    36