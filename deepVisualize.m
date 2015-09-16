addpath(genpath('.'));

% Resample factor
resampleFactor = 128;
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
% Import all datasets (without densification)
%
% --------------
[fullTrainSeries, fullTrainLabels, trainLabels, trainSeries] = datasetImportTS(mainDirectory, datasets, 'TRAIN', resampleFactor, 0);
[fullTestSeries, fullTestLabels, testLabels, testSeries] = datasetImportTS(mainDirectory, datasets, 'TEST', resampleFactor, 0);

groupTrain = [];
for i = 1:length(datasets)
    groupTrain = [groupTrain repmat(i, 1, length(trainLabels{i}))];
end
groupTest = [];
for i = 1:length(datasets)
    groupTest = [groupTest repmat(i + length(datasets), 1, length(testLabels{i}))];
end

%% --------------
%
% Perform dataset-wise full t-SNE
%
% --------------
% Parameters for dimensionality reduction
numDims = 2; pcaDims = 64; perplexity = 50;
for d = 1:length(datasets)
    % Perform fast version on non-densified training
    map = tsne(trainSeries{d}, [], numDims, pcaDims, perplexity);
    % Plot non-densified training result
    h = figure;
    gscatter(map(:,1), map(:,2), trainLabels{d});
    print(h, ['export/' datasets{d} '_train_base.fig']);
    print(h, '-djpeg', ['export/' datasets{d} '_train_base.jpg']);
    close(h);
    % Perform fast version on non-densified testing
    map = tsne(testSeries{d}, [], numDims, pcaDims, perplexity);
    % Plot non-densified training result
    h = figure;
    gscatter(map(:,1), map(:,2), testLabels{d});
    print(h, ['export/' datasets{d} '_test_base.fig']);
    print(h, '-djpeg', ['export/' datasets{d} '_test_base.jpg']);
    close(h);
    % Perform fast version on non-densified complete set
    map = tsne([trainSeries{d} ; testSeries{d}], [], numDims, pcaDims, perplexity);
    % Plot non-densified training result
    h = figure;
    gscatter(map(:,1), map(:,2), [trainLabels{d} ; (testLabels{d} + max(trainLabels{d}))]);
    print(h, ['export/' datasets{d} '_both_base.fig']);
    print(h, '-djpeg', ['export/' datasets{d} '_both_base.jpg']);
    close(h);
end

% --------------
%
% Perform Barnes-Hut t-SNE
%
% --------------
% Parameters for dimensionality reduction
numDims = 2; pcaDims = 50; perplexity = 50; theta = .5;
% Perform fast version on non-densified training
map = fast_tsne(fullTrainSeries, numDims, pcaDims, perplexity, theta);
% Plot non-densified training result
h = figure;
gscatter(map(:,1), map(:,2), groupTrain);
print(h, 'export/All_data_train_base.fig');
print(h, '-djpeg', 'export/All_data_train_base.jpg');
close(h);
% Perform fast version on non-densified testing
map = fast_tsne(fullTestSeries, numDims, pcaDims, perplexity, theta);
% Plot non-densified training result
h = figure;
gscatter(map(:,1), map(:,2), groupTest);
print(h, 'export/All_data_test_base.fig');
print(h, '-djpeg', 'export/All_data_test_base.jpg');
close(h);
% Perform fast version on non-densified complete set
map = fast_tsne([fullTrainSeries ; fullTestSeries], numDims, pcaDims, perplexity, theta);
% Plot non-densified training result
h = figure;
gscatter(map(:,1), map(:,2), [groupTrain groupTest]);
print(h, 'export/All_data_both_base.fig');
print(h, '-djpeg', 'export/All_data_both_base.jpg');
close(h);

%% --------------
% 
% Import all densified datasets
%
% --------------
[fullTrainSeries, fullTrainLabels, trainLabels, trainSeries] = datasetImportTS(mainDirectory, datasets, 'TRAIN', resampleFactor, 1);
[fullTestSeries, fullTestLabels, testLabels, testSeries] = datasetImportTS(mainDirectory, datasets, 'TEST', resampleFactor, 1);

groupTrain = [];
for i = 1:length(datasets)
    groupTrain = [groupTrain repmat(i, 1, length(trainLabels{i}))];
end
groupTest = [];
for i = 1:length(datasets)
    groupTest = [groupTest repmat(i + length(datasets), 1, length(testLabels{i}))];
end

%% --------------
%
% Perform dataset-wise full t-SNE
%
% --------------
% Parameters for dimensionality reduction
numDims = 2; pcaDims = 64; perplexity = 50;
for d = 1:length(datasets)
    % Perform fast version on non-densified training
    map = fast_tsne(trainSeries{d}, numDims, pcaDims, perplexity, theta);
    % Plot non-densified training result
    h = figure;
    gscatter(map(:,1), map(:,2), trainLabels{d});
    print(h, ['export/' datasets{d} '_train_dense.fig']);
    print(h, '-djpeg', ['export/' datasets{d} '_train_dense.jpg']);
    close(h);
    % Perform fast version on non-densified testing
    map = fast_tsne(testSeries{d}, numDims, pcaDims, perplexity, theta);
    % Plot non-densified training result
    h = figure;
    gscatter(map(:,1), map(:,2), testLabels{d});
    print(h, ['export/' datasets{d} '_test_dense.fig']);
    print(h, '-djpeg', ['export/' datasets{d} '_test_dense.jpg']);
    close(h);
    % Perform fast version on non-densified complete set
    map = fast_tsne([trainSeries{d} ; testSeries{d}], numDims, pcaDims, perplexity, theta);
    % Plot non-densified training result
    h = figure;
    gscatter(map(:,1), map(:,2), [trainLabels{d} ; (testLabels{d} + max(trainLabels{d}))]);
    print(h, ['export/' datasets{d} '_both_dense.fig']);
    print(h, '-djpeg', ['export/' datasets{d} '_both_dense.jpg']);
    close(h);
end

% --------------
%
% Perform Barnes-Hut t-SNE
%
% --------------
% Parameters for dimensionality reduction
numDims = 2; pcaDims = 50; perplexity = 50; theta = .5;
% Perform fast version on non-densified training
map = fast_tsne(fullTrainSeries, numDims, pcaDims, perplexity, theta);
% Plot non-densified training result
h = figure;
gscatter(map(:,1), map(:,2), groupTrain);
print(h, 'export/All_data_train_dense.fig');
print(h, '-djpeg', 'export/All_data_train_dense.jpg');
close(h);
% Perform fast version on non-densified testing
map = fast_tsne(fullTestSeries, numDims, pcaDims, perplexity, theta);
% Plot non-densified training result
h = figure;
gscatter(map(:,1), map(:,2), groupTest);
print(h, 'export/All_data_test_dense.fig');
print(h, '-djpeg', 'export/All_data_test_dense.jpg');
close(h);
% Perform fast version on non-densified complete set
map = fast_tsne([fullTrainSeries ; fullTestSeries], numDims, pcaDims, perplexity, theta);
% Plot non-densified training result
h = figure;
gscatter(map(:,1), map(:,2), [groupTrain groupTest]);
print(h, 'export/All_data_both_dense.fig');
print(h, '-djpeg', 'export/All_data_both_dense.jpg');
close(h);
