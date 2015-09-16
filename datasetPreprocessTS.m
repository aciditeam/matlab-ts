function [fullTrainSeries, fullTrainLabels, Wsep, Wmix, mX] = datasetPreprocessTS(fullTrainSeries, fullTrainLabels, whiten, Wsep, Wmix, mX)
fprintf('Pre-processing all data.\n')
% Shuffle the training data
perm_idx = randperm(size(fullTrainSeries,1));
fullTrainSeries = fullTrainSeries(perm_idx, :);
fullTrainLabels = fullTrainLabels(perm_idx);
fprintf('Shuffled training data.\n');
if whiten
    fprintf('Whitening training data.\n');
    % ZCA
    if (nargin < 4)
        [~, Wsep, Wmix, mX] = zca(fullTrainSeries, 0.1);
    end
    fullTrainSeries = zca_whiten(fullTrainSeries, Wsep, Wmix, mX);
else
    Wsep = [];
    Wmix = [];
    mX = [];
end
end