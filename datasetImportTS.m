function [fullTrainSeries, fullTrainLabels, trainSeries, trainLabels] = datasetImportTS(mainDirectory, datasets, setType, resampleFactor, densifyManifold)
% Importing datasets
fprintf(['Importing ' setType ' data\n']);
nbSeries = zeros(length(datasets), 1);
trainLabels = cell(length(datasets), 1);
trainSeries = cell(length(datasets), 1);
% First import all training datas
for d = 1:length(datasets)
    fprintf(' * Dataset %32s\t', datasets{d});
    curDatasets = [mainDirectory '/' datasets{d} '/' datasets{d} '_' setType];
    curTrain = importdata(curDatasets);
    tmpLabels = curTrain(:, 1) - 0.5;
    uniqueLabels = sort(unique(tmpLabels), 'ascend');
    for i = 1:length(uniqueLabels)
        tmpLabels(tmpLabels == uniqueLabels(i)) = i;
    end
    trainLabels{d} = tmpLabels;
    curSeries = curTrain(:, 2:end);
    sampledSeries = zeros(size(curTrain, 1), resampleFactor);
    % For a starters we perform resampling of the series
    for ts = 1:size(curTrain, 1)
        sampledSeries(ts, :) = resample(curSeries(ts, :), resampleFactor, size(curSeries, 2));
        sampledSeries(ts, :) = (sampledSeries(ts, :) - mean(sampledSeries(ts, :))) ./ max(sampledSeries(ts, :));
    end
    fprintf('%d series.\n', size(curTrain, 1));
    trainSeries{d} = sampledSeries;
    nbSeries(d) = size(curTrain, 1);
end
% Densify the manifold of series by adding data
if densifyManifold
    fprintf('Densifying manifold (factor %f).\n', densifyManifold);
    maxSeries = max(nbSeries) * densifyManifold;
    for d = 1:length(datasets)
        curNbSeries = nbSeries(d);
        seriesMissing = maxSeries - curNbSeries;
        curTrainSeries = trainSeries{d};
        curTrainLabels = trainLabels{d};
        newSeries = zeros(seriesMissing, resampleFactor);
        % Perform class-sensitive sampling
        nbLabels = length(unique(curTrainLabels));
        idealClassNumber = maxSeries / nbLabels;
        classDistribution = hist(curTrainLabels, unique(curTrainLabels));
        classDisparity = -(classDistribution - repmat(idealClassNumber, 1, nbLabels));
        classDisparity = classDisparity .* (classDisparity > 0);
        classDisparity = classDisparity ./ sum(classDisparity);
        classSeriesIDs = [];
        for i = 1:nbLabels
            nbSeriesCls = ceil(classDisparity(i) * seriesMissing);
            classWiseIDs = find(curTrainLabels == i);
            classSeriesIDs = [classSeriesIDs randsample(classWiseIDs, nbSeriesCls, 1)'];
        end
        newSeriesID = randsample(classSeriesIDs, seriesMissing, 1);
        newLabels = curTrainLabels(newSeriesID);
        nbNoise = 0; nbOutlier = 0; nbWarp = 0;
        for s = 1:seriesMissing
            tmpSeries = curTrainSeries(newSeriesID(s), :);
            switch floor(rand * 10)
                case 0
                    % Add white gaussian noise
                    tmpSeries = tmpSeries + random('norm', 0, 0.1, 1, resampleFactor);
                    nbNoise = nbNoise + 1;
                case 1
                    % Add random outliers
                    tmpIDs = randsample(resampleFactor, floor(resampleFactor / 100), 0);
                    tmpSeries(tmpIDs) = -tmpSeries(tmpIDs);
                    nbOutlier = nbOutlier + 1;
                otherwise
                    % Add temporal warping
                    switchPosition = ceil(rand * (resampleFactor / 4)) + (resampleFactor / 2);
                    leftSide = tmpSeries(1:switchPosition);
                    rightSide = tmpSeries((switchPosition+1):end);
                    direction = sign(rand - 0.5);
                    leftSide = resample(leftSide, switchPosition - (direction * 10), switchPosition);
                    rightSide = resample(rightSide, resampleFactor - switchPosition + (direction * 10), resampleFactor - switchPosition);
                    tmpSeries = [leftSide rightSide];
                    nbWarp = nbWarp + 1;
            end
            newSeries(s, :) = tmpSeries;
        end
        nbSeries(d) = maxSeries;
        trainSeries{d} = [curTrainSeries ; newSeries];
        trainLabels{d} = [curTrainLabels ; newLabels];
        fprintf(' * Densify %32s \t [%d noise - %d outlier - %d warp].\n', datasets{d}, nbNoise, nbOutlier, nbWarp);
    end
end
idEnd = cumsum(nbSeries);
idStart = idEnd + 1;
idStart = [1 ; idStart(1:end-1)];
fullTrainLabels = zeros(sum(nbSeries), 1);
fullTrainSeries = zeros(sum(nbSeries), resampleFactor);
fprintf('Stacking dataset to full matrix.\n');
for d = 1:length(datasets)
    fprintf(' * Stackin %32s \t [%d - %d].\n', datasets{d}, idStart(d), idEnd(d));
    fullTrainLabels(idStart(d):idEnd(d)) = trainLabels{d};
    fullTrainSeries(idStart(d):idEnd(d), :) = trainSeries{d};
end
fprintf('Import ended, matrix %d x %d.\n', size(fullTrainSeries, 1), size(fullTrainSeries, 2));
clear labelsSeries;
clear resampledSeries;
clear curTrain;
end