function [errorRate, trainTime] = deepClassifyRND(dataset, model, trainSeries, trainLabels, testSeries, testLabels, params)
nbLayers = length(model.structure);
fprintf('(%d-layers) %32s \t', nbLayers, dataset);
fprintf('Random: %.2f\t', 100/length(unique(trainLabels)));
% Get random architecture feature
sdae_feature = sdae_get_hidden(trainSeries, model);
% Plug a softmax layer on top of it
tic; softModel = train(trainLabels, sparse(double(sdae_feature)), '-s 0');
trainTime = toc;
sdae_feature = sdae_get_hidden(testSeries, model);
[L, accuracy, probs] = predict(testLabels, sparse(double(sdae_feature)), softModel, '-b 1');
errorRate = (100 - accuracy) / 100;
end