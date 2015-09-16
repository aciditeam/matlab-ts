function [optimize] = hyperparameters_gather(optimize, model, curNetwork, errors)
% Generate past and errors structure for pre-training type
preTNames = optimize.pretrainNames;
curParams = zeros(1, length(optimize.pretrainNames));
for i = 1:length(preTNames)
    curName = preTNames{i};
    eval(['curVal = model.pretrain(1).' curName ';']);
    curParams(i) = curVal;
end
optimize.pretrain.past(curNetwork, :) = curParams;
optimize.pretrain.errors(curNetwork, :) = errors;
% Generate past and errors structure for training type
preTNames = optimize.trainNames;
curParams = zeros(1, length(optimize.trainNames));
for i = 1:length(preTNames)
    curName = preTNames{i};
    eval(['curVal = model.train.' curName ';']);
    curParams(i) = curVal;
end
optimize.train.past(curNetwork, :) = curParams;
optimize.train.errors(curNetwork, :) = errors;
end