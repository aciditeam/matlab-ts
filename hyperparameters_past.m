function [optimize] = hyperparameters_past(optimize, trainType, pretrainType, nbDatasets, nbNetworks)
% Generate past and errors structure for pre-training type
optimize.pretrainSteps = [optimize.pretrain.step optimize.pretrain.(pretrainType).step];
optimize.pretrainValues = [optimize.pretrain.values optimize.pretrain.(pretrainType).values];
optimize.pretrainNames = [optimize.pretrain.names optimize.pretrain.(pretrainType).names];
optimize.pretrainContinuous = [optimize.pretrain.continuous optimize.pretrain.(pretrainType).continuous];
optimize.pretrain.past = zeros(nbNetworks, length(optimize.pretrainNames));
optimize.pretrain.errors = zeros(nbNetworks, nbDatasets);
% Generate past and errors structure for training type
optimize.trainSteps = [optimize.train.step optimize.train.(trainType).step];
optimize.trainValues = [optimize.train.values optimize.train.(trainType).values];
optimize.trainNames = [optimize.train.names optimize.train.(trainType).names];
optimize.trainContinuous = [optimize.train.continuous optimize.train.(trainType).continuous];
optimize.train.past = zeros(nbNetworks, length(optimize.trainNames));
optimize.train.errors = zeros(nbNetworks, nbDatasets);
end