function [errorRate, model, trainTime, testTime] = deepClassifyMLP(dataset, model, trainSeries, trainLabels, sampledSeries, testLabels)
fprintf('* Classifying %s with MLP.\n', dataset);
% Adjust size of final layer
nbLayers = model.nbLayers;
nbClasses = length(unique(trainLabels));
model.train.structure.layers(end) = nbClasses;
% DEBUG
% DEBUG
model.train.output.binary = 1;
model.train.dropout.use = 0;
model.train.adadelta.use = 1;
model.train.adadelta.epsilon = 1e-8;
model.train.adadelta.momentum = 0.9;
model.train.learning.lrate = 1e-3;
model.train.learning.lrate0 = 5000;
model.train.learning.minibatch_sz = 32;
% DEBUG
% DEBUG
layers = model.train.structure.layers;
% Adjust the dropout probability
if model.train.dropout.use
    for l = 1:nbLayers
        model.train.dropout.probs{l} = model.train.dropout.global_proba;% * ones(1, layers(l));
    end
end
% initializations
model.train.W = cell(nbLayers, 1);
model.train.biases = cell(nbLayers, 1);
for l = 1:nbLayers
    model.train.biases{l} = zeros(layers(l), 1);
    if l < nbLayers
        model.train.W{l} = 2 * sqrt(6)/sqrt(layers(l)+layers(l+1)) * (rand(layers(l), layers(l+1)) - 0.5);
    end
end
if model.train.adagrad.use
    model.train.adagrad.W = cell(nbLayers, 1);
    model.train.adagrad.biases = cell(nbLayers, 1);
    for l = 1:n_layers
        model.train.adagrad.biases{l} = zeros(layers(l), 1);
        if l < n_layers
            model.train.adagrad.W{l} = zeros(layers(l), layers(l+1));
        end
    end
end
if model.train.adadelta.use
    model.train.adadelta.gW = cell(nbLayers, 1);
    model.train.adadelta.gbiases = cell(nbLayers, 1);
    model.train.adadelta.W = cell(nbLayers, 1);
    model.train.adadelta.biases = cell(nbLayers, 1);
    for l = 1:nbLayers
        model.train.adadelta.gbiases{l} = zeros(layers(l), 1);
        model.train.adadelta.biases{l} = zeros(layers(l), 1);
        if l < nbLayers
            model.train.adadelta.gW{l} = zeros(layers(l), layers(l+1));
            model.train.adadelta.W{l} = zeros(layers(l), layers(l+1));
        end
    end
end
% Use pre-trained layers as initialization
for l = 1:(nbLayers-2)
    model.train.biases{l+1} = model.pretrain(l).hbias;
    model.train.W{l} = model.pretrain(l).W;
end
% if model.train.adagrad.use
%     model.train.adagrad.W = model.train.W;
% 	model.train.adagrad.biases = model.train.biases;
% end
% if model.train.adadelta.use
%     model.train.adadelta.W = model.train.W;
% 	model.train.adadelta.biases = model.train.biases;
% end
% Perform dataset-wise fine-tuning
tic; model.train = mlp(model.train, trainSeries, trainLabels);
trainTime = toc;
tic; [pred] = mlp_classify(model.train, sampledSeries);
testTime = toc;
disp('MLP Resulting classes');
disp('Dataset classes :');
disp(unique(testLabels));
disp('Dataset counts :');
n = hist(testLabels, unique(testLabels));
disp(n);
disp('Labels uniques :');
disp(unique(pred));
disp('Labels counts :');
n = hist(pred, unique(pred));
disp(n);
n_correct = sum(testLabels == pred);
errorRate = (size(sampledSeries, 1) - n_correct) / size(sampledSeries, 1);
end
