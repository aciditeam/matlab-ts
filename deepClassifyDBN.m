function [errorRate, model, trainTime, testTime] = deepClassifyDBN(dataset, model, trainSeries, trainLabels, testSeries, testLabels)
fprintf('Classifying %s with DBN.\n', dataset);
% Adjust size of final layer
nbLayers = model.nbLayers;
nbClasses = length(unique(trainLabels));
model.train.structure.layers(end) = nbClasses;
layers = model.train.structure.layers;
% Initialize the network
model.train.rec.W = cell(nbLayers-2, 1);
model.train.rec.biases = cell(nbLayers-2, 1);
for l = 1:nbLayers-2
    model.train.rec.biases{l} = zeros(layers(l), 1);
    if l < nbLayers
        model.train.rec.W{l} = 1/sqrt(layers(l)+layers(l+1)) * randn(layers(l), layers(l+1));
    end
end
model.train.gen.W = cell(nbLayers-2, 1);
model.train.gen.biases = cell(nbLayers-2, 1);
for l = 1:nbLayers-2
    model.train.gen.biases{l} = zeros(layers(l), 1);
    if l < nbLayers
        model.train.gen.W{l} = 1/sqrt(layers(l)+layers(l+1)) * randn(layers(l), layers(l+1));
    end
end
model.train.top.W = 1/sqrt(layers(end-1)+layers(end)) * randn(layers(end-1), layers(end));
model.train.top.vbias = zeros(layers(end-1), 1);
model.train.top.hbias = zeros(layers(end), 1);
% Gather weights from pre-training
for l = 1:nbLayers-2
    if l > 1
        model.train.gen.biases{l} = (model.train.gen.biases{l} + model.pretrain(l).vbias)/2;
    else
        model.train.gen.biases{l} = model.pretrain(l).vbias;
    end
    model.train.gen.biases{l+1} = model.pretrain(l).hbias;
    model.train.gen.W{l} = model.pretrain(l).W;

    if l > 1
        model.train.rec.biases{l} = (model.train.rec.biases{l} + model.pretrain(l).vbias)/2;
    else
        model.train.rec.biases{l} = model.pretrain(l).vbias;
    end
    model.train.rec.biases{l+1} = model.pretrain(l).hbias;
    model.train.rec.W{l} = model.pretrain(l).W;
end
% Init top layer
model.train.top.W = model.pretrain(nbLayers-1).W;
model.train.top.vbias = model.pretrain(nbLayers-1).vbias;
model.train.top.hbias = model.pretrain(nbLayers-1).hbias;
fprintf(1, 'Training DBN\n');
tic; model.train = dbn (model.train, trainSeries);
fprintf(1, 'Training is done after %f seconds\n', toc);
% Plug a softmax layer on top of it
fprintf('Training the classifier: ');
dbn_feature = dbn_get_hidden(trainSeries, model.train);
modelSoft = train(trainLabels, sparse(double(dbn_feature)), '-s 0');
trainTime = toc;
tic; dbn_feature = dbn_get_hidden(testSeries, model.train);
[L, accuracy, probs] = predict(testLabels, sparse(double(dbn_feature)), modelSoft, '-b 1');
disp('DBN Resulting classes');
disp('Dataset classes :');
disp(unique(testLabels));
disp('Dataset counts :');
n = hist(testLabels, unique(testLabels));
disp(n);
disp('Labels uniques :');
disp(unique(L));
disp('Labels counts :');
n = hist(L, unique(L));
disp(n);
testTime = toc;
errorRate = (100 - accuracy) / 100;
fprintf(2, 'Done\n');
end
