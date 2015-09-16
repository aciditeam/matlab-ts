function [errorRate, model, trainTime, testTime] = deepClassifySDAE(dataset, model, trainSeries, trainLabels, testSeries, testLabels)
fprintf('Classifying %s with SDAE.\n', dataset);
nbLayers = model.nbLayers;
% Retrieve training model
curModel = model.train;
for l = 1:nbLayers-2
    curModel.biases{l+1} = model.pretrain(l).hbias;
    curModel.W{l} = model.pretrain(l).W;
end
% Train the Stacked-DAE
tic; curModel = sdae(curModel, trainSeries);
% Plug a softmax layer on top of it
fprintf(2, 'Training the classifier: ');
sdae_feature = sdae_get_hidden(trainSeries, curModel);
modelSoft = train(trainLabels, sparse(double(sdae_feature)), '-s 0');
trainTime = toc;
fprintf(2, 'Testing the classifier: ');
tic; sdae_feature = sdae_get_hidden(testSeries, curModel);
[L, accuracy, probs] = predict(testLabels, sparse(double(sdae_feature)), modelSoft, '-b 1');
disp('SDAE Resulting classes');
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
% Keep the model after optimization
model.train = curModel;
end
