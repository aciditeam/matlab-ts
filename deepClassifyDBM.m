function [errorRate, model, trainTime, testTime] = deepClassifyDBM(dataset, model, trainSeries, trainLabels, testSeries, testLabels, params)
fprintf('Classifying %s with DBM.\n', dataset);
% Fix the last layers
nbClasses = length(unique(trainLabels));
model.train.structure.layers(end) = nbClasses;
layers = model.train.structure.layers;
nbLayers = length(layers);
Qpre = cell(nbLayers, 1);
Qpre_mask = zeros(nbLayers, 1);
Xp = trainSeries;
for l = 1:nbLayers
    if mod(l, 2) == 0
        continue;
    end
	if l+2 > nbLayers
        break;
	end
    Xp = rbm_get_hidden(Xp, model.pretrain(l));
    Qpre{l+2} = Xp;
    Qpre_mask(l+2) = 1;
end
% pretraining (stage 2)
model.train.learning.persistent_cd = 0;
model.train.learning.lrate = 1e-2;
model.train.learning.lrate0 = 1000;
model.train.adaptive_lrate.lrate_ub = model.train.learning.lrate;
fprintf(1, 'Training DBM\n');
tic; [model.train] = dbm (model.train, trainSeries, 1, Qpre, Qpre_mask);
fprintf(1, 'Training is done.\n');
% finetuning (stage 3)
model.train.learning.persistent_cd = 1;
model.train.learning.lrate = 1e-4;
model.train.learning.lrate0 = 5000;
mH = mean(trainSeries, 1)';
model.train.biases{1} = min(max(log(mH./(1 - mH)), -4), 4);
if model.train.centering.use
    model.train = set_dbm_centers(model.train);
end
fprintf(1, 'Finetuning DBM\n');
[model.train] = dbm (model.train, trainSeries);
fprintf(1, 'Finetuning is done.\n');
% classification
[Q_mf] = dbm_get_hidden(trainSeries, model.train, 30, 1e-6, model.train.mf.reg);
[Q_test] = dbm_get_hidden(testSeries, model.train, 30, 1e-6, model.train.mf.reg);
M = default_mlp(layers);
M = set_mlp_dbm(M);
M.output.binary = 1;
M.hidden.use_tanh = 0;
M.dropout.use = 0.5;
M.hook.per_epoch = {@save_intermediate, [model.train.outFiles '_dbm_mlp.mat']};
M.learning.lrate = model.train.learning.lrate;
M.learning.lrate0 = 5000;
M.learning.momentum = 0.9;
M.learning.weight_decay = 0.0001;
M.learning.minibatch_sz = 128; 
M.adagrad.use = 1;
M.adagrad.epsilon = 1e-8;
M.noise.drop = 0;
M.noise.level = 0;
M.iteration.n_epochs = model.train.iteration.n_epochs;
for l = 1:nbLayers
    M.biases{l} = model.train.biases{l};
    if D.centering.use
        if l > 1
            M.biases{l} = M.biases{l} - M.W{l-1}' * model.train.centering.centers{l-1};
        end
        if l < nbLayers
            M.biases{l} = M.biases{l} - M.W{l} * model.train.centering.centers{l+1};
        end
    end
    if l < nbLayers
        M.W{l} = model.train.W{l};
        M.dbm.W{l} = model.train.W{l};
    end
end
fprintf(1, 'Training MLP\n');
M = mlp_dbm (M, trainSeries, Q_mf, trainLabels);
trainTime = toc;
tic; [pred] = mlp_classify (M, testSeries, Q_test);
testTime = toc;
n_correct = sum(testLabels == pred);
disp('Softmax Resulting classes');
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
errorRate = (size(testSeries, 1) - n_correct) / (size(testSeries, 1));
fprintf(2, 'Error rate: %f\n', errorRate);
end
