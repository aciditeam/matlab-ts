addpath(genpath('.'));

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

%% --------------
%
% Hyper-parameters
%
% --------------
% Perform individual training
individualTraining = 0;
% Factor of resampling for the time series
resampleFactor = 128;
% Use the densification of the manifold
densifyManifold = 0;
% Whiten the data in pre-processing
whitenData = 0;
% Use the initialization of the weights as randomly-centered gaussians
gaussianInit = 0;
gaussianVariance = 0.5;
% Should we perform a random search
architectureRandomSearch = 0;

% --------------
% 
% Import all datasets
%
% --------------
[fullTrainSeries, fullTrainLabels, trainLabels, trainSeries] = datasetImportTS(mainDirectory, datasets, 'TRAIN', resampleFactor, densifyManifold);
[fullTestSeries, fullTestLabels, testLabels, testSeries] = datasetImportTS(mainDirectory, datasets, 'TEST', resampleFactor, 0);

% --------------
% 
% Pre-processing
%
% --------------
[fullTrainSeries, fullTrainLabels] = datasetPreprocessTS(fullTrainSeries, fullTrainLabels, whitenData);

% --------------
%
% Random architecture search
%
% --------------
if architectureRandomSearch
    deepStructure
else
    layers = {[],[], ...
        [resampleFactor 512 10], ...
        [resampleFactor 512 512 10], ...
        [resampleFactor 512 512 512 10], ...
        [resampleFactor 512 512 512 512 10], ...
        [resampleFactor 512 512 512 512 512 10], ...
    };
end

%%
% REMOVE
% REMOVE
%
% This should be removed before HPC launch (parameters to each core)
%
% REMOVE
% REMOVE
nbLayers = 4;
typePretrain = 'DAE';
typeTrain = 'SDAE';

% --------------
%
% Random parameters search
%
% --------------
repeat = 2;
nbSteps = 100;
nbBatch = 2;
nbNetworks = nbSteps * nbBatch;
% Initialize the optimizer structure
optimize = hyperparameters_optimize(nbLayers);
% Prepare the past structures
optimize = hyperparameters_past(optimize, typeTrain, typePretrain, length(datasets), nbNetworks);
curNetwork = 1;
nextBatch = {};
% Create one example model
model = hyperparameters_init(optimize, nbLayers, resampleFactor, 10, typePretrain, typeTrain, '/tmp', 'default', 1, layers{nbLayers});
% Optimization step
for steps = 1:nbSteps
    % Architecture batch
    for batch = 1:nbBatch
        if (isempty(nextBatch) || ((batch-1) > (steps / nbBatch)))
            model = hyperparameters_init(optimize, nbLayers, resampleFactor, 10, typePretrain, typeTrain, '/tmp', 'random', 1, layers{nbLayers});
        else
            model = nextBatch{batch};
        end
        errorRates = ones(length(datasets), repeat);
        %try
            for r = 1:repeat
                if ~individualTraining
                    model = deepPretrain(model, typePretrain, fullTrainSeries);
                end
                for d = 1:length(datasets)
                    fprintf('* Dataset %s.\n', datasets{d});
                    if individualTraining
                        [trainSeries{d}, trainLabels{d}] = datasetPreprocessTS(trainSeries{d}, trainLabels{d}, whitenData);
                        model = deepPretrain(model, typePretrain, fullTrainSeries);
                    end
                    [errorRates(d, r), eFeat, eSoft, model, trainT, testT] = deepClassify(datasets{d}, typeTrain, model, trainSeries{d}, trainLabels{d}, testSeries{d}, testLabels{d}, 0);
                end
            end
        %catch
            disp('Erroneous network configurations.');
        %end
        optimize = hyperparameters_gather(optimize, model, curNetwork, mean(errorRates, 2));
        curNetwork = curNetwork + 1;
    end
    % Extract current error rates and architectures
    curError = optimize.pretrain.errors(1:(curNetwork-1), :);
    curValue = [optimize.pretrain.past(1:(curNetwork-1), :) optimize.train.past(1:(curNetwork-1), :)];
    % Rank different architecture against each other
    ranks = hyperparameters_criticaldifference(curError');
    % Generate a grid that will in fact be a very wide set of random models
    finalGrid = hyperparameters_grid(model, optimize, 'full', 1e6);
    % Find the next values of parameters to evaluate
    nextBatch = hyperparameters_fit(optimize, model, curValue, ranks, finalGrid, nbLayers, 10, 'full');
    % Save the current state of optimization (only errors and structures)
    save(['optimizedParameters_' typeTrain '_' typePretrain '_' num2str(nbLayers) '.mat'], 'optimize');
end
return;

%% Checking parameters

hStruct = hyperparameters_optimize(4);
fprintf('Pre-train:\n');
curName = hStruct.pretrain.names;
curVals = hStruct.pretrain.values;
curCont = hStruct.pretrain.continuous;
curStep = hStruct.pretrain.step;
curDefs = hStruct.pretrain.default;
for i = 1:length(curName)
    fprintf('%s \t (%d) d:%f %f : %f : %f\n', curName{i}, curCont(i), curDefs(i), curVals{i}(1), curStep(i), curVals{i}(end));
end
fprintf('DAE:\n');
curName = hStruct.pretrain.DAE.names;
curVals = hStruct.pretrain.DAE.values;
curCont = hStruct.pretrain.DAE.continuous;
curStep = hStruct.pretrain.DAE.step;
curDefs = hStruct.pretrain.DAE.default;
for i = 1:length(curName)
    fprintf('%s \t (%d) d:%f %f : %f : %f\n', curName{i}, curCont(i), curDefs(i), curVals{i}(1), curStep(i), curVals{i}(end));
end
fprintf('RBM:\n');
curName = hStruct.pretrain.RBM.names;
curVals = hStruct.pretrain.RBM.values;
curCont = hStruct.pretrain.RBM.continuous;
curStep = hStruct.pretrain.RBM.step;
curDefs = hStruct.pretrain.RBM.default;
for i = 1:length(curName)
    fprintf('%s \t (%d) d:%f %f : %f : %f\n', curName{i}, curCont(i), curDefs(i), curVals{i}(1), curStep(i), curVals{i}(end));
end
fprintf('Train:\n');
curName = hStruct.train.names;
curVals = hStruct.train.values;
curCont = hStruct.train.continuous;
curStep = hStruct.train.step;
curDefs = hStruct.train.default;
for i = 1:length(curName)
    fprintf('%s \t (%d) d:%f %f : %f : %f\n', curName{i}, curCont(i), curDefs(i), curVals{i}(1), curStep(i), curVals{i}(end));
end
fprintf('SDAE:\n');
curName = hStruct.train.SDAE.names;
curVals = hStruct.train.SDAE.values;
curCont = hStruct.train.SDAE.continuous;
curStep = hStruct.train.SDAE.step;
curDefs = hStruct.train.SDAE.default;
for i = 1:length(curName)
    fprintf('%s \t (%d) d:%f %f : %f : %f\n', curName{i}, curCont(i), curDefs(i), curVals{i}(1), curStep(i), curVals{i}(end));
end
fprintf('DBM:\n');
curName = hStruct.train.DBM.names;
curVals = hStruct.train.DBM.values;
curCont = hStruct.train.DBM.continuous;
curStep = hStruct.train.DBM.step;
curDefs = hStruct.train.DBM.default;
for i = 1:length(curName)
    fprintf('%s \t (%d) d:%f %f : %f : %f\n', curName{i}, curCont(i), curDefs(i), curVals{i}(1), curStep(i), curVals{i}(end));
end
fprintf('DBN:\n');
curName = hStruct.train.DBN.names;
curVals = hStruct.train.DBN.values;
curCont = hStruct.train.DBN.continuous;
curStep = hStruct.train.DBN.step;
curDefs = hStruct.train.DBN.default;
for i = 1:length(curName)
    fprintf('%s \t (%d) d:%f %f : %f : %f\n', curName{i}, curCont(i), curDefs(i), curVals{i}(1), curStep(i), curVals{i}(end));
end
fprintf('MLP:\n');
curName = hStruct.train.MLP.names;
curVals = hStruct.train.MLP.values;
curCont = hStruct.train.MLP.continuous;
curStep = hStruct.train.MLP.step;
curDefs = hStruct.train.MLP.default;
for i = 1:length(curName)
    fprintf('%s \t (%d) d:%f %f : %f : %f\n', curName{i}, curCont(i), curDefs(i), curVals{i}(1), curStep(i), curVals{i}(end));
end

%% Test instantiate 
model = hyperparameters_init(hStruct, 5, 128, 10, 'DAE', 'SDAE', '/tmp', 'random', 1);
disp(model);
model = hyperparameters_init(hStruct, 5, 128, 10, 'DAE', 'DBN', '/tmp', 'random', 1);
disp(model);
model = hyperparameters_init(hStruct, 5, 128, 10, 'DAE', 'DBM', '/tmp', 'random', 1);
disp(model);
model = hyperparameters_init(hStruct, 5, 128, 10, 'DAE', 'MLP', '/tmp', 'random', 1);
disp(model);
model = hyperparameters_init(hStruct, 5, 128, 10, 'RBM', 'SDAE', '/tmp', 'random', 1);
disp(model);
model = hyperparameters_init(hStruct, 5, 128, 10, 'RBM', 'DBN', '/tmp', 'random', 1);
disp(model);
model = hyperparameters_init(hStruct, 5, 128, 10, 'RBM', 'DBM', '/tmp', 'random', 1);
disp(model);
model = hyperparameters_init(hStruct, 5, 128, 10, 'RBM', 'MLP', '/tmp', 'random', 1);
disp(model);

% ---------------------
%
% Convolutional Network
%
% ---------------------

% TODO
% TODO
% TODO
% Warning here about the convolution
% What we really want is to do 1-dimensional convolution
% Which I assume will not be the case !
% TODO
% TODO
% TODO

%%
% + Here we COULD avoid resampling !
%
[fullTrainSeries, fullTrainLabels, trainLabels, trainSeries] = datasetImportTS(mainDirectory, datasets, 'TRAIN', 1024, densifyManifold);
[fullTestSeries, fullTestLabels, testLabels, testSeries] = datasetImportTS(mainDirectory, datasets, 'TEST', 1024, 0);
for d = 1:length(datasets)
% Setting up the network architecture
pad_k = 2;
pad_v = 0;
trainSeriesPad = padimages(trainSeries{d}, 32, 3, pad_k, pad_v);
testSeriesPad = padimages(testSeries{d}, 32, 3, pad_k, pad_v);
size_in = 32;
channel_in = 1;
full_layers = [2000, 2000, 10];
conv_layers = [32, 64, 32, 64]; % 32 5x5 filters x 2
poolratios = [3, 3]; % 3x3 pooling x 2
pooling = [0, 1]; % max pooling + average pooling
strides = [1, 1]; % every data point
% construct convnet
C = default_convnet (size_in, channel_in, full_layers, conv_layers, poolratios, strides);
% Pooling layers
C.pooling = pooling;
% Learning parameters
C.learning.lrate = 1e-3;
C.learning.lrate0 = 5000;
C.learning.momentum = 0;
C.learning.weight_decay = 0.0005;
C.learning.minibatch_sz = 32;
C.hidden.use_tanh = 2;
C.conv.use_tanh = 2;
% Adadelta parameters
C.adadelta.use = 1;
C.adadelta.momentum = 0.95;
C.adadelta.epsilon = 1e-8;
% Normalization
C.do_normalize = 1;
C.do_normalize_std = 1;
% Whitening
if 0
    D.do_normalize = 0;
    D.do_normalize_std = 0;
end
% Dropout and noise
C.dropout.use = 1;
C.noise.drop = 0.2;
C.noise.level = 0.1;
% Use lcn
C.lcn.use = 1;
C.lcn.neigh = 4;
% Number of iterations
C.iteration.n_epochs = 150;
C.valid_min_epochs = 50;
% Set the stopping criterion
C.stop.criterion = 0;
C.stop.recon_error.tolerate_count = 1000;
% save the intermediate data after every epoch
C.hook.per_epoch = {@save_intermediate, {'convnet_cifar10.mat'}};
% print learining process
C.verbose = 1;
% train RBM
fprintf(1, 'Training convnet\n');
tic;
C = convnet(C, trainSeriesPad, trainLabels{d});
fprintf(1, 'Training is done after %f seconds\n', toc);
if C.do_normalize
    % make it zero-mean
    Xm = mean(X, 1);
    X_test = bsxfun(@minus, X_test, Xm);
end
if C.do_normalize_std
    % make it unit-variance
    Xs = std(X, [], 1);
    X_test = bsxfun(@rdivide, X_test, Xs);
end
[pred] = convnet_classify(C, testSeriesPad);
n_correct = sum(testLabels{d} == pred);
fprintf(2, 'Correctly classified test samples: %d/%d\n', n_correct, size(X_test, 1));
end

%-----------------
%
% Deep Belief Network
%
%-----------------

% add the path of RBM code
addpath('..');

% load MNIST
load 'mnist_14x14.mat';

% shuffle the training data
perm_idx = randperm (size(X,1));
X = X(perm_idx, :);
X_labels = X_labels(perm_idx);

layers = [size(X, 2) 500 500 1000];
n_layers = length(layers);

Rs = cell(n_layers, 1);

% construct RBM and use default configurations
H = X;

for l=1:n_layers-1
    R = default_rbm (size(H, 2), layers(l+1));

    R.data.binary = 1;

    mH = mean(H, 1)';
    R.vbias = min(max(log(mH./(1 - mH)), -4), 4);
    %R.hbias = -4 * ones(size(R.hbias));

    R.learning.lrate = 1e-3;

    R.learning.persistent_cd = 0;
    R.parallel_tempering.use = 0;
    R.adaptive_lrate.use = 1;
    R.adaptive_lrate.lrate_ub = R.learning.lrate;
    R.enhanced_grad.use = 1;
    R.learning.minibatch_sz = 256;

    % max. 100 epochs
    R.iteration.n_epochs = 200;

    % set the stopping criterion
    R.stop.criterion = 0;
    R.stop.recon_error.tolerate_count = 1000;

    % save the intermediate data after every epoch
    R.hook.per_epoch = {@save_intermediate, {sprintf('rbm_%d.mat', l)}};
    R.hook.per_update = {};

    % print learining process
    R.verbose = 0;
    R.debug.do_display = 0;
    R.debug.display_interval = 10;
    R.debug.display_fid = 1;
    R.debug.display_function = @visualize_rbm;

    % train RBM
    fprintf(1, 'Training RBM\n');
    tic;
    R = train_rbm (R, H);
    fprintf(1, 'Training is done after %f seconds\n', toc);

    Rs{l} = R;

    H = rbm_get_hidden(H, R);
end


D = default_dbn (layers);

D.hook.per_epoch = {@save_intermediate, {'dbn_mnist.mat'}};

D.learning.lrate = 1e-3;
D.learning.lrate0 = 5000;
D.learning.momentum = 0;
D.learning.weight_decay = 0.0001;
D.learning.minibatch_sz = 256;

D.learning.contrastive_step = 10;
D.learning.persistent_cd = 0;
D.learning.ffactored = 1;

D.iteration.n_epochs = 200;

for l = 1:n_layers-2
    if l > 1
        D.gen.biases{l} = (D.gen.biases{l} + Rs{l}.vbias)/2;
    else
        D.gen.biases{l} = Rs{l}.vbias;
    end
    D.gen.biases{l+1} = Rs{l}.hbias;
    D.gen.W{l} = Rs{l}.W;

    if l > 1
        D.rec.biases{l} = (D.rec.biases{l} + Rs{l}.vbias)/2;
    else
        D.rec.biases{l} = Rs{l}.vbias;
    end
    D.rec.biases{l+1} = Rs{l}.hbias;
    D.rec.W{l} = Rs{l}.W;
end

D.top.W = Rs{n_layers-1}.W;
D.top.vbias = Rs{n_layers-1}.vbias;
D.top.hbias = Rs{n_layers-1}.hbias;

fprintf(1, 'Training DBN\n');
tic;
D = dbn (D, X);
fprintf(1, 'Training is done after %f seconds\n', toc);

n_chains = 20;
n_samples = 11;
rndidx = randperm(size(X, 1));
Sall = zeros(n_samples * n_chains, size(X, 2));
for ci = 1:n_chains
    %S = dbn_sample(rand(1, size(X, 2)), D, n_samples, 1);
    S = dbn_sample(X(rndidx(ci),:), D, n_samples-1, 1);
    Sall(((ci-1) * n_samples + 1), :) = X(rndidx(ci),:);
    Sall(((ci-1) * n_samples + 2):(ci * n_samples), :) = S;
end
save 'dbn_samples.mat' Sall;

% ---------------
%
% Stacked DAE + finetuning
%
% ---------------

% add the path of RBM code
addpath('..');
addpath('~/work/Algorithms/liblinear-1.7/matlab');

% load MNIST
load 'mnist_14x14.mat';

% shuffle the training data
perm_idx = randperm (size(X,1));

n_all = size(X, 1);
n_train = ceil(n_all * 3 / 4);
n_valid = floor(n_all /4);

X_valid = X(perm_idx(n_train+1:end), :);
X_valid_labels = X_labels(perm_idx(n_train+1:end));
X = X(perm_idx(1:n_train), :);
X_labels = X_labels(perm_idx(1:n_train));

layers = [size(X,2), 200, 100, 50, 2];
n_layers = length(layers);
blayers = [1, 1, 1, 1, 0];

use_tanh = 0;
do_pretrain = 1;

if do_pretrain
    Ds = cell(n_layers - 1, 1);
    H = X;
    H_valid = X_valid;

    for l = 1:n_layers-1
        % construct DAE and use default configurations
        D = default_dae (layers(l), layers(l+1));

        D.data.binary = blayers(l);
        D.hidden.binary = blayers(l+1);

        if use_tanh 
            if l > 1
                D.visible.use_tanh = 1;
            end
            D.hidden.use_tanh = 1;
        else
            if D.data.binary
                mH = mean(H, 1)';
                D.vbias = min(max(log(mH./(1 - mH)), -4), 4);
            else
                D.vbias = mean(H, 1)';
            end
        end

        D.learning.lrate = 1e-1;
        D.learning.lrate0 = 5000;
        D.learning.weight_decay = 0.0001;
        D.learning.minibatch_sz = 128;

        D.valid_min_epochs = 10;

        D.noise.drop = 0.2;
        D.noise.level = 0;

        %D.adagrad.use = 1;
        %D.adagrad.epsilon = 1e-8;
        D.adagrad.use = 0;
        D.adadelta.use = 1;
        D.adadelta.epsilon = 1e-8;
        D.adadelta.momentum = 0.99;

        D.iteration.n_epochs = 500;

        % save the intermediate data after every epoch
        D.hook.per_epoch = {@save_intermediate, {sprintf('dae_mnist_%d.mat', l)}};

        % print learining process
        D.verbose = 0;
        % display the progress
        D.debug.do_display = 0;

        % train RBM
        fprintf(1, 'Training DAE (%d)\n', l);
        tic;
        D = dae (D, H, H_valid, 0.1);
        fprintf(1, 'Training is done after %f seconds\n', toc);

        H = dae_get_hidden(H, D);
        H_valid = dae_get_hidden(H_valid, D);

        Ds{l} = D;
    end
end

S = default_sdae (layers);

S.data.binary = blayers(1);
S.bottleneck.binary = blayers(end);
S.hidden.use_tanh = use_tanh;

S.hook.per_epoch = {@save_intermediate, {'sdae_mnist.mat'}};

S.learning.lrate = 1e-1;
S.learning.lrate0 = 5000;
%S.learning.momentum = 0.9;
S.learning.weight_decay = 0.0001;
S.learning.minibatch_sz = 128;

%S.noise.drop = 0.2;
%S.noise.level = 0;
S.adadelta.use = 1;
S.adadelta.epsilon = 1e-8;
S.adadelta.momentum = 0.99;

%S.adagrad.use = 1;
%S.adagrad.epsilon = 1e-8;
S.valid_min_epochs = 10;

S.iteration.n_epochs = 100;

if do_pretrain
    for l = 1:n_layers-1
        S.biases{l+1} = Ds{l}.hbias;
        S.W{l} = Ds{l}.W;
    end
else
    if S.data.binary
        mH = mean(X, 1)';
        S.biases{1} = min(max(log(mH./(1 - mH)), -4), 4);
    else
        S.biases{1} = mean(X, 1)';
    end
end

fprintf(1, 'Training sDAE\n');
tic;
S = sdae (S, X, X_valid, 0.1);
fprintf(1, 'Training is done after %f seconds\n', toc);

H = sdae_get_hidden (X, S);
save 'sdae_mnist_vis.mat' H X_labels;

% -------------------
%
% Stacked DAE + RBM + finetuning
%
% -------------------

% load MNIST
load 'mnist_14x14.mat';

% shuffle the training data
perm_idx = randperm (size(X,1));

n_all = size(X, 1);
n_train = ceil(n_all * 3 / 4);
n_valid = floor(n_all /4);

X_valid = X(perm_idx(n_train+1:end), :);
X_valid_labels = X_labels(perm_idx(n_train+1:end));
X = X(perm_idx(1:n_train), :);
X_labels = X_labels(perm_idx(1:n_train));

layers = [size(X,2), 200, 100, 50, 2];
n_layers = length(layers);
blayers = [1, 1, 1, 1, 0];

use_tanh = 0;
do_pretrain = 1;

if do_pretrain
    Ds = cell(n_layers - 1, 1);
    H = X;
    H_valid = X_valid;

    for l = 1:n_layers-2
        % construct RBM and use default configurations
        R = default_rbm (size(H, 2), layers(l+1));

        R.data.binary = blayers(l);

        mH = mean(H, 1)';
        R.vbias = min(max(log(mH./(1 - mH)), -4), 4);
        R.hbias = zeros(size(R.hbias));
        %R.W = 2 / sqrt(layers(l) + layers(l+1)) * (rand(layers(l), layers(l+1)) - 0.5);
        R.W = 0.01 * (randn(layers(l), layers(l+1)));

        R.learning.lrate = 1e-2;
        R.adaptive_lrate.lrate_ub = 1e-2;

        R.learning.persistent_cd = 0;
        R.fast.use = 0;
        R.fast.lrate = R.learning.lrate;

        R.parallel_tempering.use = 0;
        R.adaptive_lrate.use = 1;
        R.enhanced_grad.use = 1;
        R.learning.minibatch_sz = 128;

        M.valid_min_epochs = 10;

        % max. 100 epochs
        R.iteration.n_epochs = 100;

        % set the stopping criterion
        R.stop.criterion = 0;
        R.stop.recon_error.tolerate_count = 1000;

        % save the intermediate data after every epoch
        R.hook.per_epoch = {@save_intermediate, {sprintf('rbm_mnist_%d.mat', l)}};
        R.hook.per_update = {};

        % print learining process
        R.verbose = 0;
        R.debug.do_display = 0;
        R.debug.display_interval = 10;
        R.debug.display_fid = 1;
        R.debug.display_function = @visualize_rbm;

        % train RBM
        fprintf(1, 'Training RBM\n');
        tic;
        R = train_rbm (R, [H; H_valid]);
        fprintf(1, 'Training is done after %f seconds\n', toc);

        Ds{l} = R;

        H = rbm_get_hidden(H, R);
        H_valid = rbm_get_hidden(H_valid, R);
    end

    l = n_layers - 1;

    % construct DAE and use default configurations
    D = default_dae (layers(l), layers(l+1));

    D.data.binary = blayers(l);
    D.hidden.binary = blayers(l+1);

    if D.data.binary
        mH = mean(H, 1)';
        D.vbias = min(max(log(mH./(1 - mH)), -4), 4);
    else
        D.vbias = mean(H, 1)';
    end

    D.learning.lrate = 1e-1;
    D.learning.lrate0 = 5000;
    D.learning.weight_decay = 0.0001;
    D.learning.minibatch_sz = 128;

    D.noise.drop = 0.2;
    D.noise.level = 0;

    D.valid_min_epochs = 10;
    D.adagrad.use = 1;
    D.adagrad.epsilon = 1e-8;

    D.iteration.n_epochs = 500;

    % save the intermediate data after every epoch
    D.hook.per_epoch = {@save_intermediate, {sprintf('dae_mnist_%d.mat', l)}};

    % print learining process
    D.verbose = 0;
    % display the progress
    D.debug.do_display = 0;

    % train RBM
    fprintf(1, 'Training DAE (%d)\n', l);
    tic;
    D = dae (D, H, H_valid, 0.1);
    fprintf(1, 'Training is done after %f seconds\n', toc);

    Ds{l} = D;
end

S = default_sdae (layers);

S.data.binary = blayers(1);
S.bottleneck.binary = blayers(end);
S.hidden.use_tanh = use_tanh;

S.hook.per_epoch = {@save_intermediate, {'sdae_rbm_mnist.mat'}};

S.learning.lrate = 1e-3;
S.learning.lrate0 = 1000;
%S.learning.momentum = 0.5;
%S.learning.weight_decay = 0.0001;
S.learning.minibatch_sz = 256;

S.valid_min_epochs = 10;

S.adagrad.use = 1;
S.adagrad.epsilon = 1e-8;

%S.noise.drop = 0.2;
S.noise.level = 0;

S.iteration.n_epochs = 100;

if do_pretrain
    for l = 1:n_layers-1
        if l > 1
            if use_tanh
                S.biases{l+1} = Ds{l}.hbias;
                S.W{l} = Ds{l}.W;
            else
                S.biases{l+1} = Ds{l}.hbias + sum(Ds{l}.W, 1)';
                S.W{l} = Ds{l}.W / 2;
            end
        else
            S.biases{l+1} = Ds{l}.hbias;
            S.W{l} = Ds{l}.W;
        end
    end
end

fprintf(1, 'Training sDAE\n');
tic;
S = sdae (S, X, X_valid, 0.1);
fprintf(1, 'Training is done after %f seconds\n', toc);

H = sdae_get_hidden (X, S);
save 'sdae_rbm_mnist_vis.mat' H X_labels;

vis_mnist_rbm;

% 
% = Generative Stochastic Network =
%  - A simple implementation of GSN according to (Bengio et al., 2013)
% 
% = Convolutional Neural Network =
%  - A naive implementation (purely using Matlab)
%  - Pooling: max (Jonathan Masci's code) and average
%  - Not for serious use!
% 
% = Restricted Boltzmann Machine & Deep Belief Networks =
%  - Binary/Gaussian Visible Units + Binary Hidden Units
%  - Enhanced Gradient, Adaptive Learning Rate
%  - Adadelta for RBM
%  - Contrastive Divergence
%  - (Fast) Persistent Contrastive Divergence
%  - Parallel Tempering
%  - DBN: Up-down Learning Algorithm
% 
% = Deep Boltzmann Machine =
%  - Binary/Gaussian Visible Units + Binary Hidden Units
%  - (Persistent) Contrastive Divergence
%  - Enhanced Gradient, Adaptive Learning Rate
%  - Two-stage Pretraining Algorithm (example)
%  - Centering Trick (fixed center variables only)
% 
% = Denoising Autoencoder (Tied Weights) =
%  - Binary/Gaussian Visible Units + Binary(Sigmoid)/Gaussian Hidden Units
%  - tanh/sigm/relu nonlinearities
%  - Shallow: sparsity, contractive, soft-sparsity (log-cosh) regularization
%  - Deep: stochastic backprop
%  - Adagrad, Adadelta
% 
% = Multi-layer Perceptron =
%  - Stochastic Backpropagation, Dropout
%  - tanh/sigm/relu nonlinearities
%  - Adagrad, Adadelta
%  - Balanced minibatches using crossvalind()

