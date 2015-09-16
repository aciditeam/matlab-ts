addpath(genpath('.'));
% High-level parameters
resampleFactor = 128;
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
% Importing datasets
fprintf('Importing training data\n');
nbSeries = zeros(length(datasets), 1);
trainLabels = cell(length(datasets), 1);
trainSeries = cell(length(datasets), 1);
% First import all training datas
for d = 1:length(datasets)
    fprintf(' * Dataset %s\n', datasets{d});
    curDatasets = [mainDirectory '/' datasets{d} '/' datasets{d} '_TRAIN'];
    curTrain = importdata(curDatasets);
    trainLabels{d} = curTrain(:, 1);
    curSeries = curTrain(:, 2:end);
    sampledSeries = zeros(size(curTrain, 1), resampleFactor);
    % For a starters we perform resampling of the series
    for ts = 1:size(curTrain, 1)
        sampledSeries(ts, :) = resample(curSeries(ts, :), resampleFactor, size(curSeries, 2));
        sampledSeries(ts, :) = (sampledSeries(ts, :) - mean(sampledSeries(ts, :))) ./ max(sampledSeries(ts, :));
    end
    trainSeries{d} = sampledSeries;
    nbSeries(d) = size(curTrain, 1);
end
maxSeries = max(nbSeries);
for d = 1:length(datasets)
    curNbSeries = nbSeries(d);
    seriesMissing = maxSeries - curNbSeries; 
    curTrainSeries = trainSeries{d};
    curTrainLabels = trainLabels{d};
    newSeries = zeros(seriesMissing, resampleFactor);
    newSeriesID = randsample(curNbSeries, seriesMissing, 1);
    newLabels = curTrainLabels(newSeriesID);
    for s = 1:seriesMissing
        tmpSeries = curTrainSeries(newSeriesID(s), :);
        switch floor(rand * 3)
            case 0
                % Add white gaussian noise
                tmpSeries = tmpSeries + random('norm', 0, 0.1, 1, resampleFactor);
            case 1
                % Add random outliers
                tmpIDs = randsample(resampleFactor, floor(resampleFactor / 20), 0);
                tmpSeries(tmpIDs) = -tmpSeries(tmpIDs);
            case 2
                % Add temporal warping
                switchPosition = ceil(rand * (resampleFactor / 4)) + (resampleFactor / 2);
                leftSide = tmpSeries(1:switchPosition);
                rightSide = tmpSeries((switchPosition+1):end);
                direction = sign(rand - 0.5);
                leftSide = resample(leftSide, switchPosition - (direction * 5), switchPosition);
                rightSide = resample(rightSide, resampleFactor - switchPosition + (direction * 5), resampleFactor - switchPosition); 
                tmpSeries = [leftSide rightSide];
        end
        newSeries(s, :) = tmpSeries;
    end
    nbSeries(d) = maxSeries;
    trainSeries{d} = [curTrainSeries ; newSeries];
    trainLabels{d} = [curTrainLabels ; newLabels];
end
idEnd = cumsum(nbSeries);
idStart = idEnd + 1;
idStart = [1 ; idStart(1:end-1)];
fullTrainLabels = zeros(sum(nbSeries), 1);
fullTrainSeries = zeros(sum(nbSeries), resampleFactor);
for d = 1:length(datasets)
    fullTrainLabels(idStart(d):idEnd(d)) = trainLabels{d};
    fullTrainSeries(idStart(d):idEnd(d), :) = trainSeries{d};
end
clear labelsSeries;
clear resampledSeries;
%% --------------
% 
% Pre-processing
%
% --------------

fprintf('Pre-processing all training data.\n')
% Shuffle the training data
perm_idx = randperm(size(fullTrainSeries,1));
fullTrainSeries = fullTrainSeries(perm_idx, :);
fullTrainLabels = fullTrainLabels(perm_idx);

% ------------------------
%
% Contractive Auto-Encoder
%
% ------------------------

% TODO
% TODO
%
% RETRY THE SAME BY NORMALIZING DATA TO [-1, 1] (divide by max(abs(.)) and
% then use the TANH (to allow reconstruction to fit the data ^^)
%
% TODO
% TODO

fprintf('Pre-training network.\n');
% Establish architecture of the network
layers = [size(fullTrainSeries,2), 512, 512, 256, 128, 10];
nbLayers = length(layers);
binaryLayers = [0, 1, 1, 1, 1, 1];
% Parameters of pretraining
use_tanh = 0;
% Pre-trained layers
Ds = cell(nbLayers - 2, 1);
% Set to use as pre-training
H = fullTrainSeries;
for l = 1:nbLayers-2
    % Construct DAE and use default configurations
    D = default_dae(layers(l), layers(l+1));
    % Parameters of current layer
    D.data.binary = binaryLayers(l);
    D.hidden.binary = binaryLayers(l+1);
    % Use of tanh 
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
    % Learning parameters
    D.learning.lrate = 1e-1;
    D.learning.lrate0 = 5000;
    D.learning.momentum = 0.9;
    D.learning.weight_decay = 0.0002;
    D.learning.weight_scale = sqrt(6)/sqrt(layers(l) + layers(l+1));
    D.learning.minibatch_sz = 128;
    D.learning.lrate_anneal = 0.9;
    % Addition of noise (outliers)
    D.noise.drop = 0.4;
    %if ~binaryLayers(l)
        % Addition of nois (gaussian)
        %D.noise.level = 0.1;
    %end
    % Adadelta parameters
    D.adadelta.use = 1;
    %D.adadelta.epsilon = 1e-8;
    %D.adadelta.momentum = 0.99;
    % Minimum number of epochs
    D.valid_min_epochs = 10;
    % Sparsity and additionnal costs
    if binaryLayers(l+1)
        D.cae.cost = 0;
        D.sparsity.target = 0.05;
        D.sparsity.cost = 0.1;
    end
    % Maximum number of iterations
    D.iteration.n_epochs = 500;
    % Gaussian-Bernoulli RBM
    D.do_normalize = 0;
    D.do_normalize_std = 0;
    % Save the intermediate data after every epoch
    D.hook.per_epoch = {@save_intermediate, {sprintf('cae_TSClassifier_%d.mat', l)}};
    % Print learning process
    D.verbose = 0;
    % Display the progress
    D.debug.do_display = 0;
    % Training the DAE of current layer
    fprintf(1, ' * Training DAE layer n.%d\n', l);
    %tic; D = dae (D, H, H_valid, 0.1);
    tic; D = dae(D, H);
    fprintf(1, '    - Done [%f seconds]\n', toc);
    % Get the activations from the current layers (to the next)
    H = dae_get_hidden(H, D);
    %H_valid = dae_get_hidden(H_valid, D);
	% Weights found for current layer
    Ds{l} = D;
end

% ----------------------
%
% Motif analysis attempt
%
% ----------------------
deepTSmotifs(Ds, fullTrainSeries, nbLayers);

%% --------------------
%
% Fine-tuning with MLP
%
% --------------------

fprintf('Performing MLP classification.\n');
rID = fopen('cae_results.txt', 'w');
for d = 1:length(datasets)
    fprintf(rID, ' * Training on %s.\n', datasets{d});
    %
    % MLP Version
    %
    % Temporary save of the dataset's MLP
    % Adjust size of final layer
    layers(end) = length(unique(trainLabels{d}));
    binaryLayers(end) = 1;
    % Multi-layer perceptron
    M = default_mlp(layers);
    M.hook.per_epoch = {@save_intermediate, {['mlp_' datasets{d} '.mat']}};
    M.output.binary = 1;
    %M.hidden.use_tanh = use_tanh;
    % Learning parameters
    %M.valid_min_epochs = 10;
    M.dropout.use = 0;
    % Learning parameters
    M.learning.lrate = 1e-3;
    M.learning.lrate0 = 5000;
    M.learning.minibatch_sz = 32;
    % Ada parameters
    M.adadelta.use = 1;
    M.adadelta.epsilon = 1e-8;
    M.adadelta.momentum = 0.99;
    % Noise parameters
    M.noise.drop = 0;
    M.noise.level = 0;
    % Number of iterations
    M.iteration.n_epochs = 500;
    % Use pre-trained layers as initialization
    for l = 1:nbLayers-2
        M.biases{l+1} = Ds{l}.hbias;
        M.W{l} = Ds{l}.W;
    end
    % Perform dataset-wise fine-tuning
    tic; M = mlp(M, trainSeries{d}, trainLabels{d});
    %
    % SDAE Version
    %
%     layers(end) = length(unique(trainLabels{d}));
%     S = default_sdae(layers);
%     S.data.binary = 0;
%     S.bottleneck.binary = 0;
%     S.hidden.use_tanh = use_tanh;
%     S.hook.per_epoch = {@save_intermediate, {'dae_TSClassifier.mat'}};
%     S.learning.lrate = 1e-1;
%     S.learning.lrate0 = 5000;
%     S.learning.weight_decay = 0.0001;
%     S.learning.minibatch_sz = 128;
%     S.adadelta.use = 1;
%     S.adadelta.epsilon = 1e-8;
%     S.adadelta.momentum = 0.99;
%     S.valid_min_epochs = 10;
%     S.iteration.n_epochs = 100;
%     for l = 1:nbLayers-2
%         S.biases{l+1} = Ds{l}.hbias;
%         S.W{l} = Ds{l}.W;
%     end    
%     S = sdae(S, fullTrainSeries);
%     %M = mlp (M, X, X_labels, X_valid, X_valid_labels, 0.1);
    fprintf(rID, '    - Done [%f seconds]\n', toc);
    curDatasets = [mainDirectory '/' datasets{d} '/' datasets{d} '_TEST'];
    curTest = importdata(curDatasets);
    testLabels = curTest(:, 1);
    curSeries = curTest(:, 2:end);
    sampledSeries = zeros(size(curTest, 1), resampleFactor);
    % For a starters we perform resampling of the series
    for ts = 1:size(curTest, 1)
        sampledSeries(ts, :) = resample(curSeries(ts, :), resampleFactor, size(curSeries, 2));
        sampledSeries(ts, :) = (sampledSeries(ts, :) - mean(sampledSeries(ts, :))) ./ std(sampledSeries(ts, :));
    end
%    fprintf(1, 'Training sDAE\n');
%     H = sdae_get_hidden(sampledSeries, S);
%     disp(H);
    [pred] = mlp_classify(M, sampledSeries);
    n_correct = sum(testLabels == pred);
    fprintf(rID, '    - Recognition [%f seconds]\n', toc);
    fprintf(rID, '    - Correct : %f\n', n_correct / size(sampledSeries, 1));
end
fclose(rID);
return;

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

% Setting up the network architecture
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
if use_whitening
    D.do_normalize = 0;
    D.do_normalize_std = 0;
end
% Dropout and noise
C.dropout.use = 1;
C.noise.drop = 0.2;
C.noise.level = 0.1;

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
C = convnet (C, X, X_labels+1, X_valid, X_valid_labels+1, 0.1);
fprintf(1, 'Training is done after %f seconds\n', toc);

save('convnet_cifar10.mat', 'C');

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

[pred] = convnet_classify (C, X_test);
n_correct = sum(X_test_labels+1 == pred);

fprintf(2, 'Correctly classified test samples: %d/%d\n', n_correct, size(X_test, 1));

%-------------
%
% Denoising auto-encoder test
%
%-------------

% add the path of RBM code
addpath('..');
%addpath('~/work/Algorithms/liblinear-1.7/matlab');

% load MNIST
load 'mnist_14x14.mat';

% shuffle the training data
perm_idx = randperm (size(X,1));
X = X(perm_idx, :);
X_labels = X_labels(perm_idx);

% construct RBM and use default configurations
D = default_dae (size(X, 2), 500);

D.data.binary = 1;
D.hidden.binary = 1;

D.learning.lrate = 1e-2;
D.learning.lrate0 = 10000;
D.learning.momentum = 0.5;
%D.learning.weight_decay = 0.0001;

D.noise.drop = 0.2;
D.noise.level = 0;

D.sparsity.cost = 0.01;
D.sparsity.target = 0.05;

% max. 100 epochs
D.iteration.n_epochs = 1 %100;

% set the stopping criterion
D.stop.criterion = 0;
D.stop.recon_error.tolerate_count = 1000;

% save the intermediate data after every epoch
D.hook.per_epoch = {@save_intermediate, {'dae_mnist.mat'}};

% print learining process
D.verbose = 0;

% display the progress
D.debug.do_display = 0;

% train RBM
fprintf(1, 'Training DAE\n');
tic;
D = dae (D, X);
fprintf(1, 'Training is done after %f seconds\n', toc);

fprintf(2, 'Training the classifier: ');
rbm_feature = 1./(1 + exp(-bsxfun(@plus, X * D.W, D.hbias')));
model = train(X_labels, sparse(double(rbm_feature)), '-s 0');
fprintf(2, 'Done\n');

fprintf(2, 'Testing the classifier: ');
rbm_feature = 1./(1 + exp(-bsxfun(@plus, X_test * D.W, D.hbias')));
[L accuracy probs] = predict(X_test_labels, sparse(double(rbm_feature)), model, '-b 1');
fprintf(2, 'Done\n');

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
% RBM MLP
%
% ---------------

% add the path of RBM code
addpath('..');
addpath('~/work/Algorithms/liblinear-1.7/matlab');

% load MNIST
load 'mnist_14x14.mat';

% shuffle the training data
X_labels = X_labels + 1;
X_test_labels = X_test_labels + 1;

perm_idx = randperm (size(X,1));

n_all = size(X, 1);
n_train = ceil(n_all * 3 / 4);
n_valid = floor(n_all /4);

X_valid = X(perm_idx(n_train+1:end), :);
X_valid_labels = X_labels(perm_idx(n_train+1:end));
X = X(perm_idx(1:n_train), :);
X_labels = X_labels(perm_idx(1:n_train));

layers = [size(X,2), 1000, 500, 10];
n_layers = length(layers);
blayers = [1, 1, 1, 1];

use_tanh = 0;
do_pretrain = 1;

if do_pretrain
    Ds = cell(n_layers - 2, 1);
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

        R.learning.lrate = 1e-3;
        R.adaptive_lrate.lrate_ub = 1e-3;

        R.learning.persistent_cd = 0;
        R.fast.use = 0;

        R.parallel_tempering.use = 0;
        R.adaptive_lrate.use = 1;
        R.enhanced_grad.use = 1;
        R.learning.minibatch_sz = 128;

        % max. 100 epochs
        R.iteration.n_epochs = 100;

        % set the stopping criterion
        R.stop.criterion = 0;
        R.stop.recon_error.tolerate_count = 1000;

        % save the intermediate data after every epoch
        R.hook.per_epoch = {@save_intermediate, {sprintf('mlp_rbm_mnist_%d.mat', l)}};
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
end

M = default_mlp (layers);

M.output.binary = blayers(end);
M.hidden.use_tanh = use_tanh;
M.dropout.use = 1;

M.hook.per_epoch = {@save_intermediate, {'mlp_mnist.mat'}};

M.learning.lrate = 1e-2;
M.learning.lrate0 = 5000;
M.learning.minibatch_sz = 128;

M.adadelta.use = 1;

M.noise.drop = 0;
M.noise.level = 0;

M.iteration.n_epochs = 100;

if do_pretrain
    for l = 1:n_layers-2
        if l > 1
            if ~use_tanh
                M.biases{l+1} = Ds{l}.hbias;
                M.W{l} = Ds{l}.W;
            else
                M.biases{l+1} = Ds{l}.hbias + sum(Ds{l}.W, 1)';
                M.W{l} = Ds{l}.W / 2;
            end
        else
            M.biases{l+1} = Ds{l}.hbias;
            M.W{l} = Ds{l}.W;
        end
    end
end

fprintf(1, 'Training MLP\n');
tic;
M = mlp (M, X, X_labels, X_valid, X_valid_labels, 0.1);
fprintf(1, 'Training is done after %f seconds\n', toc);

[pred] = mlp_classify (M, X_test);
n_correct = sum(X_test_labels == pred);

fprintf(2, 'Correctly classified test samples: %d/%d\n', n_correct, size(X_test, 1));

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

