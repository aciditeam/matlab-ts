function [model, optimize] = hyperparameters_init(optimize, nLayers, nData, nClasses, pretrainType, trainType, tmpOut, initType, trial, layers, binaryLayers, initStruct)
% First initialize architecture
model.nbLayers = nLayers;
model.structure = [nData zeros(1, nLayers - 2) nClasses];
model.binary = [0 ones(1, nLayers - 1)];
for l = 2:(nLayers - 1)
    switch initType
        case 'random'
            if nargin < 10
                model.structure(l) = optimize.units(randi(length(optimize.units)));
                model.binary(l) = randi(2) - 1;
            else
                model.structure(l) = layers(l);
                model.binary(l) = randi(2) - 1;
            end
        case 'default'
           model.structure(l) = layers(l);
           model.binary(l) = binaryLayers(l);
        case 'user'
           model.structure(l) = layers(l);
           model.binary(l) = binaryLayers(l);
    end
end
model.id = ['mod_' regexprep(num2str(model.structure), ' ', '_')];
model.outFiles = [tmpOut '_' model.id];
model.trainType = trainType;
model.pretrainType = pretrainType;
% Then initialize pre-training layers
switch pretrainType
    case 'DAE'
        for l = nLayers-1:-1:1
            model.pretrain(l) = default_dae(model.structure(l), model.structure(l + 1));
            model.pretrain(l).hook.per_epoch = {@save_intermediate, {[model.outFiles '_dae_' num2str(l) '.mat']}};
            model.pretrain(l).data.binary = model.binary(l);
            model.pretrain(l).hidden.binary = model.binary(l+1);
        end
    case 'RBM'
        for l = 1:1:nLayers-1
            model.pretrain(l) = default_rbm(model.structure(l), model.structure(l + 1)); 
            if (strcmp(trainType, 'DBM') && (mod(l, 2) ~= 0) && l+2 <= nLayers)
                model.pretrain(l) = default_rbm(model.structure(l), model.structure(l + 2)); 
            end
            model.pretrain(l).hook.per_epoch = {@save_intermediate, {[model.outFiles '_rbm_' num2str(l) '.mat']}};
            model.pretrain(l).data.binary = model.binary(l);
        end
    otherwise
        error('Unrecognized model');
end
switch trainType
    case 'MLP'
        for l = 1:nLayers-2
            model.train = default_mlp(model.structure);
            model.train.hook.per_epoch = {@save_intermediate, {[model.outFiles '_mlp']}};
        end
    case 'DBN'
        for l = 1:nLayers-2
            model.train = default_dbn(model.structure(1:(end-1)));
            model.train.hook.per_epoch = {@save_intermediate, {[model.outFiles '_dbn']}};
        end
    case 'SDAE'
        for l = 1:nLayers-2
            model.train = default_sdae(model.structure(1:(end-1)));
            model.train.hook.per_epoch = {@save_intermediate, {[model.outFiles '_sdae']}};
        end
    case 'DBM'
        for l = 1:nLayers-2
            model.train = default_dbm(model.structure);
            model.train.hook.per_epoch = {@save_intermediate, {[model.outFiles '_dbm']}};
        end
end
% Set model binary values
model.train.data.binary = model.binary(1);
% Number of iterations in pre-training
for l = 1:nLayers
    model.pretrain(l).iteration.n_epochs = optimize.pretrain.iteration.n_epochs;
end
model.train.iteration.n_epochs = optimize.train.iteration.n_epochs;
switch initType
    case 'random'
        for l = 1:nLayers
            for i = 1:length(optimize.pretrain.names)
                if (optimize.pretrain.continuous(i))
                    value = rand * (optimize.pretrain.values{i}(end) - optimize.pretrain.values{i}(1));
                    value = value + optimize.pretrain.values{i}(1);
                else
                    value = optimize.pretrain.values{i}(randi(length(optimize.pretrain.values{i})));
                end
                eval(['model.pretrain(l).' (optimize.pretrain.names{i}) ' = ' num2str(value) ';']);
            end
            for i = 1:length(optimize.pretrain.(pretrainType).names)
                if (optimize.pretrain.(pretrainType).continuous(i))
                    value = rand * (optimize.pretrain.(pretrainType).values{i}(end) - optimize.pretrain.(pretrainType).values{i}(1));
                    value = value + optimize.pretrain.(pretrainType).values{i}(1);
                else
                    value = optimize.pretrain.(pretrainType).values{i}(randi(length(optimize.pretrain.(pretrainType).values{i})));
                end
                eval(['model.pretrain(l).' (optimize.pretrain.(pretrainType).names{i}) ' = ' num2str(value) ';']);
            end
        end
        for i = 1:length(optimize.train.names)
            if (optimize.train.continuous(i))
                value = rand * (optimize.train.values{i}(end) - optimize.train.values{i}(1));
                value = value + optimize.train.values{i}(1);
            else
                value = optimize.train.values{i}(randi(length(optimize.train.values{i})));
            end
            eval(['model.train.' (optimize.train.names{i}) ' = ' num2str(value) ';']);
        end
        for i = 1:length(optimize.train.(trainType).names)
            if (optimize.train.(trainType).continuous(i))
                value = rand * (optimize.train.(trainType).values{i}(end) - optimize.train.(trainType).values{i}(1));
                value = value + optimize.train.(trainType).values{i}(1);
            else
                value = optimize.train.(trainType).values{i}(randi(length(optimize.train.(trainType).values{i})));
            end
            eval(['model.train.' (optimize.train.(trainType).names{i}) ' = ' num2str(value) ';']);
        end
    case 'default'
        for l = 1:nLayers
            for i = 1:length(optimize.pretrain.names)
                value = optimize.pretrain.default(i);
                eval(['model.pretrain(l).' (optimize.pretrain.names{i}) ' = ' num2str(value) ';']);
            end
            for i = 1:length(optimize.pretrain.(pretrainType).names)
                value = optimize.pretrain.(pretrainType).default(i);
                eval(['model.pretrain(l).' (optimize.pretrain.(pretrainType).names{i}) ' = ' num2str(value) ';']);
            end
        end
        for i = 1:length(optimize.train.names)
            value = optimize.train.default(i);
            eval(['model.train.' (optimize.train.names{i}) ' = ' num2str(value) ';']);
        end
        for i = 1:length(optimize.train.(trainType).names)
            value = optimize.train.(trainType).default(i);
            eval(['model.train.' (optimize.train.(trainType).names{i}) ' = ' num2str(value) ';']);
        end
    case 'user'
        for l = 1:nLayers
            for i = 1:length(optimize.pretrain.names)
                value = optimize.pretrain.default(i);
                eval(['model.pretrain(l).' (optimize.pretrain.names{i}) ' = ' num2str(value) ';']);
            end
            for i = 1:length(optimize.pretrain.(pretrainType).names)
                value = optimize.pretrain.(pretrainType).default(i);
                eval(['model.pretrain(l).' (optimize.pretrain.(pretrainType).names{i}) ' = ' num2str(value) ';']);
            end
            userNames = initStruct.pretrain.names;
            userValues = initStruct.pretrain.values;
            for i = 1:length(userNames)
                eval(['model.pretrain(l).' (userNames{i}) ' = ' num2str(userValues(i)) ';']);
            end
        end
        for i = 1:length(optimize.train.names)
            value = optimize.train.default(i);
            eval(['model.train.' (optimize.train.names{i}) ' = ' num2str(value) ';']);
        end
        for i = 1:length(optimize.train.(trainType).names)
            value = optimize.train.(trainType).default(i);
            eval(['model.train.' (optimize.train.(trainType).names{i}) ' = ' num2str(value) ';']);
        end
        userNames = initStruct.train.names;
        userValues = initStruct.train.values;
        for i = 1:length(userNames)
            eval(['model.train.' (userNames{i}) ' = ' num2str(userValues(i)) ';']);
        end
end
end