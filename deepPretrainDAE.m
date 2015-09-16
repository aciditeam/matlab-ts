function [model] = deepPretrainDAE(model, fullTrainSeries)
    fprintf('Pre-training DAE network.\n');
    % Number of layers
    nbLayers = model.nbLayers;
    % Set to use as pre-training
    H = fullTrainSeries;
    % Parse through layers
    for l = 1:nbLayers-2
        % Take back DAE and use default configurations
        curModel = model.pretrain(l);
        % Use of different activation functions 
        if curModel.use_tanh
            curModel.visible.use_tanh = curModel.use_tanh;
            curModel.hidden.use_tanh = curModel.use_tanh;
        else
            % Initialization of dae
            if curModel.data.binary
                mH = mean(H, 1)';
                curModel.vbias = min(max(log(mH./(1 - mH)), -4), 4);
            else
                curModel.vbias = mean(H, 1)';
            end
            % Gaussian initialization
            if curModel.gaussianInit
                nVis = curModel.structure.n_visible;
                nHid = curModel.structure.n_hidden;
                gMeans = repmat(rand(1, nHid) * (nVis), nVis, 1);
                gDevs = repmat(rand(1, nHid) * (nVis), nVis, 1);
                xVect = (repmat((1:nVis)', 1, nHid));
                gaussVect = ((1 ./ (gDevs .* sqrt(2 * pi))) .* exp((-1/2) .* (((xVect - gMeans) ./ gDevs) .^ 2)));
                curModel.W = curModel.W .* gaussVect;
            end
        end
        % Training the DAE of current layer
        fprintf(1, ' * Training DAE layer n.%d\n', l);
        tic; curModel = dae(curModel, H);
        fprintf(1, '    - Done [%f seconds]\n', toc);
        % Get the activations from the current layers (to the next)
        H = dae_get_hidden(H, curModel);
        % Weights found for current layer
        model.pretrain(l) = curModel;
    end
end
