function [model] = deepPretrainRBM(model, fullTrainSeries)
    fprintf('Pre-training RBM network.\n');
    % Pre-trained layers
    nbLayers = model.nbLayers;
    % Set to use as pre-training
    H = fullTrainSeries;
    for l = 1:(nbLayers - 1)
        curModel = model.pretrain(l);
        if (strcmp(model.trainType, 'DBM'))        
            if mod(l, 2) == 0
                continue;
            end
            if l+2 > nbLayers
                break;
            end
            % Set different learning rates
            curModel.fast.lrate = curModel.learning.lrate;
            curModel.adaptive_lrate.lrate_ub = curModel.learning.lrate;
            % Initialize weights
            if model.binary(l)
                mH = mean(H, 1)';
                curModel.vbias = min(max(log(mH./(1 - mH)), -4), 4);
                curModel.fast.vbias = min(max(log(mH./(1 - mH)), -4), 4);
            else
                curModel.vbias = mean(H, 1)';
                curModel.fast.vbias = mean(H, 1)';
            end
            % train RBM
            fprintf(1, 'Training RBM\n');
            tic; curModel = train_rbm(curModel, H);
            fprintf(1, 'Training is done after %f seconds\n', toc);
            model.pretrain(l) = curModel;
            H = rbm_get_hidden(H, curModel);
            continue;
        end
        % Pre-initialize weights
        mH = mean(H, 1)';
        curModel.vbias = min(max(log(mH./(1 - mH)), -4), 4);
        curModel.hbias = zeros(size(curModel.hbias));
        curModel.W = 0.01 * (randn(model.structure(l), model.structure(l+1)));
        if model.binary(l)
            mH = mean(H, 1)';
            curModel.vbias = min(max(log(mH./(1 - mH)), -4), 4);
            curModel.fast.vbias = min(max(log(mH./(1 - mH)), -4), 4);
        else
            curModel.vbias = mean(H, 1)';
            curModel.fast.vbias = mean(H, 1)';
        end
        % train RBM
        fprintf(1, 'Training RBM\n');
        tic; curModel = train_rbm (curModel, H);
        fprintf(1, 'Training is done after %f seconds\n', toc);
        % Retrieve the pre-trained weights
        model.pretrain(l) = curModel;
        % Fetch activation of RBM
        H = rbm_get_hidden(H, curModel);
    end
end
