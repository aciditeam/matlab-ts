function [errorRate, layersErrorRate] = deepClassifyLayerSoftmax(type, model, trainSeries, trainLabels, testSeries, testLabels)
    H_train = trainSeries;
    H_test = testSeries;
    switch type
        case 'SDAE'
            H_train = sdae_get_hidden(H_train, model);
            H_test = sdae_get_hidden(H_test, model);
        case 'DBM'
            H_train = dbm_get_hidden(H_train, model);
            H_test = dbm_get_hidden(H_test, model);
        case 'DBN'
            H_train = dbn_get_hidden(H_train, model);
            H_test = dbn_get_hidden(H_test, model);
        case 'MLP'
            nbLayers = length(model.structure.layers);
            for l = 2:(nbLayers - 1)
                H_train = bsxfun(@plus, H_train * model.W{l-1}, model.biases{l}');
                H_train = sigmoid(H_train, model.hidden.use_tanh);
                H_test = bsxfun(@plus, H_test * model.W{l-1}, model.biases{l}');
                H_test = sigmoid(H_test, model.hidden.use_tanh);
            end
        otherwise
            layersErrorRate = zeros(length(model), 1);
            for l = 1:(length(model) - 2)
                curLayer = model(l);
                switch type
                    case 'RBM'
                        H_train = rbm_get_hidden(H_train, curLayer);
                        H_test = rbm_get_hidden(H_test, curLayer);
                    case {'DAE','MLP'}
                        H_train = dae_get_hidden(H_train, curLayer);
                        H_test = dae_get_hidden(H_test, curLayer);
                end
                % Plug a softmax layer on top of it (if layersError asked)
                if nargout > 1 || (l == (length(model) - 2))
                    softModel = train(trainLabels, sparse(double(H_train)), '-s 0 -q');
                    [L, accuracy, probs] = predict(testLabels, sparse(double(H_test)), softModel, '-b 1');                    
                    layersErrorRate(l) = (100 - accuracy) / 100;
                    errorRate = layersErrorRate(l);
                    fprintf('Error %f at level %d.\n', layersErrorRate(l), l);
                end
            end
            return;
    end
    % Plug a softmax layer on top of it
    softModel = train(trainLabels, sparse(double(H_train)), '-s 0');
    [L, accuracy, probs] = predict(testLabels, sparse(double(H_test)), softModel, '-b 1');
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
    errorRate = (100 - accuracy) / 100;
end
