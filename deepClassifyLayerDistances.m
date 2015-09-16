function [errorRate, layersErrorRate] = deepClassifyLayerDistances(type, Ds, trainSeries, trainLabels, testSeries, testLabels)
    H_train = trainSeries;
    H_test = testSeries;
    switch type
        case 'SDAE'
            H_train = sdae_get_hidden(H_train, Ds);
            H_test = sdae_get_hidden(H_test, Ds);
        case 'DBM'
            H_train = dbm_get_hidden(H_train, Ds);
            H_test = dbm_get_hidden(H_test, Ds);
        case 'DBN'
            H_train = dbn_get_hidden(H_train, Ds);
            H_test = dbn_get_hidden(H_test, Ds);
        otherwise
            layersErrorRate = zeros(length(Ds), 1);
            for l = 1:(length(Ds) - 1)
                curLayer = Ds{l};
                switch type
                    case 'RBM'
                        H_train = rbm_get_hidden(H_train, curLayer);
                        H_test = rbm_get_hidden(H_test, curLayer);
                    case 'DAE'
                        H_train = dae_get_hidden(H_train, curLayer);
                        H_test = dae_get_hidden(H_test, curLayer);
                end
                pred = zeros(size(H_test, 1), 1);
                for s = 1:size(H_test, 1)
                    tmpTrain = repmat(H_test(s, :), size(H_train, 1), 1);
                    distsL2 = sqrt(sum((tmpTrain - H_train) .^ 2, 2));
                    [~, bestID] = min(distsL2);
                    pred(s) = trainLabels(bestID);
                end
                n_correct = sum(pred == testLabels);
                layersErrorRate(l) = (size(testSeries, 1) - n_correct) / size(testSeries, 1);
                fprintf('Error %f at level %d.\n', layersErrorRate(l), l);
            end
            errorRate = layersErrorRate(end);
            return;
    end
    pred = zeros(size(H_test, 1), 1);
    for s = 1:size(H_test, 1)
        tmpTrain = repmat(H_test(s, :), size(H_train, 1), 1);
        distsL2 = sqrt(sum((tmpTrain - H_train) .^ 2, 2));
        [~, bestID] = min(distsL2);
        pred(s) = trainLabels(bestID);
    end
    n_correct = sum(pred == testLabels);
    disp('Layer distances Resulting classes');
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
    errorRate = (size(testSeries, 1) - n_correct) / size(testSeries, 1);
end
