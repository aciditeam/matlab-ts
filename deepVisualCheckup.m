function deepVisualCheckup(type, Ds, trainSeries, trainLabels, testSeries, testLabels, N)
    % First select N random series from training and testing
    uniqueTrain = unique(trainLabels);
    H_train = zeros(N * length(uniqueLabels), size(trainSeries, 2));
    H_test = zeros(N * length(uniqueLabels), size(trainSeries, 2));
    for i = 1:length(uniqueTrain)
        curTrain = trainSeries(trainLabels == i, :);
        curTest = testSeries(testLabels == i, :);
        rndTrain = randi(size(curTrain, 1), N);
        rndTest = randi(size(curTest, 1), N);
        for n = 1:N
            H_train((i-1)*N + n, :) = curTrain(rndTrain(n), :);
            H_test((i-1)*N + n, :) = curTest(rndTest(n), :);
        end
    end
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
            end
    end
    disp(size(H_train));
    disp(size(H_test));
    afsdfg;
end
