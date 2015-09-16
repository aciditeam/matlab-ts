function [errorRate, errorFeat, errorSoft, model, trainTime, testTime] = deepClassify(dataset, type, model, trainSeries, trainLabels, testSeries, testLabels, doLayerClassif)
    switch type
        case 'DBN'
            [errorRate, model, trainTime, testTime] = deepClassifyDBN(dataset, model, trainSeries, trainLabels, testSeries, testLabels);
        case 'DBM'
            [errorRate, model, trainTime, testTime] = deepClassifyDBM(dataset, model, trainSeries, trainLabels, testSeries, testLabels);
        case 'SDAE'
            [errorRate, model, trainTime, testTime] = deepClassifySDAE(dataset, model, trainSeries, trainLabels, testSeries, testLabels);
        case 'MLP'
            [errorRate, model, trainTime, testTime] = deepClassifyMLP(dataset, model, trainSeries, trainLabels, testSeries, testLabels);
        otherwise
            error(['Unknown model ' type 'for classification.']);
    end
    if (doLayerClassif)
        errorFeat = deepClassifyLayerDistances(type, model.train, trainSeries, trainLabels, testSeries, testLabels);
        errorSoft = 1;%deepClassifyLayerSoftmax(type, model.train, trainSeries, trainLabels, testSeries, testLabels);
    else
        errorFeat = 1; errorSoft = 1;
    end
end
