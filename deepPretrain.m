function [model] = deepPretrain(model, type, fullTrainSeries)
    switch type
        case 'RBM'
            model = deepPretrainRBM(model, fullTrainSeries);
        case 'DAE'
            model = deepPretrainDAE(model, fullTrainSeries);
        otherwise
            error(['Unrecognized model ' type ' for pretraining.']);
    end
end
