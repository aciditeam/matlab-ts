function [model, optimize] = hyperparameters_output(fID, optimize, model, pretrainType, trainType)
fprintf(fID, 'Parameters :\n');
for l = 1:1
    for i = 1:length(optimize.pretrain.names)
       eval(['fprintf(fID, ''%.32s \t = %f\n'', ''' (optimize.pretrain.names{i}) ''', model.pretrain(l).' (optimize.pretrain.names{i}) ');']);
    end
    for i = 1:length(optimize.pretrain.(pretrainType).names)
       eval(['fprintf(fID, ''%.32s \t = %f\n'', ''' (optimize.pretrain.(pretrainType).names{i}) ''', model.pretrain(l).' (optimize.pretrain.(pretrainType).names{i}) ');']);
    end
end