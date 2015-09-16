function [model, optimize] = hyperparameters_random(optimize, nbLayers, nData, nClasses)
% First initialize architecture
model.nbLayers = nbLayers;
model.structure = [nData zeros(1, nbLayers - 2) nClasses];
model.binary = [0 ones(1, nbLayers - 1)];
for l = 2:(nbLayers - 1)
    model.structure(l) = rand * (optimize.structure.units.values(2) - optimize.structure.units.values(1));
    model.structure(l) = floor(model.structure(l) + optimize.structure.units.values(1));
    model.binary(l) = randi(2) - 1;
end
model.trainType = 'RND';
% Model will be a fake SDAE (basic sigmoid hidden network)
model.train = default_sdae(model.structure(1:(end-1)));
model.train.data.binary = model.binary(1);
end