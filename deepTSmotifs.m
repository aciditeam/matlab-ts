function [ output_args ] = deepTSmotifs(W, TS, N)
% 1st layer motifs are the "traditionnal notion of motifs
fLayer = W;
% We first get the activations of current layer
% H = dae_get_hidden(TS, fLayer);
% We can look which neuron are reccurently activated
% [~, neuronIDs] = sort(sum(H), 'descend');
% We plot these while sorting them by recurrence
% DEBUG
% For debug purposes let's just look at the 64 first neurons
% DEBUG
neuronIDs = 1:64;
deepTSvisualize(fLayer(:, neuronIDs)');
% 2nd layer can be coarse series prototypes
% TODO 
% TODO
% <= 
% 2. weighted sum of 1st layer
%sLayer = W{2};
%sLayer.W(:, neuronIDs) .* fLayer.W(:, neuronIDs);
% =>
% TODO 
% TODO
% 3rd layer can be finer series prototypes
end

