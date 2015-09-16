addpath(genpath('.'));
% Results directory
mainDirectory = 'results';
fileChars = {'a','b','c'};
% Datasets used
for nbLayers = 3:6
    fullError = [];
    fullValue = [];
    for fileIDchar = 1:3
        curFile = char(['optimizedStructure_' num2str(nbLayers) '_' char(fileChars{fileIDchar}) '_layers.mat']);
        load(curFile);
        [~, lastNon] = find(sum(curError, 2) == 0, 1, 'first');
        if ~isempty(lastNon)
            curError = curError(1:lastNon-1, :);
            curValue = curValue(1:lastNon-1, :);
        end
        fullError = [fullError ; curError];
        fullValue = [fullValue ; curValue];
    end
    % Rank different architecture against each other
    ranks = hyperparameters_criticaldifference(fullError');
    [ranks, rankID] = sort(ranks, 'ascend');
    bestVals = fullValue(rankID(1:500), :);
    [dummy I] = unique(bestVals, 'rows');
    disp(bestVals(sort(I), :));
    disp(mean(fullError(rankID(1), :)));
end