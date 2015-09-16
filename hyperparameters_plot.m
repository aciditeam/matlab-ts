function hyperparameters_plot()
% --------------
%
% Checking the random learner
%
% ---------------
% Extract current error rates and architectures
disp(optimize.structure.units.past(1:60, :));
nbLayers = 5;
for steps = 1:nbSteps
    curNetwork = (steps*10) + 1;
    curError = optimize.structure.units.errors(1:(curNetwork-1), :);
    curValue = optimize.structure.units.past(1:(curNetwork-1), :);
    disp(curValue);
    if sum(curError(end, :)) == 0
        break;
    end
	% Rank different architecture against each other
	ranks = hyperparameters_criticaldifference(curError');
	% Construct an evaluation grid
	cellVals = cell(nbLayers - 2, 1);
	gridStr = ''; gridOut = '';
	for i = 2:nbLayers-1
        gridStr = strcat(gridStr, '32:32:2048,');
        gridOut = strcat(gridOut, ['cellVals{' num2str(i - 1) '},']);
    end
	eval([ '[' gridOut(1:end-1) '] = ndgrid(' gridStr(1:end-1) ');' ]);
	finalGrid = [];
	for i = 2:nbLayers-1
        finalGrid = [finalGrid cellVals{i-1}(:)];
    end
	% We estimate the best infered values from Nadaraya-Watson kernel regression
	kernel = ksrmv(curValue(:, 2:end-1), ranks, repmat(64, 1, nbLayers-2), finalGrid);
    disp(kernel);
%     % First plot the infered surface
%     figure; hold on;
%     X = reshape(kernel.x(:, 1), sqrt(length(kernel.f)), sqrt(length(kernel.f)));
%     Y = reshape(kernel.x(:, 2), sqrt(length(kernel.f)), sqrt(length(kernel.f)));
%     Z = reshape(kernel.f, sqrt(length(kernel.f)), sqrt(length(kernel.f)));
%     surf(X, Y, Z);
%     % Then plot the previous computation points
%     scatter3(curValue(:, 2), curValue(:, 3), ranks, repmat(100, length(ranks), 1), repmat([0.1 1.0 0.1], length(ranks), 1));
% 	% Prediction of what would be the best hyperparameters
	[~, bestIDs] = sort(kernel.f, 'ascend');
	topCoords = zeros(nbBatch, 3);
    topValues = zeros(nbBatch, 1);
	for b = 1:nbBatch
        topCoords(b, :) = kernel.x(bestIDs(b), :);
        topValues(b) = kernel.f(bestIDs(b));
    end
    disp(topCoords(1,:));
    % Then plot the infered best points
%    scatter3(topCoords(:, 1), topCoords(:, 2), topValues(:), repmat(100, length(topValues), 1), repmat([1.0 0.1 0.1], length(topValues), 1));
end
end