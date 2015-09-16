function nextBatch = hyperparameters_fit(optimize, model, curValue, ranks, finalGrid, nbLayers, nbBatch, type)
if nargin < 8
    type = 'structure';
end
switch type
    case 'structure'
        % We estimate the best infered values from Nadaraya-Watson kernel regression
        kernel = ksrmv(curValue(:, 2:(end-1)), ranks, repmat(optimize.structure.units.step * 4, 1, nbLayers-2), finalGrid);
        % Prediction of what would be the best hyperparameters
        [~, bestIDs] = sort(kernel.f, 'ascend');
        % Prepare the next batch
        nextBatch = cell(nbBatch, 1);
        for b = 1:nbBatch
            nextBatch{b} = model;
            curSetup = kernel.x(bestIDs(b), :);
            % Generate a point around the current
            for coords = 1:length(curSetup)
                varUnits = round((rand() * (optimize.structure.units.step * 2)) - optimize.structure.units.step);
                curSetup(coords) = curSetup(coords) + varUnits;
            end
            nextBatch{b}.structure(2:end-1) = curSetup;
        end
    case 'full'
        % We estimate the best infered values from Nadaraya-Watson kernel regression
        kernel = ksrmv(curValue, ranks, ([optimize.pretrainSteps optimize.trainSteps] * 2), finalGrid);
        % Prediction of what would be the best hyperparameters
        [~, bestIDs] = sort(kernel.f, 'ascend');
        % Prepare the next batch
        nextBatch = cell(nbBatch, 1);
        for b = 1:nbBatch
            nextBatch{b} = model;
            curSetup = kernel.x(bestIDs(b), :);
            % Need to update all values
            pNames = optimize.pretrainNames;
            tNames = optimize.trainNames;
            for n = 1:length(pNames)
                for l = 1:(nbLayers-1)
                    eval(['nextBatch{' num2str(b) '}.pretrain(' num2str(l) ').' pNames{n} ' = curSetup(' num2str(n) ');']);
                end
            end
            for n = 1:length(tNames)
                eval(['nextBatch{' num2str(b) '}.train.' tNames{n} ' = curSetup(' num2str(n + length(pNames)) ');']);
            end
        end
    case 'pretrain'
        % We estimate the best infered values from Nadaraya-Watson kernel regression
        kernel = ksrmv(curValue, ranks, (optimize.pretrainSteps * 2), finalGrid);
        % Prediction of what would be the best hyperparameters
        [~, bestIDs] = sort(kernel.f, 'ascend');
        % Prepare the next batch
        nextBatch = cell(nbBatch, 1);
        for b = 1:nbBatch
            nextBatch{b} = model;
            curSetup = kernel.x(bestIDs(b), :);
            % Need to update all values
            pNames = optimize.pretrainNames;
            for n = 1:length(pNames)
                for l = 1:(nbLayers-1)
                    eval(['nextBatch{' num2str(b) '}.pretrain(' num2str(l) ').' pNames{n} ' = curSetup(' num2str(n) ');']);
                end
            end
        end
    case 'train'
        % We estimate the best infered values from Nadaraya-Watson kernel regression
        kernel = ksrmv(curValue, ranks, (optimize.trainSteps * 2), finalGrid);
        % Prediction of what would be the best hyperparameters
        [~, bestIDs] = sort(kernel.f, 'ascend');
        % Prepare the next batch
        nextBatch = cell(nbBatch, 1);
        for b = 1:nbBatch
            nextBatch{b} = model;
            curSetup = kernel.x(bestIDs(b), :);
            % Need to update all values
            tNames = optimize.trainNames;
            for n = 1:length(tNames)
                eval(['nextBatch{' num2str(b) '}.train.' tNames{n} ' = curSetup(' num2str(n) ');']);
            end
        end
    otherwise
        error(['Unknown fitting type ' type ' for hyperparameters.']);
end
end