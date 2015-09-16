function grid = hyperparameters_grid(model, optimize, type, nbTrials)
grid = [];
switch type
    case 'structure'
        % Construct an evaluation grid
        cellVals = cell(model.nbLayers - 2, 1);
        gridStr = ''; gridOut = '';
        for i = 2:model.nbLayers-1
            minVal = num2str(optimize.structure.units.values(1));
            maxVal = num2str(optimize.structure.units.values(end));
            step = num2str(optimize.structure.units.step);
            gridStr = strcat(gridStr, [minVal ':' step ':' maxVal ',']);
            gridOut = strcat(gridOut, ['cellVals{' num2str(i - 1) '},']);
        end
        eval([ '[' gridOut(1:end-1) '] = ndgrid(' gridStr(1:end-1) ');' ]);
        for i = 2:model.nbLayers-1
            grid = [grid cellVals{i-1}(:)];
        end
    case 'full'
        names = [optimize.pretrainNames optimize.trainNames];
        grid = zeros(nbTrials, length(names));
        % Fill the grid with a wide set of random configurations
        for i = 1:length(optimize.pretrainNames)
            if (optimize.pretrainContinuous(i))
                grid(:, i) = rand(nbTrials, 1) * (optimize.pretrainValues{i}(end) - optimize.pretrainValues{i}(1));
                grid(:, i) = grid(:, i) + optimize.pretrainValues{i}(1);
            else
                grid(:, i) = optimize.pretrainValues{i}(randi(length(optimize.pretrainValues{i})));
            end
        end
        for i = 1:length(optimize.trainNames)
            if (optimize.trainContinuous(i))
                grid(:, i + length(optimize.pretrainNames)) = rand(nbTrials, 1) * (optimize.trainValues{i}(end) - optimize.trainValues{i}(1));
                grid(:, i + length(optimize.pretrainNames)) = grid(:, i + length(optimize.pretrainNames)) + optimize.trainValues{i}(1);
            else
                grid(:, i + length(optimize.pretrainNames)) = optimize.trainValues{i}(randi(length(optimize.trainValues{i})));
            end
        end
    case 'pretrain'
        names = optimize.pretrainNames;
        grid = zeros(nbTrials, length(names));
        % Fill the grid with a wide set of random configurations
        for i = 1:length(optimize.pretrainNames)
            if (optimize.pretrainContinuous(i))
                grid(:, i) = rand(nbTrials, 1) * (optimize.pretrainValues{i}(end) - optimize.pretrainValues{i}(1));
                grid(:, i) = grid(:, i) + optimize.pretrainValues{i}(1);
            else
                grid(:, i) = optimize.pretrainValues{i}(randi(length(optimize.pretrainValues{i})));
            end
        end
    case 'train'
        names = optimize.trainNames;
        grid = zeros(nbTrials, length(names));
        % Fill the grid with a wide set of random configurations
        for i = 1:length(optimize.trainNames)
            if (optimize.trainContinuous(i))
                grid(:, i) = rand(nbTrials, 1) * (optimize.trainValues{i}(end) - optimize.trainValues{i}(1));
                grid(:, i) = grid(:, i) + optimize.trainValues{i}(1);
            else
                grid(:, i) = optimize.trainValues{i}(randi(length(optimize.trainValues{i})));
            end
        end
    otherwise
        error(['Unknown grid type ' type ' for hyperparameters.']);
end
end