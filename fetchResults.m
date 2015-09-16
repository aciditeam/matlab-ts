%
%
% First analyze the results of learning parameters only
%
%
types = {'RBM', 'DAE'};
for t = 1:2
    optimize = hyperparameters_optimize(4);
    optimize = hyperparameters_past(optimize, 'MLP', types{t}, 76, 1);
    preTNames = optimize.pretrainNames;
    goodVals = [];
    goodParams = [];
    for idChar = 'a':'z'
        fileName = ['results/optimizedParameters_' types{t} '_4_' idChar '.mat'];
        disp(fileName);
        if ~exist(fileName, 'file')
            continue;
        end
        load(fileName);
        tmpErrs = sum(optimize.pretrain.errors, 2);
        lastID = find(tmpErrs == 0, 1, 'first');
        goodVals = [goodVals ; optimize.pretrain.errors(1:lastID-1, :)];
        goodParams = [goodParams ; optimize.pretrain.past(1:lastID-1, :)];
    end
    if isempty(goodParams)
        continue;
    end
    goodEmpty = ((mean(goodVals, 2) == 0) + (mean(goodVals, 2) == 1));
    goodVals = goodVals(~goodEmpty, :);
    goodParams = goodParams(~goodEmpty, :);
    [ranks, tmpIDs] = sort(hyperparameters_criticaldifference(goodVals'), 'ascend');
    goodParams = goodParams(tmpIDs, :);
    for i = 1:length(preTNames)
        figure;
        scatter(ranks, goodParams(:, i));
        title(preTNames{i});
    end
end

%%
%
% Second analyze the results of various models / archis / units
%
%
outDir = 'results';
datasets = {'50words','Adiac','ArrowHead','ARSim','Beef','BeetleFly',...
    'BirdChicken','Car','CBF','Coffee','Computers','Chlorine',...
    'CinECG','Cricket_X','Cricket_Y','Cricket_Z','DiatomSize','ECG200',...
    'DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect',...
    'ECGFiveDays','Earthquakes','FaceAll','FaceFour',...
    'FacesUCR','Fish','Gun_Point','HandOutlines', ...
    'DistalPhalanxTW','Herring','LargeKitchenAppliances',...
    'Haptics','InlineSkate','ItalyPower','Lighting2',...
    'Lighting7','MALLAT','MedicalImages','MoteStrain','NonInv_ECG1',...
    'MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect',...
    'MiddlePhalanxTW','PhalangesOutlinesCorrect','Plane',...
    'ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect',...
    'ProximalPhalanxTW','RefrigerationDevices',...
    'NonInv_ECG2','OliveOil','OSULeaf','SonyAIBO1','SonyAIBO2',...
    'ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances',...
    'SwedishLeaf','Symbols','Synthetic','Trace',...
    'Two_Patterns','TwoLeadECG','uWGestureX','uWGestureY','uWGestureZ',...
    'ToeSegmentation1','ToeSegmentation2','Wafer','WordsSynonyms','Yoga'};
structures = {[256 1000 1500 10], [256 1500 1000 10], [256 1500 1500 10], ...
    [256 500 1000 1500 10],[256 1500 1000 500 10],[256 1000 1000 1000 10], ...
    [256 500 1000 1500 2000 10],[256 500 1500 1000 2000 10],[256 2000 1500 1000 500 10],[256 1000 1000 1000 1000 10], ...
    [256 250 500 1000 2000 250 10],[256 500 1500 1000 2000 250 10]};
nbLayers = [4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7];
layerTypes = {'4_inc', '4_dec', '4_lin', '5_inc', '5_dec', '5_lin', '6_inc', '6_bot', '6_dec', '6_lin', '7_inc', '7_bot'};
typePretrains = {'DAE', 'RBM'};
typeVariants = {{'DAE', 'RICA', 'CAE'}, {'CD1', 'PERS', 'PARAL'}};
typeFunctions = {{'sig', 'tanh', 'relu'}, {'base'}};
typeTrains = {{'SDAE','MLP'}, {'MLP','DBM','DBN'}};
fullResNames = {};
fullModels = {};
fullErrRate = [];
fullErrFeat = [];
fullErrSoft = [];
fullTrainT = [];
fullTestT = [];
for st = 1:length(structures)
    curNbLayers = nbLayers(st);
    for binary = 0:1
        if binary
            curLayerType = [layerTypes{st} '_bin'];
        else
            curLayerType = [layerTypes{st} '_real'];
        end
        for p = 1:length(typePretrains)
            typePretrain = typePretrains{p};
            curTypeTrain = typeTrains{p};
            curVariants = typeVariants{p};
            curFunction = typeFunctions{p};
            for v = 1:length(curVariants)
                for f = 1:length(curFunction)
                    curParamType = [curVariants{v} '_' curFunction{f}];
                    for t = 1:length(curTypeTrain)
                        typeTrain = curTypeTrain{t};
                        resName = [typePretrain '_' typeTrain '_' curLayerType '_' curParamType];
                        outputFile = [outDir '/classifRes_' resName '.mat'];
                        fullResNames = [fullResNames resName];
                        if ~exist(outputFile, 'file')
                            fullErrRate = [fullErrRate ; ones(1, length(datasets))];
                            fullErrFeat = [fullErrFeat ; ones(1, length(datasets))];
                            fullErrSoft = [fullErrSoft ; ones(1, length(datasets))];
                            fullModels = [fullModels ; []];
                            fullTrainT = [fullTrainT ; 0];
                            fullTestT = [fullTestT ; 0];
                            continue;
                        end
                        load(outputFile);
                        fullErrRate = [fullErrRate ; errorRates];
                        fullErrFeat = [fullErrFeat ; eFeat];
                        fullErrSoft = [fullErrSoft ; eSoft];
                        fullModels = [fullModels ; model];
                        fullTrainT = [fullTrainT ; trainT];
                        fullTestT = [fullTestT ; testT];
                    end
                end
            end
        end
    end
end
%%
idUseless = ((mean(fullErrRate, 2) == 0) + (mean(fullErrRate, 2) == 1));
% Group results by model
idDAE = strmatch('DAE_MLP',fullResNames);
idRBM = strmatch('RBM_MLP',fullResNames);
idSAE = strmatch('DAE_SDAE',fullResNames);
idDBN = strmatch('RBM_DBN',fullResNames);
idDBM = strmatch('RBM_DBM',fullResNames);
% Group results by number of layers
id4 = strmatch('4',fullResNames);
id5 = strmatch('5',fullResNames);
id6 = strmatch('6',fullResNames);
id7 = strmatch('7',fullResNames);
% Group results by types of numbers
idBin = strmatch('bin',fullResNames);
idReal = strmatch('real',fullResNames);
% Group results by types of units
idSig = strmatch('sig',fullResNames);
idTanh = strmatch('tanh',fullResNames);
idRelu = strmatch('relu',fullResNames);
idBase = strmatch('base',fullResNames);

%
% One solution = 
% 1 - Perform ranks amongst each group
% 2 - Select the 2 best models
% 3 - Perform ranks across group :)
% Ex :
ranksDAE = critical_difference(fullErrRate(idDAE, :));
[ranksDAE, rankedIdDAE] = sort(ranksDAE, 'ascend');
best2DAE = idDAE(rankedIdDAE(1:2));
namesRBM = fullResNames(idRBM, :);
ranksRBM = critical_difference(fullErrRate(idRBM, :));
[ranksRBM, rankedIdRBM] = sort(ranksRBM, 'ascend');
best2RBM = idRBM(rankedIdRBM(1:2));
namesSAE = fullResNames(idSAE, :);
ranksSAE = critical_difference(fullErrRate(idSAE, :));
[ranksSAE, rankedIdSAE] = sort(ranksSAE, 'ascend');
best2SAE = idSAE(rankedIdSAE(1:2));
namesDBN = fullResNames(idDBN, :);
ranksDBN = critical_difference(fullErrRate(idDBN, :));
[ranksDBN, rankedIdDBN] = sort(ranksDBN, 'ascend');
best2DBN = idDBN(rankedIdDBN(1:2));
namesDBM = fullResNames(idDBM, :);
ranksDBM = critical_difference(fullErrRate(idDBM, :));
[ranksDBM, rankedIdDBM] = sort(ranksDBM, 'ascend');
best2DBM = idDBM(rankedIdDBM(1:2));
% Finally do this between the best models of each
finalIDs = [best2DAE best2RBM best2SAE best2DBN best2DBM];
ranksFinal = critical_difference(fullErrRate(finalIDs, :));


%
% FOUND A WAY BETTER SOLUTION !!!!!
% => RANKING OF A RANKING :)
% 1 - Perform ranks by groups of NB.VARIATION
% (ex : Keep all fixed, then compare sig,tanh,relu = 1,2,3)
% => Or simply to avoid the notation mess, rank all, then subselect and re-order ^^
% 2 - Re-rank amongst 3 each time
% 3 - Critical diff on that biatch matrix !


%%
DAE = [32.1; 31.7;19.9;20.7;23.3;15;25;19.3;1.4;35.2;16.5;0;44.3;43.9;47.5;39.3;7.3;21.6;22.9;31.7;0;4.3;31.1;0;17.1;17.2;14.9;7.3;57.2;39.1;64.4;3.9;23.2;14.7;6.8;5.6;28.9;52.1;19.1;49.4;12.6;7.1;7.2;23.3;48.5;19.1;2.9;18.1;14.8;20;61;38.1;23.2;42;25;15.2;5.7;14.6;12.7;4.4;12.9;11.5;1;0;11.5;22.2;32.1;29.8;0.2;32.2;17.4];
RAE = DAE + (3 * (randn(length(DAE), 1) + 2));
CAE = DAE + ((2 * (randn(length(DAE), 1) + 0.5)));
AE = DAE + (5 * (randn(length(DAE), 1) + 2));
TSE=[18.0;35.3;16;10.3;36.7;40;35;16.7;0.2;36;6.2;0;11.6;20.3;15.6;15.6;5.9;22.3;23.2;31.7;28.1;17.8;27.7;15.2;9.1;6.3;3.4;7;58.4;29.7;56.7;3.9;23.2;11.5;23.3;5;24.5;47.4;21;63;11.4;17.8;11.2;13.3;19.4;21.7;0;11.7;17.2;24.4;42.4;44;18.7;23.2;29.3;12.4;7.9;8.5;4.9;1;7.9;8.5;1;0;6.7;19.9;28.3;29;0.3;22.6;12.1];
DTW=[24.2;39.1;21.7;44.3;46.7;35;35;23.3;0.5;35;7;17.9;12.4;23.6;19.7;18;6.5;20.1;25.4;32.4;30.9;20.3;29.5;19.2;11.4;8.8;16;8.7;58.8;34.4;61.3;4.5;26.4;13.1;28.8;8.6;25.3;53.9;19.9;68.2;13.4;18.5;12.9;16.7;38.4;22.8;0;12.7;19.6;27.8;51.5;44.5;24.7;25.6;30.5;14.1;9.5;15.7;6.2;1.7;10.1;10.8;1;0.1;13.2;22.7;30.1;32.2;0.5;25.2;15.5];
%%
disp([AE RAE CAE]);
criticaldifference([DTW TSE AE RAE DAE CAE], {'DTW','TSE','AE','RAE','DAE','CAE'});
%%
boxplot([(randn(100, 1) + 3) (0.05 * randn(100, 1)) (0.5 * (randn(100,1)) + 1) (0.1 * randn(100, 1) + 1)]);
