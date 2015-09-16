% %----------------------------
% % Architecture search jobs
% %----------------------------
% fileIDchars = {'a', 'b', 'c'};
% fullSH = fopen('deepStructures.sh', 'w');
% fprintf(fullSH, '#!/bin/sh\n');
% for nbLayers = 3:7
%     for i = 1:length(fileIDchars)
%         fprintf(fullSH, 'sbatch deepStructures_%d_%s.sh\n', nbLayers, fileIDchars{i});
%         confID = fopen(['deepStructures_' num2str(nbLayers) '_' fileIDchars{i} '.sh'], 'w');
%         fprintf(confID, '#!/bin/sh\n');
%         fprintf(confID, 'source /etc/profile.modules\n\n');
%         fprintf(confID, '#SBATCH -J DeepStruct-%d-%s\n', nbLayers, fileIDchars{i});
%         fprintf(confID, '#SBATCH -e DeepStruct-%d-%s.err.txt\n', nbLayers, fileIDchars{i});
%         fprintf(confID, '#SBATCH -o DeepStruct-%d-%s.out.txt\n', nbLayers, fileIDchars{i});
%         fprintf(confID, '#SBATCH --licenses=statistics_toolbox@matlablm.unige.ch\n');
%         fprintf(confID, '#SBATCH -c 16\n');
%         fprintf(confID, '#SBATCH -n 1\n');
%         fprintf(confID, '#SBATCH -p parallel\n');
%         fprintf(confID, '#SBATCH -t 4-0:00\n');
%         fprintf(confID, '\n');
%         fprintf(confID, 'unset DISPLAY\n');
%         fprintf(confID, 'module load matlab\n');
%         fprintf(confID, 'srun matlab -nodesktop -nosplash -nodisplay -r "');
%         fprintf(confID, 'cd ~/deepLearn;');
%         fprintf(confID, 'nbLayers=%d;', nbLayers);
%         fprintf(confID, 'fileIDchar=''%s'';', fileIDchars{i});
%         fprintf(confID, 'deepStructure"\n');
%     end
% end
% fclose(fullSH);
% 
% %----------------------------
% % Hyperparameters search jobs
% %----------------------------
% fileIDchars = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'};
% typePretrain = {'DAE', 'RBM'};
% typeTrain = {'SDAE', 'MLP', 'DBN', 'DBM'};
% fullSH = fopen('deepJobs.sh', 'w');
% fprintf(fullSH, '#!/bin/sh\n');
% for i = 1:length(typePretrain)
%     for nbLayers = 4:7
%         for fileC = 1:5
%             fprintf(fullSH, 'sbatch deepJobs_%s_%d_%s.sh\n', typePretrain{i}, nbLayers, fileIDchars{fileC});
%             confID = fopen(['deepJobs_' typePretrain{i} '_' num2str(nbLayers) '_' fileIDchars{fileC} '.sh'], 'w');
%             fprintf(confID, '#!/bin/sh\n');
%             fprintf(confID, 'source /etc/profile.modules\n\n');
%             fprintf(confID, '#SBATCH -J Deep-%s-%d\n', typePretrain{i}, nbLayers);
%             fprintf(confID, '#SBATCH -e Deep-%s-%d-%s.err.txt\n', typePretrain{i}, nbLayers, fileIDchars{fileC});
%             fprintf(confID, '#SBATCH -o Deep-%s-%d-%s.out.txt\n', typePretrain{i}, nbLayers, fileIDchars{fileC});
%             fprintf(confID, '#SBATCH --licenses=statistics_toolbox@matlablm.unige.ch\n');
%             %fprintf(confID, '#SBATCH -c 16\n');
%             fprintf(confID, '#SBATCH --ntasks=1\n');
%             fprintf(confID, '#SBATCH --partition=mono-shared\n');
%             fprintf(confID, '#SBATCH --time=4-0:00\n');
%             fprintf(confID, '\n');
%             fprintf(confID, 'unset DISPLAY\n');
%             fprintf(confID, 'module load matlab\n');
%             fprintf(confID, 'srun matlab -nodesktop -nosplash -nodisplay -r "');
%             fprintf(confID, 'cd ~/deepLearn;');
%             fprintf(confID, 'nbLayers=%d;', nbLayers);
%             fprintf(confID, 'fileIDchar=''%s'';', fileIDchars{fileC});
%             fprintf(confID, 'typePretrain=''%s'';', typePretrain{i});
%             fprintf(confID, 'typeTrain=''%s'';', 'MLP');
%             fprintf(confID, 'deepOptimizePretrain"\n');
%         end
%     end
% end
% % for i = 1:length(typePretrain)
% %     for j = 1:length(typeTrain)
% %         for nbLayers = 4:7
% %             for fileC = 1:3
% %             fprintf(fullSH, 'sbatch deepJobs_%s_%s_%d_%s.sh\n', typePretrain{i}, typeTrain{j}, nbLayers, fileIDchars{i});
% %             confID = fopen(['deepJobs_' typePretrain{i} '_' typeTrain{j} '_' num2str(nbLayers) '_' fileIDchars{fileC} '.sh'], 'w');
% %             fprintf(confID, '#!/bin/sh\n');
% %             fprintf(confID, 'source /etc/profile.modules\n\n');
% %             fprintf(confID, '#SBATCH -J Deep-%s-%s-%d\n', typePretrain{i}, typeTrain{j}, nbLayers);
% %             fprintf(confID, '#SBATCH -e HV-%s-%s-%d.err.txt\n', typePretrain{i}, typeTrain{j}, nbLayers);
% %             fprintf(confID, '#SBATCH -o HV-%s-%s-%d.out.txt\n', typePretrain{i}, typeTrain{j}, nbLayers);
% %             fprintf(confID, '#SBATCH --licenses=statistics_toolbox@matlablm.unige.ch\n');
% %             fprintf(confID, '#SBATCH -c 16\n');
% %             fprintf(confID, '#SBATCH -n 1\n');
% %             fprintf(confID, '#SBATCH -p parallel\n');
% %             fprintf(confID, '#SBATCH -t 4-0:00\n');
% %             fprintf(confID, '\n');
% %             fprintf(confID, 'unset DISPLAY\n');
% %             fprintf(confID, 'module load matlab\n');
% %             fprintf(confID, 'srun matlab -nodesktop -nosplash -nodisplay -r "');
% %             fprintf(confID, 'cd ~/deepLearn;');
% %             fprintf(confID, 'nbLayers=%d;', nbLayers);
% %             fprintf(confID, 'fileIDchar=''%s'';', fileIDchars{fileC});
% %             fprintf(confID, 'typePretrain=''%s'';', typePretrain{i});
% %             fprintf(confID, 'typeTrain=''%s'';', typeTrain{j});
% %             fprintf(confID, 'deepOptimizeAll"\n');
% %             end
% %         end
% %     end
% % end
% fclose(fullSH);

rF = 128;
structures = {[128 1000 1500 10], [128 1500 1000 10], [128 1500 1500 10], ...
    [128 500 1000 1500 10],[128 1500 1000 500 10],[128 1000 1000 1000 10], ...
    [128 500 1000 1500 2000 10],[128 500 1500 1000 2000 10],[128 2000 1500 1000 500 10],[128 1000 1000 1000 1000 10], ...
    [128 250 500 1000 2000 250 10],[128 500 1500 1000 2000 250 10]};
nbLayers = [4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7];
layerTypes = {'4_inc', '4_dec', '4_lin', '5_inc', '5_dec', '5_lin', '6_inc', '6_bot', '6_dec', '6_lin', '7_inc', '7_bot'};
typePretrains = {'DAE', 'RBM'};
typeVariants = {{'DAE', 'RICA', 'CAE'}, {'CD1', 'PERS', 'PARAL'}};
variantsParamsPretrain = {'''iteration.n_epochs'', ''learning.lrate'', ''use_tanh'', ''noise.drop'', ''noise.level'', ''rica.cost'', ''cae.cost''',...
    '''iteration.n_epochs'', ''learning.lrate'', ''learning.cd_k'', ''learning.persistent_cd'', ''parallel_tempering.use'''};
curPreVals = {{{'200 1e-3 0 0.1 0.1 0 0','200 1e-3 1 0.1 0.1 0 0', '200 1e-3 2 0.1 0.1 0 0'},... %DAE_DAE
    {'200 1e-3 0 0 0 0.1 0','200 1e-3 1 0 0 0.1 0', '200 1e-3 2 0 0 0.1 0'},... %DAE_RICA
    {'200 1e-3 0 0 0 0.01 0','200 1e-3 1 0 0 0.01 0', '200 1e-3 2 0 0 0.01 0'}},... %DAE_CAE
    {{'200 1e-3 1 0 0'},... %RBM_CD1
    {'200 1e-3 1 1 0'},... %RBM_PERS
    {'200 1e-3 1 0 1'}}}; %RBM_PARAL
typeFunctions = {{'sig', 'tanh', 'relu'}, {'base'}};
typeTrains = {{'SDAE','MLP'}, {'MLP','DBM','DBN'}};
variantsParamsTrain = {{'''iteration.n_epochs'', ''use_tanh''','''iteration.n_epochs'', ''use_tanh'''}...
    {'''iteration.n_epochs''', '''iteration.n_epochs'', ''learning.persistent_cd''', '''iteration.n_epochs'', ''learning.persistent_cd'''}};
curTrainVals = {{{{'200 0','200 0'},{'200 1','200 1'},{'200 2','200 2'}},... %DAE_DAE
    {{'200 0','200 0'},{'200 1','200 1'},{'200 2','200 2'}},... %DAE_RICA
    {{'200 0','200 0'},{'200 1','200 1'},{'200 2','200 2'}}},... %DAE_CAE
    {{{'200 0', '200 1', '200 0'},{'200 0', '200 1', '200 0'},{'200 0', '200 1', '200 0'}},... %RBM_CD1
    {{'200 0', '200 1', '200 0'},{'200 0', '200 1', '200 0'},{'200 0', '200 1', '200 0'}},... %RBM_PERS
    {{'200 0', '200 1', '200 0'},{'200 0', '200 1', '200 0'},{'200 0', '200 1', '200 0'}}}};
binaryLayers = [0 1];
fullSH = fopen('deepJobs.sh', 'w');
fprintf(fullSH, '#!/bin/sh\n');
for st = 1:length(structures)
    curNbLayers = nbLayers(st);
    for binary = 0:1
        if binary
            curLayerType = [layerTypes{st} '_bin'];
            bLayer = [0 ones(1, curNbLayers-1)];
        else
            curLayerType = [layerTypes{st} '_real'];
            bLayer = [0 zeros(1, curNbLayers-1)];
        end
        for p = 1:length(typePretrains)
            typePretrain = typePretrains{p};
            curTypeTrain = typeTrains{p};
            curVariants = typeVariants{p};
            curFunction = typeFunctions{p};
            curPreNames = variantsParamsPretrain{p};
            curTrainNamesF = variantsParamsTrain{p};
            for v = 1:length(curVariants)
                for f = 1:length(curFunction)
                    curParamType = [curVariants{v} '_' curFunction{f}];
                    curPreVal = curPreVals{p}{v}{f};
                    for t = 1:length(curTypeTrain)
                        typeTrain = curTypeTrain{t};
                        curTrainNames = curTrainNamesF{t};
                        curTrainVal = curTrainVals{p}{v}{f}{t};
                        fullName = [typePretrain '_' typeTrain '_' curLayerType '_' curParamType];
                        fprintf(fullSH, 'sbatch deepJobs_%s.sh\n', fullName);
                        confID = fopen(['deepJobs_' fullName '.sh'], 'w');
                        fprintf(confID, '#!/bin/bash\n\n');
                        fprintf(confID, '#SBATCH --partition=mono\n');
                        fprintf(confID, '#SBATCH --ntasks=1\n');
                        fprintf(confID, '#SBATCH --time=4-0:00\n');
                        fprintf(confID, '#SBATCH --mem-per-cpu=8000\n');
                        fprintf(confID, '#SBATCH -J Deep-%s\n', fullName);
                        fprintf(confID, '#SBATCH -e Deep-%s.err.txt\n', fullName);
                        fprintf(confID, '#SBATCH -o Deep-%s.out.txt\n', fullName);
                        %fprintf(confID, '#SBATCH --licenses=statistics_toolbox@matlablm.unige.ch\n');
                        fprintf(confID, '\n');
                        %
                        % MATLAB Version
                        %
%                         fprintf(confID, 'unset DISPLAY\n');
%                         fprintf(confID, 'module load matlab\n');
%                         fprintf(confID, 'srun matlab -nodesktop -nosplash -nodisplay -r "');
%                         fprintf(confID, 'cd ~/deepLearn;');
%                         fprintf(confID, 'nbLayers=%d;', curNbLayers);
%                         fprintf(confID, 'typePretrain=''%s'';', typePretrain);
%                         fprintf(confID, 'typeTrain=''%s'';', typeTrain);
%                         fprintf(confID, 'layers=[%s];', num2str(structures{st}));
%                         fprintf(confID, 'binaryLayers=%s;', bLayer);
%                         fprintf(confID, 'layerType=''%s'';', curLayerType);
%                         fprintf(confID, 'paramsType=''%s'';', curParamType);
%                         fprintf(confID, 'userStruct = struct;');
%                         fprintf(confID, 'userStruct.pretrain = struct;');
%                         fprintf(confID, curPreNames);
%                         fprintf(confID, 'userStruct.pretrain.values = %s;', curPreVal);
%                         fprintf(confID, 'userStruct.train = struct;');
%                         fprintf(confID, curTrainNames);
%                         fprintf(confID, 'userStruct.train.values = %s;', curTrainVal);
%                         fprintf(confID, 'deepFramework"\n');
                        %
                        % MCC Version
                        %
                        %fprintf(confID, 'unset DISPLAY\n');
                        fprintf(confID, 'source /etc/profile.modules\n\n');
                        fprintf(confID, 'module load gcc\n');
                        fprintf(confID, 'module load matlab\n');
                        %fprintf(confID, 'export LD_LIBRARY_PATH=/opt/gcc/gcc482/lib64:$LD_LIBRARY_PATH\n');
                        %fprintf(confID, 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/unige/matlab2013b/bin/glnxa64:/unige/matlab2013b/runtime/glnxa64\n');
                        %fprintf(confID, 'export XAPPLRESDIR=/unige/matlab2013b/X11/app-defaults\n');
                        %fprintf(confID, 'export PATH=$PATH:$LD_LIBRARY_PATH\n');
                        %fprintf(confID, 'export PATH=$PATH:$XAPPLRESDIR\n');
%                         fprintf(confID, 'srun matlab -nodesktop -nosplash -nodisplay -r "');
                        fprintf(confID, 'cd ~/deepLearn && srun ./deepFunction');
                        fprintf(confID, ' %d', curNbLayers);
                        fprintf(confID, ' ''%s''', typePretrain);
                        fprintf(confID, ' ''%s''', typeTrain);
                        fprintf(confID, ' ''%s''', num2str(structures{st}));
                        fprintf(confID, ' ''%s''', num2str(bLayer));
                        fprintf(confID, ' ''%s''', curLayerType);
                        fprintf(confID, ' ''%s''', curParamType);
                        fprintf(confID, ' "%s"', curPreNames);
                        fprintf(confID, ' ''%s''', curPreVal);
                        fprintf(confID, ' "%s"', curTrainNames);
                        fprintf(confID, ' ''%s''', curTrainVal);
                        fclose(confID);
                    end
                end
            end
        end
    end
end
fclose(fullSH);