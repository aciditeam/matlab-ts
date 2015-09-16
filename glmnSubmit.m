%%
%
% Validation step
%
%

% 1 - Try using the automatic validation

% 2 - Use the following manual validation
setSchedulerMessageHandler(@disp)
cluster = parcluster('guillimin');
cluster.NumWorkers = 3;
job = createCommunicatingJob(cluster, 'Type', 'spmd') ;
createTask(job, @labindex, 1, {});
submit(job);
wait(job);
out = fetchOutputs(job);

%%
%
% Verification step
%
%

% Just check the values of the overall script
test = glmnPBS();
test.getSubmitArgs()

%%
%
% CPUs vs GPUs speed checking
%
%
clear glmnPBS;
clear all;
% Workspace variable
workspace = struct;
workspace.typePretrain = 'DAE';
workspace.typeTrain = 'MLP';
workspace.nbLayers = 5;
workspace.fileIDchar = 'a';
% Submit this single job to the cluster
setSchedulerMessageHandler(@disp)
cluster = parcluster('guillimin');
cluster.NumWorkers = 11;
typesPre = {'DAE', 'RBM'};
curJob = 3;
for nbLayers = 4%:5
	for idChars = 'e':'f'
        for type = 1%:2
            curJob = curJob + 1;
            test = glmnPBS();
            test.submitTo(cluster, {nbLayers, typesPre{type}, 'MLP', idChars});
        end
	end
end