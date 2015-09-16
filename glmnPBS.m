classdef glmnPBS
    %Guillimin PBS submission arguments
    properties
        localScript         = 'deepOptimizePretrain';
        workingDirectory    = '.';          % Remote directory (home default)
        nbWorkers           = 1;            % Number of Matlab workers
        numberOfNodes       = 1;            % Number of nodes
        procsPerNode        = 1;            % Processors per nodes
        gpus                = 1;            % Number of GPUs
        phis                = 0;            % Number of PhiX CPUs
        attributes          = '';           % Additionnal attributes
        pmem                = '4000m';      % Memory per process
        walltime            = '72:00:00';   % Requested walltime
        queue               = 'aw';         % Queue used (metaq = auto)
        account             = 'ymd-084-aa'; % Project (never change)
        otherOptions        = '';           % Options to QSUB command
    end

    methods(Static)
        function job = submitTo(cluster, wSpace)
            opt = glmnPBS();
            job = batch(cluster, opt.localScript, 0, wSpace,    ...
                'matlabpool',       opt.getNbWorkers(),         ...
                'CurrentDirectory', opt.workingDirectory        ...
                );
        end
    end

    methods
        function nbWorkers = getNbWorkers(obj)
            nbWorkers = obj.nbWorkers;
        end

        function submitArgs = getSubmitArgs(obj)
            pbsAccount = '';
            if size(obj.account) > 0
                pbsAccount = sprintf('-A %s', obj.account);
            end

            compRes = sprintf('nodes=%d:ppn=%d', obj.numberOfNodes, obj.procsPerNode);

            if obj.gpus > 0
                compRes = sprintf('%s:gpus=%d', compRes, obj.gpus);
            end

            if obj.phis > 0
                compRes = sprintf('%s:phis=%d', compRes, obj.phis);
            end

            if not(isempty(obj.attributes))
                compRes = sprintf('%s:%s', compRes, obj.attributes);
            end

            compRes = sprintf('%s -l pmem=%s -l walltime=%s', compRes, obj.pmem, obj.walltime);

            nLicenses = obj.getNbWorkers() + 1;

            submitArgs = sprintf('%s -q %s -l %s -W x=GRES:MATLAB_Distrib_Comp_Engine:%d %s', ...
                pbsAccount, obj.queue, compRes, nLicenses, obj.otherOptions);
        end
    end
end





