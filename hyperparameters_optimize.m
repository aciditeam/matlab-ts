function [optimize] = hyperparameters_optimize(nbLayers)
optimize = struct;
% Parameters of the structure
optimize.structure.units.values = [32 4096];
optimize.structure.units.step = 32;
optimize.structure.units.tuning = cell(length(nbLayers), 1);
optimize.structure.units.past = cell(length(nbLayers), 1);
optimize.structure.units.errors = cell(length(nbLayers), 1);
optimize.structure.types.values = [0 1];
optimize.structure.types.tuning = cell(length(nbLayers), 1);
optimize.structure.types.past = cell(length(nbLayers), 1);
optimize.structure.types.errors = cell(length(nbLayers), 1);
optimize.units = [64 128 256 512 1024 2048 4096];
% Parameters shared by all pre-training variants
optimize.pretrain.names = {'learning.lrate', 'learning.momentum', 'learning.weight_decay', 'learning.lrate_anneal', 'learning.minibatch_sz', 'adadelta.use', 'adadelta.epsilon', 'adadelta.momentum'};
optimize.pretrain.values = {[1e-4 1e-3 1e-2 1e-1], [0 0.9], [0.000001 0.01], [0 0.99], [16 32 50 64 100 128 256], [0 1], [1e-9 1e-5], [0 0.9]};
optimize.pretrain.continuous = [1 1 1 1 0 0 1 1];
optimize.pretrain.default = [1e-2 0.9 0.0002 0.9 150 1 1e-8 0.9];
optimize.pretrain.step = [1e-5, 0.1, 0.0005, 0.1, 32, 1 1e-7 0.1];
optimize.pretrain.tuning = cell(length(optimize.pretrain.names), 1);
optimize.pretrain.past = cell(length(optimize.pretrain.names), 1);
optimize.pretrain.errors = cell(length(optimize.pretrain.names), 1);
% Parameters shared by all training variants
optimize.train.names = {'learning.lrate', 'learning.momentum', 'learning.weight_decay', 'learning.lrate_anneal', 'learning.minibatch_sz'};
optimize.train.values = {[1e-4 1e-3 1e-2 1e-1], [0 0.9], [0.000001 0.01], [0 0.9], [16 32 50 64 100 128 256]};
optimize.train.continuous = [1 1 1 1 0];
optimize.train.default = [1e-2 0.9 0.0002 0.9 150];
optimize.train.step = [1e-5, 0.1, 0.0005, 0.1, 32];
optimize.train.tuning = cell(length(optimize.train.names), 1);
optimize.train.past = cell(length(optimize.train.names), 1);
optimize.train.errors = cell(length(optimize.train.names), 1);
% Parameters specific to DAE
optimize.pretrain.DAE.names = {'use_tanh', 'sparsity.target', 'sparsity.cost', 'do_normalize', 'do_normalize_std', 'noise.drop', 'noise.level', 'rica.cost', 'cae.cost'};
optimize.pretrain.DAE.values = {[0 1 2], [0.0001 0.1], [0 0.9], [0 1], [0 1], [0 0.3], [0 0.5], [0 0], [0 0]};
optimize.pretrain.DAE.continuous = [0 1 1 0 0 1 1 0 0];
optimize.pretrain.DAE.default = [0 0.01 0.5 0 0 0 0 0 0];
optimize.pretrain.DAE.step = [1, 0.1, 0.05, 1, 1, 0.05, 0.05 0.05 0.05];
optimize.pretrain.DAE.tuning = cell(length(optimize.pretrain.DAE.names), 1);
optimize.pretrain.DAE.past = cell(length(optimize.pretrain.DAE.names), 1);
optimize.pretrain.DAE.errors = cell(length(optimize.pretrain.DAE.names), 1);
% Parameters specific to RBM
optimize.pretrain.RBM.names = {'learning.cd_k', 'learning.persistent_cd', 'grbm.do_vsample', 'grbm.do_normalize', 'grbm.do_normalize_std', 'grbm.learn_sigmas', 'grbm.use_single_sigma', 'adaptive_lrate.use', 'enhanced_grad.use', 'adaptive_momentum.use', 'fast.use', 'fast.lrate', 'parallel_tempering.use','parallel_tempering.n_chains','parallel_tempering.swap_interval'};
optimize.pretrain.RBM.values = {[1 2 3 4 5 6 7 8 9 10], [0 1], [0 1], [0 1], [0 1], [0 1], [0 1], [0 1], [0 1], [0 1], [0 1], [1e-6 1e-5 1e-4 1e-3], [0 1], [3 5 7 9 11 13 15 17 19 21 23], [1 2 3 4 5]};
optimize.pretrain.RBM.continuous = [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0];
optimize.pretrain.RBM.default = [1 0 0 1 1 0 0 1 1 0 0 1e-2 0 11 1];
optimize.pretrain.RBM.step = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-5, 1, 2, 1];
optimize.pretrain.RBM.tuning = cell(length(optimize.pretrain.RBM.names), 1);
optimize.pretrain.RBM.past = cell(length(optimize.pretrain.RBM.names), 1);
optimize.pretrain.RBM.errors = cell(length(optimize.pretrain.RBM.names), 1);
% Parameters specific to SDAE
optimize.train.SDAE.names = {'use_tanh', 'bottleneck.binary', 'do_normalize', 'do_normalize_std', 'noise.drop', 'noise.level', 'adadelta.use', 'adadelta.epsilon', 'adadelta.momentum'};
optimize.train.SDAE.values = {[0 1 2], [0 1], [0 1], [0 1], [0 0.5], [0 0.5], [0 1], [1e-9 1e-5], [0 0.99]};
optimize.train.SDAE.continuous = [0 0 0 0 1 1 0 1 1];
optimize.train.SDAE.default = [0 1 0 0 0.1 0.1 1 1e-5 0.9];
optimize.train.SDAE.step = [1, 1, 1, 1, 0.05, 0.05, 1, 1e-7, 0.1];
optimize.train.SDAE.tuning = cell(length(optimize.train.SDAE.names), 1);
optimize.train.SDAE.past = cell(length(optimize.train.SDAE.names), 1);
optimize.train.SDAE.errors = cell(length(optimize.train.SDAE.names), 1);
% Parameters specific to DBM
optimize.train.DBM.names = {'learning.cd_k', 'learning.persistent_cd', 'grbm.do_vsample', 'grbm.do_normalize', 'grbm.do_normalize_std', 'grbm.learn_sigmas', 'grbm.use_single_sigma', 'adaptive_lrate.use', 'enhanced_grad.use', 'centering.use'};
optimize.train.DBM.values = {[0 1], [0 1], [0 1], [0 1], [0 1], [0 1], [0 1], [0 1], [0 1], [0 1]};
optimize.train.DBM.continuous = [0 0 0 0 0 0 0 0 0 0];
optimize.train.DBM.default = [1 0 1 1 1 1 1 1 1 0];
optimize.train.DBM.step = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
optimize.train.DBM.tuning = cell(length(optimize.train.DBM.names), 1);
optimize.train.DBM.past = cell(length(optimize.train.DBM.names), 1);
optimize.train.DBM.errors = cell(length(optimize.train.DBM.names), 1);
% Parameters specific to DBN
optimize.train.DBN.names = {'learning.contrastive_step', 'learning.persistent_cd', 'learning.ffactored'};
optimize.train.DBN.values = {[0 1], [0 1], [0 1]};
optimize.train.DBN.continuous = [0 0 0];
optimize.train.DBN.default = [1 1 0];
optimize.train.DBN.step = [1 1 1];
optimize.train.DBN.tuning = cell(length(optimize.train.DBN.names), 1);
optimize.train.DBN.past = cell(length(optimize.train.DBN.names), 1);
optimize.train.DBN.errors = cell(length(optimize.train.DBN.names), 1);
% Parameters specific to MLP
optimize.train.MLP.names = {'use_tanh', 'dropout.use', 'dropout.global_proba', 'noise.drop', 'noise.level'};
optimize.train.MLP.values = {[0 1 2], [0 1], [0.2 0.9], [0 0.8], [0 0.8]};
optimize.train.MLP.continuous = [0 0 1 1 1];
optimize.train.MLP.default = [0 1 0.5 0.1 0.1];
optimize.train.MLP.step = [1 1 0.05 0.05 0.05];
optimize.train.MLP.tuning = cell(length(optimize.train.MLP.names), 1);
optimize.train.MLP.past = cell(length(optimize.train.MLP.names), 1);
optimize.train.MLP.errors = cell(length(optimize.train.MLP.names), 1);
% Number of iterations in training
optimize.pretrain.iteration.n_epochs = 20;
optimize.train.iteration.n_epochs = 20;
end