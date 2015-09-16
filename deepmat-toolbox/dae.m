% dae.m     : Training a single-layer Deep Auto-Encoder
%
% Input     :
% - D       : Structure specifying the model
% - patches : Input patches to train on
%
function [D] = dae(D, patches, valid_patches, valid_portion)
% If un-specified don't use validation
if nargin < 3
    early_stop = 0;
    valid_patches = [];
    valid_portion = 0;
else
    early_stop = 1;
    valid_err = -Inf;
    valid_best_err = -Inf;
end
% Preset learning rate (for later adaptive rate)
actual_lrate = D.learning.lrate;
% Number of input examples
n_samples = size(patches, 1);
% Check coherence of visible layer and data
if D.structure.n_visible ~= size(patches, 2)
    error('Visible layer size is not coherent with data dimensionality');
end
% Initialize gradients variables
vbias_grad_old = zeros(size(D.vbias'));
hbias_grad_old = zeros(size(D.hbias'));
W_grad_old = zeros(size(D.W));
% Size of each minibatch
minibatch_sz = D.learning.minibatch_sz;
% Corresponding number of minibatches required
n_minibatches = ceil(n_samples / minibatch_sz);
% Number of training epochs to use
n_epochs = D.iteration.n_epochs;
% Learning momentum (gradient signal smoothing)
momentum = D.learning.momentum;
% Influence of weight decay penalty
weight_decay = D.learning.weight_decay;
% Number of visible and hidden units
n_visible = D.structure.n_visible;
n_hidden = D.structure.n_hidden;
% Parameters for early stopping
min_recon_error = Inf;
min_recon_error_update_idx = 0;
stopping = 0;
% Variables for eventual data normalization
do_normalize = D.do_normalize;
do_normalize_std = D.do_normalize_std;
% Normalize data if it is real-valued
if D.data.binary == 0
    if do_normalize == 1
        % Normalize data to zero-mean
        patches_mean = mean(patches, 1);
        patches = bsxfun(@minus, patches, patches_mean);
    end
    if do_normalize_std ==1
        % Normalize data to unit-variance
        patches_std = std(patches, [], 1);
        patches = bsxfun(@rdivide, patches, patches_std);
    end
end
% Parameters for annealing
anneal_counter = 0;
actual_lrate0 = actual_lrate;
% Debug mode parameters
if D.debug.do_display == 1
    figure(D.debug.display_fid);
end
% Check if GPU can be used for computation
try
   use_gpu = gpuDeviceCount;
catch errgpu
   use_gpu = false;
   disp(['Could not use CUDA. Error: ' errgpu.identifier])
end
% Iterate learning for number of epochs
for step=1:n_epochs
    if D.verbose
        fprintf(2, 'Epoch %d/%d: ', step, n_epochs)
    end
    % GPU Arrays creation (push)
    if use_gpu
        D.W = gpuArray(single(D.W));
        D.vbias = gpuArray(single(D.vbias));
        D.hbias = gpuArray(single(D.hbias));
        % Adagrad arrays
        if D.adagrad.use
            D.adagrad.W = gpuArray(single(D.adagrad.W));
            D.adagrad.vbias = gpuArray(single(D.adagrad.vbias));
            D.adagrad.hbias = gpuArray(single(D.adagrad.hbias));
        end
        % Adadelta arrays
        if D.adadelta.use
            % Gradient arrays
            D.adadelta.gW = gpuArray(single(D.adadelta.gW));
            D.adadelta.gvbias = gpuArray(single(D.adadelta.gvbias));
            D.adadelta.ghbias = gpuArray(single(D.adadelta.ghbias));
            % Normal arrays
            D.adadelta.W = gpuArray(single(D.adadelta.W));
            D.adadelta.vbias = gpuArray(single(D.adadelta.vbias));
            D.adadelta.hbias = gpuArray(single(D.adadelta.hbias));
        end
    end
    % Iterate over minibatches
    for mb=1:n_minibatches
        % Number of iterations performed by the system
        D.iteration.n_updates = D.iteration.n_updates + 1;
        % Retrieve data for current minibatch
        v0 = patches((mb - 1) * minibatch_sz + 1:min(mb * minibatch_sz, n_samples), :);
        mb_sz = size(v0,1);
        % Transfer data to GPU memory
        if use_gpu
            v0 = gpuArray(single(v0));
        end
        % Original version
        v0_clean = v0;
        % Add noise to all the series
        if D.data.binary == 0 && D.noise.level > 0
            v0 = v0 + (D.noise.level * randn(size(v0)));
        end
        % Add outliers to all the series
        if D.noise.drop > 0
            mask = binornd(1, 1 - D.noise.drop, size(v0));
            v0 = v0 .* mask;
            clear mask;
        end
        % Add warping to all the series
        if D.noise.warp > 0
            % Compute set of switch positions
            switch_pos = ceil(rand(size(v0, 1), 1) * (size(v0, 2) / 2)) + (size(v0, 2) / 4);
            % Iterate and warp over minibatch
            for i = 1:size(v0, 1)
                leftSide = v0(i, 1:switch_pos(i));
                rightSide = v0(i, (switch_pos(i)+1):end);
                direction = sign(rand - 0.5);
                leftSide = resample(leftSide, switch_pos(i) - (direction * round(D.noise.warp * size(v0, 2))), switch_pos(i));
                rightSide = resample(rightSide, size(v0, 2) - switch_pos(i) + (direction * round(D.noise.warp * size(v0, 2))), size(v0, 2) - switch_pos(i));
                v0(i, :) = [leftSide rightSide];
            end
        end
        % Compute hidden activations (real data is RELU)
        h0 = bsxfun(@plus, v0 * D.W, D.hbias');
        % Binary data is sigmoid activation
        if D.hidden.binary
            h0 = sigmoid(h0, D.hidden.use_tanh);
        end
        % Compute hidden activations (based on clean data)
        hr = bsxfun(@plus, v0_clean * D.W, D.hbias');
        % Binary data is sigmoid activation
        if D.hidden.binary
            hr = sigmoid(hr, D.hidden.use_tanh);
        end
        % Compute reconstruction based on clean data
        vr = bsxfun(@plus,hr * D.W',D.vbias');
        if D.data.binary
            vr = sigmoid(vr, D.visible.use_tanh);
        end
        % Compute reconstruction error (between clean and its reconstruction ?)
        if D.data.binary && ~D.visible.use_tanh
            % Computation for sigmoid-based activations (cross entropy)
            rerr = -mean(sum(v0_clean .* log(max(vr, 1e-16)) + (1 - v0_clean) .* log(max(1 - vr, 1e-16)), 2));
        else
            rerr = mean(sum((v0_clean - vr).^2,2));
        end
        % Retrieve error from GPU
        if use_gpu
            rerr = gather(rerr);
        end
        % NaN reconstruction error
        if isnan(rerr)
            disp('v0 has NaN ?');
            disp(sum(sum(isnan(v0))));
            disp('v0_clean has NaN ?');
            disp(sum(sum(isnan(v0_clean))));
            disp('h0 has NaN ?');
            disp(sum(sum(isnan(h0))));
            disp('hr has NaN ?');
            disp(sum(sum(isnan(hr))));
            disp('vr has NaN ?');
            disp(sum(sum(isnan(vr))));
            error('Reconstruction error of DAE is NaN');
        end
        % Record the reconstruction error
        D.signals.recon_errors = [D.signals.recon_errors rerr];
        % Compute the gradient (based on noisy data)
        vr = bsxfun(@plus,h0 * D.W',D.vbias');
        if D.data.binary
            vr = sigmoid(vr, D.visible.use_tanh);
        end
        % Difference between noisy reconstruction and clean data
        % DEBUG
        % DEBUG
        % DEBUG
        % DEBUG
        % THIS DOES NOT ACCOUNT FOR CROSS-ENTROPY LOSS !
        % DEBUG
        % DEBUG
        % DEBUG
        % DEBUG
        deltao = vr - v0_clean;
        % Multiply by the sigmoid derivative (if used)
        if D.data.binary && D.visible.use_tanh
            deltao = deltao .* dsigmoid(vr, D.visible.use_tanh);
        end
        % Gradient of output bias is mean
        vbias_grad = mean(deltao, 1);
        % Gradient of hidden weights
        deltah = deltao * D.W;
        % Multiply by the sigmoid derivative (if used)
        if D.hidden.binary
            deltah = deltah .* dsigmoid(h0, D.hidden.use_tanh);
        end
        % Gradient of hidden bias is mean
        hbias_grad = mean(deltah, 1);
        % Gradient of hidden weights
        W_grad = ((deltao' * h0) + (v0' * deltah)) / size(v0,1);
        % Empty the deltas
        clear deltao deltah;
        % DEBUG
        % DEBUG
        % DEBUG
        % DEBUG
        % THIS SPARSITY MEASURE SEEMS WRONG !
        % DEBUG
        % DEBUG
        % DEBUG
        % DEBUG
        if D.sparsity.cost > 0 && D.hidden.use_tanh == 0
            
            diff_sp = (h0 - D.sparsity.target);
            hbias_grad = hbias_grad + D.sparsity.cost * mean(diff_sp, 1);
            %W_grad = W_grad + (D.sparsity.cost/size(v0,1)) * (v0_clean' * diff_sp);
            W_grad = W_grad + (D.sparsity.cost/size(v0,1)) * (v0' * diff_sp);
            clear diff_sp;
        end
        % Compute cost and gradient for CAE penalty
        if D.cae.cost > 0 && D.hidden.use_tanh == 0
            % Compute the cost
            W_cae1 = bsxfun(@times, D.W, mean(h0 .* (1 - h0).^2, 1));
            W_cae2 = D.W.^2 .* (v0' * (...
                (1 - 2 * h0) .* h0 .* (1 - h0).^2 ...
            ) / size(v0, 1));
            W_cae = W_cae1 + W_cae2;
            % Compute the weight gradient
            W_grad = W_grad + D.cae.cost * W_cae;
            clear W_cae1 W_cae2 W_cae;
            % Compute the bias cost and gradient
            hbias_cae = sum(bsxfun(@times, D.W, mean(h0 .* (1 - h0).^2 .* (1 - 2 * h0),1)), 1);
            hbias_grad = hbias_grad + D.cae.cost * hbias_cae;
            clear hbias_cae;
        end
        % Compute cost and gradient for RICA penalty
        if D.rica.cost > 0
            W_rica = v0_clean' * tanh(hr);
            W_grad = W_grad + D.rica.cost * W_rica;
            clear W_rica;
        end
        % Clear reconstruction
        clear hr vr;
        % Use adagrad to update the values
        if D.adagrad.use
            % Smooth gradients based on momentum
            vbias_grad_old = (1-momentum) * vbias_grad + momentum * vbias_grad_old;
            hbias_grad_old = (1-momentum) * hbias_grad + momentum * hbias_grad_old;
            W_grad_old = (1-momentum) * W_grad + momentum * W_grad_old;
            % Update the values 
            D.adagrad.W = D.adagrad.W + W_grad_old.^2;
            D.adagrad.vbias = D.adagrad.vbias + vbias_grad_old.^2';
            D.adagrad.hbias = D.adagrad.hbias + hbias_grad_old.^2';            
            % DEBUG
            % DEBUG
            % DEBUG
            % DEBUG
            % UPDATE THE VBIAS AND HBIAS ONLY IF NO RICA RECONSTRUCTION ?
            % DEBUG
            % DEBUG
            % DEBUG
            % DEBUG
            if D.rica.cost <= 0
                D.vbias = D.vbias - D.learning.lrate * (vbias_grad_old' + weight_decay * D.vbias) ./ sqrt(D.adagrad.vbias + D.adagrad.epsilon);
                D.hbias = D.hbias - D.learning.lrate * (hbias_grad_old' + weight_decay * D.hbias) ./ sqrt(D.adagrad.hbias + D.adagrad.epsilon);
            end
            % Update the weights
            D.W = D.W - D.learning.lrate * (W_grad_old + weight_decay * D.W) ./ sqrt(D.adagrad.W + D.adagrad.epsilon);
        % Use adadelta to update the values
        elseif D.adadelta.use
            % Smooth gradients based on momentum
            vbias_grad_old = (1-momentum) * vbias_grad + momentum * vbias_grad_old;
            hbias_grad_old = (1-momentum) * hbias_grad + momentum * hbias_grad_old;
            W_grad_old = (1-momentum) * W_grad + momentum * W_grad_old;
            % Update the adaptive momentum
            if D.iteration.n_updates == 1
                adamom = 0;
            else
                adamom = D.adadelta.momentum;
            end
            % Keep the updated gradient based on adadelta
            D.adadelta.gW = adamom * D.adadelta.gW + (1 - adamom) * W_grad_old.^2;
            D.adadelta.gvbias = adamom * D.adadelta.gvbias + (1 - adamom) * vbias_grad_old.^2';
            D.adadelta.ghbias = adamom * D.adadelta.ghbias + (1 - adamom) * hbias_grad_old.^2';
            % Update based on RICA
            if D.rica.cost <= 0
                dvbias = -(vbias_grad_old' + ...
                    weight_decay * D.vbias) .* (sqrt(D.adadelta.vbias + D.adadelta.epsilon) ./ sqrt(D.adadelta.gvbias + D.adadelta.epsilon));
                dhbias = -(hbias_grad_old' + ...
                    weight_decay * D.hbias) .* (sqrt(D.adadelta.hbias + D.adadelta.epsilon) ./ sqrt(D.adadelta.ghbias + D.adadelta.epsilon));
                D.vbias = D.vbias + dvbias;
                D.hbias = D.hbias + dhbias;
            end
            % Compute the weights and update
            dW = -(W_grad_old + weight_decay * D.W) .* ...
                (sqrt(D.adadelta.W + D.adadelta.epsilon) ./ sqrt(D.adadelta.gW + D.adadelta.epsilon));
            D.W = D.W + dW;
            D.adadelta.W = adamom * D.adadelta.W + (1 - adamom) * dW.^2;
            clear dW;
            % Update without RICA
            if D.rica.cost <= 0
                D.adadelta.vbias = adamom * D.adadelta.vbias + (1 - adamom) * dvbias.^2;
                D.adadelta.hbias = adamom * D.adadelta.hbias + (1 - adamom) * dhbias.^2;
                clear dvbias dhbias;
            end
        else
            % When not using adagrad nor adadelta, but use annealing
            if D.learning.lrate_anneal > 0 && (step >= D.learning.lrate_anneal * n_epochs)
                anneal_counter = anneal_counter + 1;
                actual_lrate = actual_lrate0 / anneal_counter;
            else
                % DEBUG
                % DEBUG
                % DEBUG
                % DEBUG
                % UPDATE THE LRATE by dividing it (even though NO
                % annealing is used ?)
                % DEBUG
                % DEBUG
                % DEBUG
                % DEBUG
                if D.learning.lrate0 > 0
                    actual_lrate = D.learning.lrate / (1 + D.iteration.n_updates / D.learning.lrate0);
                else
                    actual_lrate = D.learning.lrate;
                end
                actual_lrate0 = actual_lrate;
            end
            % Keep successive values of the learning rates
            D.signals.lrates = [D.signals.lrates actual_lrate];
            % Update the old gradients
            vbias_grad_old = (1-momentum) * vbias_grad + momentum * vbias_grad_old;
            hbias_grad_old = (1-momentum) * hbias_grad + momentum * hbias_grad_old;
            W_grad_old = (1-momentum) * W_grad + momentum * W_grad_old;
            % Update for the RICA case
            if D.rica.cost <= 0
                D.vbias = D.vbias - actual_lrate * (vbias_grad_old' + weight_decay * D.vbias);
                D.hbias = D.hbias - actual_lrate * (hbias_grad_old' + weight_decay * D.hbias);
            end
            % Final update of the weights
            D.W = D.W - actual_lrate * (W_grad_old + weight_decay * D.W);
        end
        if D.verbose == 1
            fprintf(2, '.');
        end
        if use_gpu
            clear v0 h0 v0_clean vr hr deltao deltah 
        end
        % Early stopping criterias
        if early_stop
            % Retrieve the validation data
            n_valid = size(valid_patches, 1);
            % Perform a random permutation
            rndidx = randperm(n_valid);
            v0valid = valid_patches(rndidx(1:round(n_valid * valid_portion)),:);
            if use_gpu
                v0valid = gpuArray(single(v0valid));
            end
            % Compute hidden activation
            hr = bsxfun(@plus, v0valid * D.W, D.hbias');
            if D.hidden.binary
                hr = sigmoid(hr, D.hidden.use_tanh);
            end
            % Compute the reconstruction
            vr = bsxfun(@plus,hr * D.W',D.vbias');
            if D.data.binary
                vr = sigmoid(vr, D.visible.use_tanh);
            end
            % Compute the reconstruction error
            if D.data.binary && ~D.visible.use_tanh
                rerr = -mean(sum(v0valid .* log(max(vr, 1e-16)) + (1 - v0valid) .* log(max(1 - vr, 1e-16)), 2));
            else
                rerr = mean(sum((v0valid - vr).^2,2));
            end
            if use_gpu
                rerr = gather(rerr);
            end
            % Keep trace of validation error
            D.signals.valid_errors = [D.signals.valid_errors rerr];
            % Watch for previous errors
            if valid_err == -Inf
                valid_err = rerr;
                valid_best_err = rerr;
            else
                prev_err = valid_err;
                valid_err = 0.99 * valid_err + 0.01 * rerr;
                % Check if the validation error is going up (over-training)
                if step > D.valid_min_epochs && (1.1 * valid_best_err) < valid_err
                    fprintf(2, 'Early-stop! %f, %f\n', valid_err, prev_err);
                    stopping = 1;
                    break;
                end
                % Keep best error
                if valid_err < valid_best_err
                    valid_best_err = valid_err;
                end
            end
        else
            % Check different stop criterias
            if D.stop.criterion > 0
                if D.stop.criterion == 1
                    if min_recon_error > D.signals.recon_errors(end)
                        min_recon_error = D.signals.recon_errors(end);
                        min_recon_error_update_idx = D.iteration.n_updates;
                    else
                        if D.iteration.n_updates > min_recon_error_update_idx + D.stop.recon_error.tolerate_count 
                            fprintf(2, '\nStopping criterion reached (recon error) %f > %f\n', ...
                                D.signals.recon_errors(end), min_recon_error);
                            stopping = 1;
                            break;
                        end
                    end
                else
                    error ('Unknown stopping criterion %d', D.stop.criterion);
                end
            end
        end
        if length(D.hook.per_update) > 1
            err = D.hook.per_update{1}(D, D.hook.per_update{2});
            if err == -1
                stopping = 1;
                break;
            end
        end
        % Display the debugging information
        if D.debug.do_display == 1 && mod(D.iteration.n_updates, D.debug.display_interval) == 0
            D.debug.display_function (D.debug.display_fid, D, v0, v1, h0, h1, W_grad, vbias_grad, hbias_grad);
            drawnow;
        end
    end
    % Retrieve various arrays from GPU memory (pull)
    if use_gpu
        D.W = gather(D.W);
        D.vbias = gather(D.vbias);
        D.hbias = gather(D.hbias);
        if D.adagrad.use
            D.adagrad.W = gather(D.adagrad.W);
            D.adagrad.vbias = gather(D.adagrad.vbias);
            D.adagrad.hbias = gather(D.adagrad.hbias);
        elseif D.adadelta.use
            D.adadelta.W = gather(D.adadelta.W);
            D.adadelta.vbias = gather(D.adadelta.vbias);
            D.adadelta.hbias = gather(D.adadelta.hbias);
            D.adadelta.gW = gather(D.adadelta.gW);
            D.adadelta.gvbias = gather(D.adadelta.gvbias);
            D.adadelta.ghbias = gather(D.adadelta.ghbias);
        end
    end
    if length(D.hook.per_epoch) > 1
        err = D.hook.per_epoch{1}(D, D.hook.per_epoch{2});
        if err == -1
            stopping = 1;
        end
    end
    if stopping == 1
        break;
    end
    if D.verbose == 1
        fprintf(2, '\n');
    end 
    fprintf(2, 'Epoch %d/%d - recon_error: %f norms: %f/%f/%f\n', step, n_epochs, ...
        D.signals.recon_errors(end), ...
        D.W(:)' * D.W(:) / length(D.W(:)), ...
        D.vbias' * D.vbias / length(D.vbias), ...
        D.hbias' * D.hbias / length(D.hbias));
end
% Retrieve GPU arrays
if use_gpu
    D.W = gather(D.W);
    D.vbias = gather(D.vbias);
    D.hbias = gather(D.hbias);
    if D.adagrad.use
        D.adagrad.W = gather(D.adagrad.W);
        D.adagrad.vbias = gather(D.adagrad.vbias);
        D.adagrad.hbias = gather(D.adagrad.hbias);
    elseif D.adadelta.use
        D.adadelta.W = gather(D.adadelta.W);
        D.adadelta.vbias = gather(D.adadelta.vbias);
        D.adadelta.hbias = gather(D.adadelta.hbias);

        D.adadelta.gW = gather(D.adadelta.gW);
        D.adadelta.gvbias = gather(D.adadelta.gvbias);
        D.adadelta.ghbias = gather(D.adadelta.ghbias);
    end
end


