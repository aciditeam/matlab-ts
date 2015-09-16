function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.

nbTrain = size(data, 2);
W1 = stack{1}.w;
b1 = stack{1}.b;
W2 = stack{2}.w;
b2 = stack{2}.b;
% First compute the activations
z2 = W1 * data + repmat(b1, 1, nbTrain);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2, 1, nbTrain);
a3 = sigmoid(z3);
% First prevent overflow
theta = bsxfun(@minus, softmaxTheta, max(softmaxTheta, [], 1));
% Compute the predictions
expoTerm = exp(theta * a3);
predict = bsxfun(@rdivide, expoTerm, sum(expoTerm));
errorTerm = -(1 / nbTrain) .* sum(sum(groundTruth .* log(predict)));
weightDecayTerm = sum(sum(softmaxTheta .^ 2)) + sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2));
cost = errorTerm + ((lambda / 2) * weightDecayTerm);

softmaxThetaGrad = (- (1 / nbTrain) .* ((groundTruth - predict) * a3')) + (lambda .* softmaxTheta);
delta3 = -(theta' * (groundTruth - predict)) .* dsigmoid(a3);
delta2 = (W2' * delta3) .* dsigmoid(a2);
nablaW2 = delta3 * a2';
nablab2 = delta3;
nablaW1 = delta2 * data';
nablab1 = delta2;
 
stackgrad{1}.w = nablaW1 ./ nbTrain + lambda .* W1;
stackgrad{2}.w = nablaW2 ./ nbTrain + lambda .* W2;
stackgrad{1}.b = sum(nablab1, 2) ./ nbTrain;
stackgrad{2}.b = sum(nablab2, 2) ./ nbTrain;


disp(size(softmaxThetaGrad));







% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end


function dsigm = dsigmoid(a)
    dsigm = a .* (1.0 - a);
end
