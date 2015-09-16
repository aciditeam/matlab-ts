function [cost, grad] = sparseCodingWeightCost(weightMatrix, featureMatrix, visibleSize, numFeatures,  patches, gamma, lambda, epsilon, groupMatrix)
%sparseCodingWeightCost - given the features in featureMatrix, 
%                         computes the cost and gradient with respect to
%                         the weights, given in weightMatrix
% parameters
%   weightMatrix  - the weight matrix. weightMatrix(:, c) is the cth basis
%                   vector.
%   featureMatrix - the feature matrix. featureMatrix(:, c) is the features
%                   for the cth example
%   visibleSize   - number of pixels in the patches
%   numFeatures   - number of features
%   patches       - patches
%   gamma         - weight decay parameter (on weightMatrix)
%   lambda        - L1 sparsity weight (on featureMatrix)
%   epsilon       - L1 sparsity epsilon
%   groupMatrix   - the grouping matrix. groupMatrix(r, :) indicates the
%                   features included in the rth group. groupMatrix(r, c)
%                   is 1 if the cth feature is in the rth group and 0
%                   otherwise.

    if exist('groupMatrix', 'var')
        assert(size(groupMatrix, 2) == numFeatures, 'groupMatrix has bad dimension');
    else
        groupMatrix = eye(numFeatures);
    end

    numExamples = size(patches, 2);

    weightMatrix = reshape(weightMatrix, visibleSize, numFeatures);
    featureMatrix = reshape(featureMatrix, numFeatures, numExamples);
    
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE --------------------    
    errorTerm = (1 ./ numExamples) * sum(sum(sqrt((weightMatrix * featureMatrix - patches) .^ 2)));
    featureSparsityCost = sum(sum(sqrt(featureMatrix .^ 2 + epsilon)));
    weightSparsityCost = (1 ./ numExamples) * sum(sum(weightMatrix .^ 2));
    cost = errorTerm + (lambda * featureSparsityCost) + (gamma * weightSparsityCost);
    
    weightSparsityGrad = gamma .* weightMatrix;
    disp(size(weightSparsityGrad));
    disp(size(weightMatrix' * (weightMatrix * featureMatrix - patches)));
    disp(size((weightMatrix * featureMatrix - patches) * featureMatrix'));
    grad = (2 .* weightMatrix' * (weightMatrix * featureMatrix - patches)) + weightSparsityGrad;

end