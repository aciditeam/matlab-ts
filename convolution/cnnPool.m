function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);
nbRegions = floor(convolvedDim / poolDim);
pooledFeatures = zeros(numFeatures, numImages, nbRegions, nbRegions);

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------

for f = 1:numFeatures
    for i = 1:numImages
        for p1 = 1:nbRegions
            p1IDs = (((p1 - 1) * poolDim)+1):(p1*poolDim);
            for p2 = 1:nbRegions
                p2IDs = (((p2 - 1) * poolDim)+1):(p2*poolDim);
                pooledFeatures(f, i, p1, p2) = mean(mean(convolvedFeatures(f, i, p1IDs, p2IDs)));
            end
        end
    end
end
end

