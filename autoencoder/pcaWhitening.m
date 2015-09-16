% Performing a zero-mean transform
avg = mean(x, 1);
x = x - repmat(avg, size(x, 1), 1);
% Compute the covariance values
sigma = x * x' / size(x, 2);
% Compute the eigenvectors
[U, S] = svd(sigma);
% Compute the rotated version
xRot = U' * x;
% Extract the eigenvalues
eigVals = diag(S);
% Find the dimensionality cutoff
varCutoff = .95;
[~, idK] = find(cumsum(S ./ sum(eigVals)) > varCutoff, 1, 'first');
% Dimensionality-reduced versions
xTilde = U(:, 1:idK)' * x;
% Compute PCA-whitened data
epsilon = 0.00001;
xPCAwhite = diag(1 ./ sqrt(diag(S) + epsilon)) * U' * x;
% Compute ZCA-whitened data
xZCAwhite = U * diag(1 ./ sqrt(diag(S) + epsilon)) * U' * x;