%
%    Produces the critical difference of the statistical significance of a matrix of
%    scores, S, achieved by a set of machine learning algorithms.
%
%    References
%    [1] Demsar, J., "Statistical comparisons of classifiers over multiple
%        datasets", Journal of Machine Learning Research, vol. 7, pp. 1-30,
%        2006.
%
function [ranks] = hyperparameters_criticaldifference(s)
% convert scores into ranks
[N,k] = size(s);
[S,r] = sort(s');
idx   = k*repmat(0:N-1, k, 1)' + r';
R     = repmat(1:k, N, 1);
S     = S';
for i=1:N
   for j=1:k
      index      = S(i,j) == S(i,:);
      R(i,index) = mean(R(i,index));
   end
end
r(idx)  = R;
ranks   = mean(r');
end