% Version 0.100 (Unsupported, unreleased)
%
% Code provided by Graham Taylor and Geoff Hinton
%
% For more information, see:
%    http://www.cs.toronto.edu/~gwtaylor/publications/icml2009
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, expressed or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.
%
% For a series of sequences stored in the cell array Motion
% Go through each, and convert CMU "Euler" data to exponential maps

for jj = 1:length(Motion)
  for ii=1:size(Motion{jj},1)
    [skel1,Motion1{jj}(ii,:)] = bvh2expmap(skel,Motion{jj}(ii,:));  
  end
end

%Legacy code: expects skel.tree(xx).expmapInd
%Even though with bvh, expmapInd = rotInd
for ii=1:length(skel1.tree)
  skel1.tree(ii).expmapInd = skel1.tree(ii).rotInd;
end