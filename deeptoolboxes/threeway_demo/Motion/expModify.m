% Version 0.100 (Unsupported, unreleased) 
%
% Code provided by Graham Taylor and Geoff Hinton 
%
% For more information, see:
%     http://www.cs.toronto.edu/~gwtaylor/publications/nips2006mhmublv
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
% Based on skelModify.m version 1.1
% Copyright (c) 2006 Neil D. Lawrence
%
% We support two types of skeletons:
%  1) Those built from the CMU database (acclaim)
%     http://mocap.cs.cmu.edu/
%  2) Those built from data from Eugene Hsu (mit)
%     http://people.csail.mit.edu/ehsu/work/sig05stf/
%
% EXPMODIFY Helper code for visualisation of skel data.
%
% Usage: expModify(handle, channels, skel, padding)
%
function expModify(handle, vals, skel, padding)

%For backwards compatability: vals can either be the x,y,z vals for each
%segment that have been pre-computed (new version)
%Or they can be the channels of rotation values (old version)
if isvector(vals)
  %just the channels have been passed in; need to convert
  if nargin<4
    padding = 0;
  end
  vals = [vals zeros(1, padding)];
  vals = exp2xyz(skel, vals);
end

%Now we assume x,y,z have already been calculated
%And passed in as vals (#segments x 3)
connect = skelConnectionMatrix(skel);

indices = find(connect);
[I, J] = ind2sub(size(connect), indices);

set(handle(1), 'Xdata', vals(:, 1), 'Ydata', vals(:, 3), 'Zdata', ...
    vals(:, 2));
for i = 1:length(indices)
    set(handle(i+1), 'Xdata', [vals(I(i), 1) vals(J(i), 1)], ...
        'Ydata', [vals(I(i), 3) vals(J(i), 3)], ...
        'Zdata', [vals(I(i), 2) vals(J(i), 2)]);
end