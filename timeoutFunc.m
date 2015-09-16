function varargout = timeoutFunc(f, t, varargin)
%
% [out1, out2, ...] = timeout(f, t, arg1, arg2, ...)
%
% INPUTS:
%    f    : function or function handle.
%    t    : length of time before timeout
%    arg1 : first input to f
%    ...
%    argN : Nth input to f
%
% OUTPUTS:
%    out1 : first output from f
%    ...
%    outM : Mth output from f
%
% This function is used to evaluate a function but with a given time
% constraint.  Therefore the example
%
%   [a, b] = timeout(f, 3, X, Y, Z)
%
% is equivalent to evaluating
%
%   [a, b] = f(X, Y, Z)
%
% except that the first call will throw an error if the evaluation does not
% complete before (approximately) 3 seconds.  Note that this function
% relies on the Parallel Computing Toolbox, specifically the functions
% BATCH and WAIT.
%
% If the evaluation contains an error, the traceback will probably contain
% references to these functions and probably others.
%

% Version information:
%  2013-03-27 @dalle   : Version 1.0
%
% List of aliases:
%  @dalle   : Derek J. Dalle <derek.dalle@gmail.com>

% Copyright © 2013 by Derek J. Dalle
% 
% Permission is hereby granted, free of charge, to any person obtaining a
% copy of this software and associated documentation files (the 
% "Software"), to deal in the Software without restriction, including 
% without limitation the rights to use, copy, modify, merge, publish, 
% distribute, sublicense, and/or sell copies of the Software, and to permit
% persons to whom the Software is furnished to do so, subject to the
% following conditions:
%
% The above copyright notice and this permission notice shall be included 
% in all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
% NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
% DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
% OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR
% THE USE OR OTHER DEALINGS IN THE SOFTWARE.

% Check the local cluster.
c = parcluster();
% Create a job.
j = batch(c, f, nargout, varargin);

% Call the job and block input for a period.
wait(j, 'finished', t)

% Check for completion.
if isempty(j.FinishTime)
	% Delete the job.
	delete(j)
	% Throw a timeout error.
	error('MATLAB:timeout', 'Evaluation timed out.');
else
	% Get the outputs from the batch.
	r = fetchOutputs(j);
	% Delete the job now that outputs have been collected.
	delete(j)
	% Extract the outputs.
	if numel(r) < nargout
		% Too many outputs.
		error('MATLAB:maxlhs', 'Too many outputs.')
	else
		% Assign first outputs.
		varargout = r(1:nargout);
	end
end
