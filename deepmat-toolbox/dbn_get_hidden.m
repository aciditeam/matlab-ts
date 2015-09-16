% dbn - training a DBN with up-down algorithm
% Copyright (C) 2011 KyungHyun Cho, Tapani Raiko, Alexander Ilin
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%
function [htop] = dbn_get_hidden(v0, D)

n_layers = length(D.structure.layers);
% Init result activations
h0r = cell(n_layers-1, 1);
h0r{1} = v0;
if D.learning.ffactored == 0
    h0r{1} = binornd(1, h0r{1});
end
% Go through the recognition part of the DBN
for l = 2:(n_layers-1)
    h0r{l} = sigmoid(bsxfun(@plus, h0r{l-1} * D.rec.W{l-1}, D.rec.biases{l}'));
    if D.learning.ffactored == 0
        h0r{l} = binornd(1, h0r{l});
    end
end
vtop = h0r{end};
htop = sigmoid(bsxfun(@plus, vtop * D.top.W, D.top.hbias'));
end

