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
% Creates a (roughly) balanced dataset
% Based on the 137 style_walks

%10 different walks, loaded into skel,Motion
load Data/137_walks_expmap.mat

% totalframes = 0;
% for ii=1:length(Motion)
%   totalframes = totalframes + size(Motion{ii},1);
%   disp(size(Motion{ii},1));
% end

%to try:
% low-pass filtering here
% integrating the range of motion (137_34.amc)

%now to build a roughly balanced dataset
% approx # frames    sequences    multiplier (total 6000)
% 2000               9,10         3
% 1500               1,3,4,5      4
% 1000               2,6,7        6
% 3000               8            2

%can we repmat cell arrays?
%note that we also add labels
numlabels = length(Motion);

style = 9;
for ii=1:3
  BigMotion{ii} = Motion{style};
  Labels{ii} = zeros(size(BigMotion{ii},1),numlabels,'single');
  Labels{ii}(:,style)=1;
end
style = 10;
for ii=4:6
  BigMotion{ii} = Motion{style};
  Labels{ii} = zeros(size(BigMotion{ii},1),numlabels,'single');
  Labels{ii}(:,style)=1;
end
style = 1;
for ii=7:10
  BigMotion{ii} = Motion{style};
  Labels{ii} = zeros(size(BigMotion{ii},1),numlabels,'single');
  Labels{ii}(:,style)=1;
end 
style = 3;
for ii=11:14
  BigMotion{ii} = Motion{style};
  Labels{ii} = zeros(size(BigMotion{ii},1),numlabels,'single');
  Labels{ii}(:,style)=1;
end 
style = 4;
for ii=15:18
  BigMotion{ii} = Motion{style};
  Labels{ii} = zeros(size(BigMotion{ii},1),numlabels,'single');
  Labels{ii}(:,style)=1;
end 
style = 5;
for ii=19:22
  BigMotion{ii} = Motion{style};
  Labels{ii} = zeros(size(BigMotion{ii},1),numlabels,'single');
  Labels{ii}(:,style)=1;
end 
style = 2;
for ii=23:28
  BigMotion{ii} = Motion{style};
  Labels{ii} = zeros(size(BigMotion{ii},1),numlabels,'single');
  Labels{ii}(:,style)=1;
end 
style = 6;
for ii=29:34
  BigMotion{ii} = Motion{style};
  Labels{ii} = zeros(size(BigMotion{ii},1),numlabels,'single');
  Labels{ii}(:,style)=1;
end 
style = 7;
for ii=35:40
  BigMotion{ii} = Motion{style};
  Labels{ii} = zeros(size(BigMotion{ii},1),numlabels,'single');
  Labels{ii}(:,style)=1;
end 
style = 8;
for ii=41:42
  BigMotion{ii} = Motion{style};
  Labels{ii} = zeros(size(BigMotion{ii},1),numlabels,'single');
  Labels{ii}(:,style)=1;
end 


% totalframes = 0;
% for ii=1:length(BigMotion)
%   totalframes = totalframes + size(BigMotion{ii},1);
%   disp(size(BigMotion{ii},1));
% end

Motion = BigMotion; 
clear BigMotion;