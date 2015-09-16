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
% Train a factored, conditional RBM with three-way interactions
% On labeled mocap data (10 styles, CMU subject 137)
%
% demonstrates a trained model
% generating different styles based on initialization
% dataset we use is the "clipped" CMU 137 style walks
% Assumes that demo_train.m has already been run and a model has been
% saved to the "snapshots" directory

%initialization frames for each style
%good init positions (not in flat points) local to each seq
inits = [200,1,191,270,400,200,78,773,138,192];
%starting frame of first sequence of styles 1:10
seqstarts = [5872,17744,9292,12628,15208,21338,24218,27008,1,3031];
initframes = seqstarts + inits;

numstyles = size(initframes,2);
styledesc = {'cat','chicken','dinosaur','drunk','gangly','graceful' ...
  'normal','old man','sexy','strong'};

%initialize RAND,RANDN to a different state
% rand('state',sum(100*clock))
% randn('state',sum(100*clock))
rand('state',0)
randn('state',0)

%Our important Motion routines are in a subdirectory
addpath('./Motion')
% addpath('./filter') %filtering stuff from SP toolbox

%filtering parameters
% r = 4; %Factor which to reduce sample rate
% Ord = 8; %Order
% R = .05; %R decibels of peak-to-peak ripple in the passband
% Wn = .8/r; %Cutoff frequency
% [bb,aa] = cheby1(Ord,R,Wn);

%load up the training data & preprocess (for initialization and playback)
make_balanced_clipped_137

%downsample to 60fps (make sure model is compatible)
for ii=1:length(Motion)
    Motion{ii}=Motion{ii}(1:2:end,:);
    %remember to downsample labels too
    Labels{ii}=Labels{ii}(1:2:end,:);
end

fprintf(1,'Preprocessing data \n');

%Load in the layer 1 model (before pre-processing)
%We will read the order from here
load snapshots/cmustyle_600hid_200fac_100feat_cd10_12taps_sharefeatfac_ep200.mat

%variable names needed for generation
[numdims, numfac] = size(visfac);
[numhid, junk] = size(hidfac);

n1 = nt; %preprocess2 looks at n1

%Run the 1st stage of pre-processing
%This converts to body-centered coordinates, and converts to ground-plane
%differences
preprocess1

%Run the 2nd stage of pre-processing
%This drops the zero/constant dimensions and builds mini-batches
preprocess2
numdims = size(batchdata,2); %data (visible) dimension

labeldata = [];
for jj=1:length(Labels)
  labeldata = [labeldata; Labels{jj}];
end

initdata = batchdata;

%how many frames to generate (per sequence)
numframes = 400;

%set up figure for display
close all; h = figure(2); 
p = get(h,'Position'); p(3) = 2*p(3); %double width
set(h,'Position',p);


for style = 1:numstyles
  fprintf('Generating %d frames from style ''%s''\n', ...
    numframes,styledesc{style});
  fr = initframes(style);
  labels = zeros(numframes,numstyles,'single');
  labels(:,style) = 1; %set the correct style label to be on
  
  %make sure labels are not being set in gen
  gen_sharefeatfac;
  %show hidden units
  subplot(1,2,1); imagesc(poshidprobs'); colormap gray; axis off
  
%   %smooth with filter
%   %perform filtering in each dimension
%   for jj=1:size(visible,2)
%     visible(:,jj) = filtfilt(bb,aa,visible(:,jj));
%   end

  %show movie
  postprocess
  subplot(1,2,2); 
  expPlayData(skel,newdata(:,:),1/60);
end
