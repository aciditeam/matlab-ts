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

clear all; close all;
more off;   %turn off paging

%initialize RAND,RANDN to a different state
rand('state',sum(100*clock))
randn('state',sum(100*clock))

%Our important Motion routines are in a subdirectory
addpath('./Motion')

%set up training data
make_balanced_clipped_137  %(originally came from bvh files)

%downsample here from 120fps to 60fps
for ii=1:length(Motion)
    Motion{ii}=Motion{ii}(1:2:end,:);
    %remember to downsample labels too
    Labels{ii}=Labels{ii}(1:2:end,:);
end

fprintf(1,'Preprocessing data \n');

%Run the 1st stage of pre-processing
%This converts to body-centered coordinates, and converts to ground-plane
%differences
preprocess1

%how-many timesteps do we look back for directed connections
%this is what we call the "order" of the model 
n1 = 12; 
        
%Run the 2nd stage of pre-processing
%This drops the zero/constant dimensions and builds mini-batches
preprocess2
numdims = size(batchdata,2); %data (visible) dimension

labeldata = [];
for jj=1:length(Labels)
  labeldata = [labeldata; Labels{jj}];
end

initdata = batchdata;

%Training properties
numhid1 = 600;
%There are three types of factors
%But for now, we will set them all to be size numfac
numfac = 200; 
numfeat = 100; %number of distributed "style" features
maxepoch = 200;
cdsteps = 10;
pastnoise = 1;

%every xxx epochs, write a snapshot of the model
%will be written to snapshot_path_epxxx.mat
snapshot_path = 'snapshots/cmustyle_600hid_200fac_100feat_cd10_12taps_sharefeatfac'
%snapshot_path = 'default' %don't overwrite our models

snapshotevery=100; %write out a snapshot of the weights every xx epochs

nt=n1;
numhid=numhid1;

fprintf(1,'Training Layer 1 FBM, order %d: %d-%d(%d) \n',nt,numdims, ...
  numhid,numfac);
restart=1;      %initialize weights

%train network with only feature-to-factor weights tied
gaussianfbm_sharefeatfac

fprintf(1,'Training finished. Run demo_generate to generate data\n');
