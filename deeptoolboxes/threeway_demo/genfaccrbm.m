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
% This program uses a factored, conditional RBM (no three-way
% interactions, hence, no labels) to generate data
%
% The program assumes that the following variables are set externally:
% numframes    -- number of frames to generate
% fr           -- a starting frame from initdata (for initialization)

numGibbs = 100; %number of alternating Gibbs iterations 
initNoise = 0.05;  %sd of Gaussian noise added to initialization

numdims = size(initdata,2);

%initialize visible layer
visible = zeros(numframes,numdims,'single');
visible(1:n1,:) = initdata(fr:fr+n1-1,:);
%initialize hidden layer
poshidprobs = zeros(numframes,numhid,'single');
hidstates = ones(numframes,numhid,'single');

%initialize
past = zeros(1,nt*numdims,'single');

for tt=n1+1:numframes
  
  %initialize using the last frame + noise
  visible(tt,:) = visible(tt-1,:) + initNoise*randn(1,numdims);
  
  %Easiest way to build past is by a loop
  %Past looks like [ [data time t-nt] ... [data time t-1] ]
  for hh=nt:-1:1 %note reverse order
    past(:,numdims*(nt-hh)+1:numdims*(nt-hh+1)) = visible(tt-hh,:);
    %Cheat and use initdata instead of generated data
    %Note the conversion to the absolute fr scale instead of the relative
    %tt scale
    %past(:,numdims*(nt-hh)+1:numdims*(nt-hh+1)) = initdata(fr+tt-hh-1,:);
  end
  
  %Input from past does not change during Alternating Gibbs 
  %Set these now and leave them

  %autoregressive model
  ypastA = past*pastfacA;
  
  %directed vis-hid model
  ypastB = past*pastfacB;

  %constant term during inference
  %(not dependent on visibles)
  constinf = -ypastB*hidfacB' - hidbiases;
  
  %constant term during reconstruction
  %(not dependent on hiddens)
  constrecon = ypastA*visfacA' + visbiases;
  
  %MAIN Gibbs sampling loop

  for gg = 1:numGibbs

    yvis = visible(tt,:)*visfac;    
    
    %pass through sigmoid    
    %only part from "undirected" model changes
    %small fix for denormalized numbers
    poshidprobs(tt,:) = 1./(1 + exp(-(yvis+1e-20)*hidfac' + constinf));       
        
    %Activate the hidden units
    hidstates(tt,:) = single(poshidprobs(tt,:) > rand(1,numhid));

    yhid = hidstates(tt,:)*hidfac;
    
    %NEGATIVE PHASE
    %Don't add noise at visibles
    %Note only the "undirected" term changes
    visible(tt,:) = (yhid+1e-20)*visfac' + constrecon;    
    
%         sfigure(3); imagesc(hidstates'); colormap gray; axis off
%         drawnow;
%     pause(0.5);

  end

  %Now do mean-field
  yhid_ = poshidprobs(tt,:)*hidfac; %smoothed version

  %Mean-field approx
  visible(tt,:) = yhid_*visfac' + constrecon;    
  
end
