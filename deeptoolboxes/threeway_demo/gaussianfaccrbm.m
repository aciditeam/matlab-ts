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
% This program trains a model that is identical to a standard crbm with
% Gaussian visible units but the weights are factored no three-way
% interactions does not use any label data

%batchdata is a big matrix of all the frames
%we index it with "minibatch", a cell array of mini-batch indices
numbatches = length(minibatch); 

numdims = size(batchdata,2); %visible dimension

%Setting learning rates
%Corresponding to the "undirected" observation model
epsilonvisfac=single(1e-3);
epsilonhidfac=single(1e-3);

%Corresponding to the "directed" Autoregressive model
epsilonpastfacA=single(1e-5);
epsilonvisfacA=single(1e-5);

%Corresponding to the "directed" past->hidden model
epsilonpastfacB=single(1e-3);
epsilonhidfacB=single(1e-3);

epsilonvisbias=single(1e-3);
epsilonhidbias=single(1e-3);

%currently we use the same weight decay for all weights
%but no weight decay for biases
wdecay = single(0.0002);

mom = single(0.9);       %momentum used only after 5 epochs of training

if restart==1,  
  restart=0;
  epoch=1;
 
  %weights  
  visfac = single(0.01*randn(numdims,numfac));  
  hidfac = single(0.01*randn(numhid,numfac));
    
  %Note the new parameterization of pastfac:
  %First numdims rows correspond to time t-nt
  %Last numdims rows correspond to time t-1
  pastfacA = single(0.01*randn(nt*numdims,numfac));   
  visfacA = single(0.01*randn(numdims,numfac));
  
  pastfacB = single(0.01*randn(nt*numdims,numfac));  
  hidfacB = single(0.01*randn(numhid,numfac));      
    
  %biases
  visbiases = zeros(1,numdims,'single');
  hidbiases = zeros(1,numhid,'single');
  %vishid = 0.01*randn(numdims,numhid);    

  %keep previous updates around for momentum
  visfacinc = zeros(size(visfac),'single');  
  hidfacinc = zeros(size(hidfac),'single');
  
  pastfacAinc = zeros(size(pastfacA),'single');
  visfacAinc = zeros(size(visfacA),'single');  
  
  pastfacBinc = zeros(size(pastfacB),'single');
  hidfacBinc = zeros(size(hidfacB),'single');  
     
  visbiasinc = zeros(size(visbiases),'single');
  hidbiasinc = zeros(size(hidbiases),'single');
  %vishidinc = zeros(size(vishid));    
end

%Main loop
for epoch = epoch:maxepoch,
  errsum=0; %keep a running total of the difference between data and recon
  for batch = 1:numbatches,     

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    numcases = length(minibatch{batch});
    mb = minibatch{batch}; %caches the indices       
    
    past = zeros(numcases,nt*numdims,'single'); %initialization
    
    data = single(batchdata(mb,:));
    %use pastindex to index the appropriate frames in batchdata
    %(for each frame in the minibatch) depending on the delay
    %past = reshape(batchdata(pastindex,:),numcases,nt*numdims);    
    %past = batchdata(mb-1,:); %one step in the past
    
    %Easiest way to build past is by a loop
    %Past looks like [ [data time t-nt] ... [data time t-1] ] 
    for hh=nt:-1:1 %note reverse order
      past(:,numdims*(nt-hh)+1:numdims*(nt-hh+1)) = batchdata(mb-hh,:) + randn(numcases,numdims);
    end      
    
    %calculate inputs to factors (will be used many times)
    yvis = data*visfac; %summing over numdims
            
    ypastA = past*pastfacA;     %summing over nt*numdims
    yvisA = data*visfacA;       %summing over numdims
    
    ypastB = past*pastfacB;     %summing over nt*numdims            

    
    %pass 3-way term + gated biases + hidbiases through sigmoid 
    poshidprobs = 1./(1 + exp(-yvis*hidfac'  ...
      -ypastB*hidfacB' - repmat(hidbiases,numcases,1)));
      %-data*vishid - repmat(hidbiases,numcases,1)));  
    
    %Activate the hidden units    
    hidstates = single(poshidprobs > rand(numcases,numhid));
    
    yhid = hidstates*hidfac;
    yhid_ = poshidprobs*hidfac; %smoothed version
    
    yhidB_ = poshidprobs*hidfacB; %smoothed version  
    
    %these are used multiple times, so cache
    %yvishid_ = yvis.*yhid_;
    %yvispastA = yvisA.*ypastA;
    %ypasthidB_ = ypastB.*yhidB_;
                    
    
    %Calculate statistics needed for gradient update
    %Gradients are taken w.r.t neg energy
    %Note that terms that are common to positive and negative stats
    %are left out
    posvisprod = data'*yhid_; %smoothed
    poshidprod = poshidprobs'*yvis; %smoothed
    
    posvisAprod = data'*ypastA;
    pospastAprod =  past'*yvisA;
   
    pospastBprod = past'*yhidB_; %smoothed    
    poshidBprod =  poshidprobs'*ypastB;
                  
    %posvishidprod = data'*poshidprobs;
    posvisact = sum(data,1);
    poshidact = sum(poshidprobs,1);  %smoothed             
    
%%%%%%%%% END OF POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

  for cdn = 1:cdsteps    
    %Activate the visible units
    %Collect 3-way terms + vis biases + gated biases 
    %note use of stochastic hidstates
    %Mean-field version (do not add Gaussian noise)        
    negdata = yhid*visfac' + ...
      ypastA*visfacA' + ...
      repmat(visbiases,numcases,1);    
    
    yvis = negdata*visfac;   
    
    %pass 3-way term + gated biases + hidbiases through sigmoid 
    neghidprobs = 1./(1 + exp(-yvis*hidfac'  ...
      -ypastB*hidfacB' - repmat(hidbiases,numcases,1)));

    if cdn == 1
      %Calculate reconstruction error
      err= sum(sum( (data(:,:,1)-negdata).^2 ));
      errsum = err + errsum;
    end
 
    if cdn == cdsteps     
      yhidB_ = neghidprobs*hidfacB; %smoothed version 
      yhid_ = neghidprobs*hidfac; %smoothed version
      %yvishid_ = yvis.*yhid_;
      yvisA = negdata*visfacA;       %summing over numdims
      %yvispastA = yvisA.*ypastA;
      %ypasthidB_ = ypastB.*yhidB_;
      
      %last cd step -- Calculate statistics needed for gradient update
      %Gradients are taken w.r.t neg energy
      %Note that terms that are common to positive and negative stats
      %are left out
      negvisprod = negdata'*yhid_; %smoothed
      neghidprod = neghidprobs'*yvis; %smoothed

      negvisAprod = negdata'*ypastA;
      negpastAprod =  past'*yvisA;

      negpastBprod = past'*yhidB_; %smoothed
      neghidBprod =  neghidprobs'*ypastB;
      
      %negvishidprod = data'*neghidprobs;
      negvisact = sum(negdata,1);
      neghidact = sum(neghidprobs,1);  %smoothed

    else
      %Stochastically sample the hidden units
      hidstates = single(neghidprobs > rand(numcases,numhid));      
      yhid = hidstates*hidfac;
    end 
  end
  
%%%%%%%%% END NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    if epoch > 5 %use momentum
        momentum=mom;
    else %no momentum
        momentum=0;
    end
    
%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
visfacinc = momentum*visfacinc + ...
  epsilonvisfac*( ( posvisprod - negvisprod)/numcases - wdecay*visfac);
hidfacinc = momentum*hidfacinc + ...
  epsilonhidfac*( (poshidprod - neghidprod)/numcases - wdecay*hidfac);

visfacAinc = momentum*visfacAinc + ...
  epsilonvisfacA*( (posvisAprod - negvisAprod)/numcases - wdecay*visfacA);
pastfacAinc = momentum*pastfacAinc + ...
  epsilonpastfacA*( (pospastAprod - negpastAprod)/numcases - wdecay*pastfacA);

hidfacBinc = momentum*hidfacBinc + ...
  epsilonhidfacB*( (poshidBprod - neghidBprod)/numcases - wdecay*hidfacB);
pastfacBinc = momentum*pastfacBinc + ...
  epsilonpastfacB*( (pospastBprod - negpastBprod)/numcases - wdecay*pastfacB);

visbiasinc = momentum*visbiasinc + ...
  (epsilonvisbias/numcases)*(posvisact - negvisact);
hidbiasinc = momentum*hidbiasinc + ...
  (epsilonhidbias/numcases)*(poshidact - neghidact);

visfac = visfac + visfacinc;
hidfac = hidfac + hidfacinc;

visfacA = visfacA + visfacAinc;
pastfacA = pastfacA + pastfacAinc;

hidfacB = hidfacB + hidfacBinc;
pastfacB = pastfacB + pastfacBinc;

visbiases = visbiases + visbiasinc;
hidbiases = hidbiases + hidbiasinc;
    
%%%%%%%%%%%%%%%% END OF UPDATES  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
  end
    
  %every 10 epochs, show output
  if mod(epoch,10) ==0
      fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
      
      if mod(epoch,100)==0
      
      %show hiddens      
      plotindex = 101:500; %frames of batchdata that we will plot
      nc = length(plotindex);
      
      data = batchdata(plotindex,:);      
      
      past = zeros(nc,nt*numdims); %initialization
      for hh=nt:-1:1 %note reverse order
        %past(:,numdims*(hh-1)+1:numdims*hh) = initdata(hh:end-(nt-hh+1),:);
        past(:,numdims*(nt-hh)+1:numdims*(nt-hh+1)) = batchdata(plotindex-hh,:);
      end
         
      yvis = data*visfac; %summing over numdims

      ypastB = past*pastfacB;     %summing over nt*numdims

      %pass 3-way term + gated biases + hidbiases through sigmoid
      poshidprobs = 1./(1 + exp(-yvis*hidfac'  ...
        -ypastB*hidfacB' - repmat(hidbiases,nc,1)));

      sfigure(32); imagesc(poshidprobs'); colormap gray; axis off;      

      yhid_ = poshidprobs*hidfac; %smoothed version
      ypastA = past*pastfacA;     %summing over nt*numdims
      
      %look at mean-field reconstruction
      negdata = yhid_*visfac' + ...
        ypastA*visfacA' + ...
        repmat(visbiases,nc,1);      
      
      sfigure(33);clf
      subplot(2,1,1); plot(data(:,7)); hold on; plot(negdata(:,7),'r');
      subplot(2,1,2); plot(data(:,18)); hold on; plot(negdata(:,18),'r');
      
      %sfigure(34); imagesc(labelfeat); colormap gray; axis off         
      
      %Hinton plots of parameters
      %Likely do not want to plot all dims, all hiddens, all factors     
      maxdims = min(30,numdims); maxhid = min(100,numhid); 
      maxfac = min(50,numfac);
      maxpast = min(2,nt); %how many time steps in past to plot (pastfac)
      
      %undirected model
      sfigure(35); 
      subplot(2,1,1); hinton(visfac(1:maxdims,1:maxfac));
      subplot(2,1,2); hinton(hidfac(1:maxhid,1:maxfac));
      set(gcf,'Name','undirected')      
           
      %autoregressive model
      sfigure(36);
      %for past, we only want to plot maxdims & maxpast
      %i don't know how to do this without a loop
      pastrows = [];
      for kk=maxpast:-1:1 %note reverse
        %select maxdims rows corresponding to time step kk
        pastrows = [pastrows; pastfacA(end-kk*numdims+1:...
          end-kk*numdims+maxdims, 1:maxfac)];
      end      
      subplot(2,1,1); hinton(pastrows);
      subplot(2,1,2); hinton(visfacA(1:maxdims,1:maxfac));
      set(gcf,'Name','autoregressive')           
      
      %directed vis -> hid model
      sfigure(37);
      %see comment above
      pastrows = [];
      for kk=maxpast:-1:1 %note reverse
        %select maxdims rows corresponding to time step kk
        pastrows = [pastrows; pastfacB(end-kk*numdims+1:...
          end-kk*numdims+maxdims, 1:maxfac)];
      end     
      subplot(2,1,1); hinton(pastrows);
      subplot(2,1,2); hinton(hidfacB(1:maxhid,1:maxfac));
      set(gcf,'Name','directed')   
      
      %biases
      sfigure(34);
      subplot(2,1,1); hinton(visbiases(1:maxdims));
      subplot(2,1,2); hinton(hidbiases(1:maxhid));
      set(gcf,'Name','biases')
      
      
%       %Could see a plot of the weights every 10 epochs
%       sfigure(33);
%       subplot(2,3,1); hinton(visfac);
%       subplot(2,3,2); hinton(pastfac);
%       subplot(2,3,3); hinton(hidfac);
%       subplot(2,3,4); hinton(vishid);
%       subplot(2,3,5); hinton(visbiases);
%       subplot(2,3,6); hinton(hidbiases);
%       drawnow;
%       sfigure(34);
%       subplot(3,1,1); imagesc(data'); colormap gray; axis off
%       subplot(3,1,2); imagesc(poshidprobs',[0 1]); colormap gray; axis off
%       subplot(3,1,3); imagesc(negdata',[0 1]); colormap gray; axis off
%       drawnow;
      %figure(3); weightreport
      %drawnow;
      end      
  end
  %Checkpoint models
  if mod(epoch,snapshotevery) ==0
    snapshot_file = [snapshot_path '_ep' num2str(epoch) '.mat'];
    save(snapshot_file, 'visfac','hidfac', ...
      'pastfacA','visfacA', ...
      'pastfacB','hidfacB', ...
      'visbiases','hidbiases', ...
      'cdsteps', 'numhid','numfac','epoch', 'nt');
  end

  drawnow; %update any plots
end