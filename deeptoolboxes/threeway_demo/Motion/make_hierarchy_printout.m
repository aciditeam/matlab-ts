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
% Given skeleton, write out hierarchy

if strcmp(skel.type,'mit')

%MIT skel doesn't have segment names

skel.tree(1).name = 'pelvis';
skel.tree(2).name = 'lfemur';
skel.tree(3).name = 'ltibia';
skel.tree(4).name = 'lfoot';
skel.tree(5).name = 'ltoes';
skel.tree(6).name = 'rfemur';
skel.tree(7).name = 'rtibia';
skel.tree(8).name = 'rfoot';
skel.tree(9).name = 'rtoes';
skel.tree(10).name = 'thorax';
skel.tree(11).name = 'lclavicle';
skel.tree(12).name = 'lhumerus';
skel.tree(13).name = 'lradius';
skel.tree(14).name = 'lhand';
skel.tree(15).name = 'rclavicle';
skel.tree(16).name = 'rhumerus';
skel.tree(17).name = 'rradius';
skel.tree(18).name = 'rhand';


%Want to know which dimensions correspond to which segments in skeleton
indx = [   1:6 7:9 14 19:21 26 31:33 38 43:45 50 55:57 61:63 67:69 ...
    73:75 79:81 85:87 91:93 97:99 103:105 ];
dim = 1;
for jj = indx
  fprintf('dim %i, index into original %i:',dim,indx(dim));
  for ii = 1:length(skel.tree)
    %try to find where this is in the tree
    pos = find(skel.tree(ii).or==indx(dim));
    if pos
      fprintf('Segment# %i (%s), rotation %i\n',ii,skel.tree(ii).name,pos);
    else
      pos = find(skel.tree(ii).offset==indx(dim));
      if pos
              fprintf('Segment# %i (%s), offset %i\n',ii,skel.tree(ii).name,pos);
      end
    end
  end
  dim = dim+1;
end

elseif strcmp(skel.type,'acclaim')
  
 indx = [ 1:6 ...        %root (special representation)
   10:12 13 16:18 19 ... %lfemur ltibia lfoot ltoes
   25:27 28 31:33 34 ... %rfemur rtibia rfoot rtoes
   37:39 40:42 43:45 46:48 49:51 52:54 ... %lowerback upperback thorax lowerneck upperneck %head
   58:60 61 65 67:69 73:75 ... %(lclavicle ignored) lhumerus lradius lwrist lhand (fingers are constant) lthumb
   79:81 82 86 88:90 94:96 ];  %(rclavicle ignored) rhumerus rradius rwrist rhand (fingers are constant) rthumb    

 dim = 1;
for jj = indx
  fprintf('dim %i, index into original %i:',dim,indx(dim));
  for ii = 1:length(skel.tree)
    %try to find where this is in the tree
    pos = find(skel.tree(ii).expmapInd==indx(dim));
    if pos
      fprintf('Segment# %i (%s), rotation %i\n',ii,skel.tree(ii).name,pos);
    else
      pos = find(skel.tree(ii).posInd==indx(dim));
      if pos
              fprintf('Segment# %i (%s), offset %i\n',ii,skel.tree(ii).name,pos);
      end
    end
  end
  dim = dim+1;
end
 
else
  error('Unknown skeleton type');
end
  