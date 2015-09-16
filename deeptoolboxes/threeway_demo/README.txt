Version 0.100 (Unsupported, unreleased)

Code provided by Graham Taylor and Geoff Hinton

For more information, see:
    http://www.cs.toronto.edu/~gwtaylor/publications/icml2009
Permission is granted for anyone to copy, use, modify, or distribute this
program and accompanying programs and documents for any purpose, provided
this copyright notice is retained and prominently displayed, along with
a note saying that the original programs are available from our
web page.
The programs and documents are distributed without any warranty, expressed or
implied.  As the programs were written for research purposes only, they have
not been tested to the degree that would be advisable in any important
application.  All use of these programs is entirely at the user's own risk.

This subdirectory contains files related to learning and generation:

demo_train.m	    Main file for training a model
demo_generate.m     Main file for generating from a trained model

Note that, by default, we train a factored conditional RBM with 
three-way interactions whose feature-factor weights are shared between
the three "sub-models" (see the paper for the definition of
sub-models). The following learning and generation scrips are called by
the demos:

gaussianfbm_sharefeatfac.m	Trains the model (shared
feature-factor weights)
gen_sharefeatfac.m		Generates from a trained model (shared
feature-factor weights)

However, there are other weight-sharing options. Some of these are
also implemented. They have different learning and generation scripts:

No sharing: gaussianfbm.m, gen.m
Sharing feature-factor, past-factor weights: gaussianfbm_sharefpfac.m,
gen_sharefpfac.m
Sharing feature-factor, past-factor, visible-factor weights:
gaussianfbm_sharefpvfac.m, gen_sharefpvfac.m
Sharing feature-factor, past-factor, visible-factor, hidden-factor
weights (full-sharing): gaussianfbm_shared.m, gen_shared.m
Sharing feature-factor, past-factor, visible-factor weights: gaussian

Also included are learning and generation scripts for a factored
version of the CRBM that does not have any three-way interactions:
gaussianfaccrbm.m, genfaccrbm.m

The Motion subdirectory contains files related to motion capture data: 
preprocessing/postprocessing, playback, etc ...

Acknowledgments

The sample data we have included has been provided by the CMU Graphics
Lab Motion Capture Database:
http://mocap.cs.cmu.edu

Several subroutines related to motion playback are adapted from Neil 
Lawrence's Motion Capture Toolbox:
http://www.cs.man.ac.uk/~neill/mocap/

Several subroutines related to conversion to/from exponential map
representation are provided by Hao Zhang:
http://www.cs.berkeley.edu/~nhz/software/rotations/
