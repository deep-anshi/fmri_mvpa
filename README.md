# fmri_mvpa

This project was done by
Heejung Jung
Xiaochun Han
Deepanshi Shokeen

Project: MVPC – classify behaving animal data in searchlights using SVM
P1. 1 out of 20 conditions
P2. Classify taxa (1 out of 5), training on videos with 3 behaviors and testing on videos with the left-out behavior.
P3. Classify behaviors (1 out of 4), training on videos with 4 taxa and testing on videos with left-out taxonomic category
Parameters:
The 3 analyses (P1, P2, P3) were conducted separately for the attention and behavioral task.
search light radii: 10
Analysis involved
GLM: modeled 20 separate regressors with nuisance regressors derived from fmriprep
searchlight: using a 10 radius, pattern classification was performed
using a linear SVM classifier within a surface-based searchlight, conducted per participant.
group average searchlight one-sample t-test & two-sample t-test

1) one-sample t-test

Group average searchlight maps were compared against the chance performance
chance performance was 0.05 for P1, 0.2 for P2, 0.25 for P3
2) two-sample t-test

For example, for the behavior classification for beh attention > tax attention task
We hypothesized that the behavioral classification performance would be higher for the behavior-attention task, as opposed to the taxonomy-attention task.
*Note that we used a one-sided uncorrected p-value of 0.05.

visualization of group average searchlight one-sample t-test & two-sample t-test

Explanation about the 1 - cross validation
We realized in that in our code, we simply ran cv = mv.CrossValidation(clf, mv.NFoldPartitioner(attr=chunks)).
