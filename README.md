# WM_Psilocybin
Spatial WM task for Psilocybin study w/ Manoj Doss

Tasks:
1. PRACTICE_WMpsilocybin_v1.psyexp - 10 trial practice task to use during orientation day for new subjects.
2. WMpsilocybin_v6.psyexp - main task that should be run for each session. Inlcudes a few practice trials to re-familiarize subjects, then moves into main-task trials.

BE SURE TO USE THE SAME SUBJECT ID FOR ALL TASKS

Important notes:
In the PsychoPy builder, there is a routine called "init" which includes the initilization code to create a subject-specific design matrix at the start of the experiment and save it out as a csv.

Currently, there are "4" practice trials and "125" main-task trials.
In the code in "init" (Begin Routine), there are variables representing the number of practice and main-task trials that can easily be changed to fit timing constraints.

In the same section of code mentioned above there is a variable controlling the minimum degree distance there can be betweend dots in the memory array (""MIN_SEP_DEG"). This can also be adjusted by changing the number.