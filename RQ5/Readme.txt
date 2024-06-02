**************************************************************************************************************************
In order to apply different methods of calculating thresholds, it is necessary to find the site packages
folder in the virtual environment used in this code on the local computer, find the cleanlab folder, and 
modify the threshold calculation method of the computeconfident_joint function in the latent_estimation. py file 
**************************************************************************************************************************

1) Average 
thresholds=[np. mean (psx [:, k] [s==k]) for k in range (K)] 
2) Median 
thresholds=[np. medium (psx [:, k] [s==k]) for k in range (K)] 
3) Upper quartile 
thresholds=[np. percentile (psx [:, k] [s==k], q=75) for k in range (K)] 
4) Lower quartile 
thresholds=[np. percentile (psx [:, k] [s==k], q=25) for k in range (K)]