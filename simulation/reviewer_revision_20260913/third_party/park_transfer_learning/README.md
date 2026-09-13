# Transfer learning under large-scale low-rank regression models

## Directory
- [Function](https://github.com/ishspsy/transfer_learning/tree/main/Function): Contains functions that implement transfer learning algorithms and generate synthetic datasets.
- [Data](https://github.com/ishspsy/transfer_learning/tree/main/Data): Includes the CCLE dataset along with accompanying data dictionary files in [Data_dictionary](https://github.com/ishspsy/transfer_learning/tree/main/Data/Data_dictionary) directory.
- [Code_example](https://github.com/ishspsy/transfer_learning/tree/main/Code_example): Includes R scripts that demonstrate the use of functions in the [Function](https://github.com/ishspsy/transfer_learning/tree/main/Function) directory.
- [Results](https://github.com/ishspsy/transfer_learning/tree/main/Results): Includes RData files containing the results.


## Data
The Cancer Cell Line Encyclopedia (CCLE) dataset consists of the expression levels of 19,221 genes and drug response data for 24 compounds across a range of human cancer cell lines, where the drug response of each cell line is measured as the area under the dose-response curve. Cancer cell line datasets, such as CCLE, have been widely used to construct predictive models of drug response. We aim to predict drug response in non-small cell lung cancer (NSCLC) cell lines with mutations in the Kirsten rat sarcoma viral oncogene homolog (KRAS) gene. We focus on a subset of the CCLE data. Specifically, we consider 8 drugs for which drug response is available for all cell lines. These 8 drugs are AZD0530, Crizotinib, Dovitinib, Lapatinib, Nutlin-3, PD0325901, TAE684, and Topotecan. In addition, we select the top 100 genes with the highest variance across KRAS-mutant NSCLC cell lines. Finally, we have 28 KRAS-mutant NSCLC cell lines on 100 predictors and 8 responses.

The main dataset [CCLEdataset.RDATA](https://github.com/ishspsy/transfer_learning/tree/main/Data/CCLEdataset.RDATA) contains four key objects:
- Xtarget: A 28 × 100 matrix of gene expression data for KRAS-mutant NSCLC cell lines.
- Ytarget: A 28 × 8 matrix of drug response values for 8 anti-cancer compounds.
- auXlist: A list of source-domain gene expression datasets (NSCLC Wild-type , Others Mutant, Others Wild-Type), each a matrix of shape n_k × 100, where n_1=47, n_2=68, n_3=326.
- auYlist: A list of corresponding drug response matrices for each source domain (NSCLC Wild-type , Others Mutant, Others Wild-Type), with shape n_k × 8,  where n_1=47, n_2=68, n_3=326.
- Note: The gene variables in auXlist and the drug variables in auYlist are identical to those in Xtarget and Ytarget, respectively, ensuring consistent feature alignment across domains.

All gene and drug identifiers are consistent across target and auxiliary domains.
- For list of genes and their descriptions, see [CCLEdataset$Xtarget_dictionary.txt](https://github.com/ishspsy/transfer_learning/tree/main/Data/Data_dictionary/CCLEdataset$Xtarget_dictionary.txt)
- For list of drugs and their descriptions, see [CCLEdataset$Ytarget_dictionary.txt](https://github.com/ishspsy/transfer_learning/tree/main/Data/Data_dictionary/CCLEdataset$Ytarget_dictionary.txt)

[CCLEdataset_trainingtest.RDATA](https://github.com/ishspsy/transfer_learning/tree/main/Data/CCLEdataset_trainingtest.RDATA) contains 100 randomly generated training-test splits based on the original CCLEdataset. In each split, the 28 KRAS-mutant NSCLC cell lines are randomly partitioned into 20 training samples and 8 test samples. This resampling procedure was repeated 100 times to allow for stable evaluation of prediction error. The resulting object stores the corresponding training and test indices for each repetition.


## Main function
- [Functions_FSDtrans.R](https://github.com/ishspsy/transfer_learning/blob/main/Function/Functions_FSDtrans.R): Function for FSD-Trans-NR, which is based on joint source selection.
- [Functions_MSDtrans.R](https://github.com/ishspsy/transfer_learning/blob/main/Function/Functions_MSDtrans.R): Function for MSD-Trans-NR, which is based on the marginal source selection.
- [Functions_naiveapproaches.R](https://github.com/ishspsy/transfer_learning/blob/main/Function/Functions_naiveapproaches.R): The other competitor methods.
- [Functions_rankestimation_simul.R](https://github.com/ishspsy/transfer_learning/blob/main/Function/Functions_rankestimation_simul.R): The rank estimation process.
- [Functions_transSCAD.R](https://github.com/ishspsy/transfer_learning/blob/main/Function/Functions_transSCAD.R): Function for SCAD-based estimator.

## Code example
- [CCLE_prediction_errors(Section_4).R](https://github.com/ishspsy/transfer_learning/blob/main/Code_example/CCLE_prediction_errors(Section_4).R): Example code for analyzing the Cancer Cell Line Encyclopedia (CCLE) dataset.
- [CCLE_pathway_analysis(Section_S5).R](https://github.com/ishspsy/transfer_learning/blob/main/Code_example/CCLE_pathway_analysis(Section_S5).R): Example code for the pathway analysis of the Cancer Cell Line Encyclopedia (CCLE) dataset.
- [Figures_code.R](https://github.com/ishspsy/transfer_learning/blob/main/Code_example/Figures_code.R): Provides code for creating visualizations used in the analysis.
- [Simulation_rank_estimation(Figure_S7-S10).R](https://github.com/ishspsy/transfer_learning/blob/main/Code_example/Simulation_rank_estimation(Figure_S7-S10).R): Simulation code for the rank estimation procedure corresponding to Figures S7–S10.
- [Simulation_estimation_error(Figure_1).R](https://github.com/ishspsy/transfer_learning/blob/main/Code_example/Simulation_estimation_error(Figure_1).R): Simulation code for the estimation error procedure corresponding to Figure 1.
- [Simulation_estimation_error(Figure_S11).R](https://github.com/ishspsy/transfer_learning/blob/main/Code_example/Simulation_estimation_error(Figure_S11).R): Simulation code for the estimation error procedure corresponding to Figure S11.
- [Simulation_source_detection(Figure_2).R](https://github.com/ishspsy/transfer_learning/blob/main/Code_example/Simulation_source_detection(Figure_2).R): Simulation code for the source detection procedure corresponding to Figure 2.
- [Simulation_source_detection(Figures_S1-S3).R](https://github.com/ishspsy/transfer_learning/blob/main/Code_example/Simulation_source_detection(Figures_S1-S3).R): Simulation code for the source detection procedure corresponding to Figures S1–S3.
- [Simulation_source_detection(Figures_S4-S6).R](https://github.com/ishspsy/transfer_learning/blob/main/Code_example/Simulation_source_detection(Figures_S4-S6).R): Simulation code for the source detection procedure corresponding to Figures S4–S6.

## Required R Environment
- R version: 4.2.0
- Required packages: The following R packages are required to run the proposed transfer learning algorithm. Please install and load them using the code below.
```r
# Install required packages
install.packages(c("corpcor", "foreach", "doParallel", "splitTools", "MASS"))

# Load the packages
library(corpcor)
library(foreach)
library(doParallel)
library(splitTools)
library(MASS)
```

### Contact
ishspsy@yonsei.ac.kr
