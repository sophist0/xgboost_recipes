##########################################################
# Running scripts
##########################################################

(1) Explore the dataset by running

	python3 explore_data.py

(2) Partition the Kaggle training dataset into training set and a testing set by running

	python3 
	import project_lib as pl
	pl.part_data()

(3) For parameter tuning run. Set the parameters to sweep in param_tuning.py under heading "Parameters to sweep". Then run

	python3 param_tuning.py

(4) Print the most accurate models parameters to screen and CSV best_results.csv

	python3
	import project_lib as pl
	pl.write_sweep_results()

(5) To test a model on the test set. Open test_model.py and under the heading "Parameters" set the parameters for the model you want to test and whether the output should be a ROC plot or the accuracy. If run to produce ROC saves ROC curve as ~/data/used/roc_MODELNAME.p where MODELNAME is the string set by variable mname.

(6) Plot all ROC curves. Currently hard coded to plot ROC's for S2M1, S2M2, S2M3, S2M4. Run the following, saves figure in ~/figures

	python3
	import project_lib as pl
	pl.plot_roc_results()

(7) Test feature selection policies. Use variable "order" to select the feature selection policy and set the model parameters. Run by

	python3 fselect.py

The results are saved as ~/data/used/features_order_NUM.p where NUM is the order selected above. 

(8) Plot the results of fselect.py for all four selection policies

	python3
	import project_lib as pl
	pl.plot_fselect()


##########################################################
# Figures generated ~/figures
##########################################################

cuisine_num_recipes.png		->	bar plot of number of recipes in each cuisine

fselect.png			-> 	model accuracy as a function of various feature selection policies

ingredients_num_recipes.png	->	bar plot of number of times popular ingredients are included in a recipe

rocs.png			->	plot of ROC curves


##########################################################
# Data files (original and generated)	~/data/used/
##########################################################

best_results.csv	-> 	parameters for the most accurate models in the parameter sweep

features_order_1.p	-> 	results for selecting the most frequent ingredients

features_order_2.p	-> 	results for selecting the most frequent ingredients for each cuisine

features_order_3.p	-> 	results for selecting the ingredients in the fewest cuisines but are one of the most frequent ingredients in a cuisine

features_order_4.p	-> 	results for selecting the ingredients in the most cuisines but are one of the most frequent ingredients in a cuisine

label_testing_data.npz	-> 	labels for testing data reformatted for xgboost

label_training_data.npz	-> 	labels for training data reformatted for xgboost

results.p		->	results from running param_tuning.py

roc_S2M1.p		-> 	ROC curve for model S2M1

roc_S2M2.p		-> 	ROC curve for model S2M2

roc_S2M3.p		-> 	ROC curve for model S2M3

roc_S2M4.p		-> 	ROC curve for model S2M4

train.json		->	original dataset

test_data_2.npz 	->	partitioned testing set

train_data_2.npz	->	partitioned training set

testing_data.npz	-> 	testing data reformatted for xgboost

training_data.npz	-> 	training data reformatted for xgboost

test_ds.p		->	temporary data structures constructed from the testing data

train_ds.p		->	temporary data structures constructed from the training data

