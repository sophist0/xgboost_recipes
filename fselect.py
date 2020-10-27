# Test the model using various feature selection policies
import xgboost as xgb
import numpy as np
import pickle
import project_lib as pl

def test_model(train_data, train_label, test_data, test_label, param, train_nc, train_cis, mname):

	# specify parameters via map
	param['num_class'] = train_nc
	param['tree_method'] = 'gpu_hist'	# faster but approximate
	#param['tree_method'] = 'exact'
	num_round = param['num_round']
	del param['num_round']

	#########################################################

	# setup datasets
	test_data = np.asarray(test_data)
	test_label = np.asarray(test_label)

	train_data = np.asarray(train_data)
	train_label = np.asarray(train_label)

	test_data = xgb.DMatrix(test_data, label=test_label)
	train_data = xgb.DMatrix(train_data, label=train_label)

	#########################################################

	bst = xgb.train(param, train_data, num_round)

	# make prediction
	preds = bst.predict(test_data)

	print()
	print("Predicting labels to calculate accuracy")
	print()

	preds = list(map(int,preds))

	l = len(test_data.get_label())
	acc = 0
	for x in range(l):
		if int(test_data.get_label()[x]) == preds[x]:
			acc += 1
	print("Accuracy:")
	acc = acc/l
	print(acc)
	print()

	return acc

########################################################
# Load training data to build model
########################################################

#order = 1	# order ingredients by most frequent ingredient. Equivalent to the ingredients with the most variance since no ingredient is in half the recipes.
#order = 2	# order ingredients by most frequent ingredient by cuisine.
order = 3	# order by lowest degree ingredients above a threshold of order=2 say 100 most frequent ingredients by cuisine.
#order = 4	# order by highest degree ingredients above a threshold of order=2 say 100 most frequent ingredients by cuisine.

dpath01 = "data/used/train_data_2.npz"
dpath11 = "data/used/training_data.npz"
dpath21 = "data/used/label_training_data.npz"
dspath1 = "data/used/train_ds.p"

dpath02 = "data/used/test_data_2.npz"
dpath12 = "data/used/testing_data.npz"
dpath22 = "data/used/label_testing_data.npz"
dspath2 = "data/used/test_ds.p"

if order == 1:
	mfvec = [10*x for x in range(1,101)] # top mf most frequent ingredient
elif order == 2:
	mfvec = [5*x for x in range(1,21)] # top mf most frequent ingredient for each cuisine (what are the fewest ingredients in any cuisine, less than 100?)
elif order == 3 or order == 4:
	mfvec = [0.05*x for x in range(1,21)] # ingredients in the fewest cuisines that are in the top th most frequent ingredients of a cuisine
	th = 100

print()
accvec = []
nivec = []
for mf in mfvec:
	
	print("################################################")
	print("mf: ", mf)
	print("################################################")

	pathvec = [dpath01,dpath11,dpath21,dspath1]
	train_data, train_label, train_nc, train_ni, train_cis, train_csi, train_isi = pl.load_data_freq(pathvec,'train',order,-1,{},{},mf,th)

	pathvec = [dpath02,dpath12,dpath22,dspath2]
	test_data, test_label, test_nc, test_ni, test_cis, test_csi, test_isi = pl.load_data_freq(pathvec,'test',order,train_ni,train_csi,train_isi,mf,th)

	########################################################
	# Parameters
	########################################################
	param = {}

	# Set objective for label prediction
	param['objective'] = 'multi:softmax'

	# TEST
	#mname = "TEST"
	#print("Model: ",mname)
	#param['eta'] = 0.5
	#param['gamma'] = 0
	#param['max_depth'] = 4
	#param['lambda'] = 0
	#param['subsample'] = 1
	#param['num_round'] = 2

	# S2M1
	mname = "S2M1"
	print("Model: ",mname)
	param['eta'] = 0.5
	param['gamma'] = 0
	param['max_depth'] = 14
	param['lambda'] = 0
	param['subsample'] = 1
	param['num_round'] = 12

	acc = test_model(train_data, train_label, test_data, test_label, param, train_nc, train_cis, mname)
	accvec.append(acc)
	nivec.append(train_ni)

results = {}
results["mf"] =  nivec
results['acc'] = accvec
pickle.dump(results,open("data/used/features_order_"+str(order)+".p","wb"))
