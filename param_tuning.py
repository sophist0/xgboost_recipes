import xgboost as xgb
import random
import numpy as np
import pickle
import itertools
import project_lib as pl
np.set_printoptions(linewidth=150)

def setup_val_sets(fdata,k):

	l = len(fdata)
	lk = int(l/k)
	validx = []
	for x in range(k-1):
		tmp = (x*lk,(x+1)*lk)
		validx.append(tmp)

	# add any left over samples to the final validation set
	validx.append(((k-1)*lk,l+1))
	return validx

def cross_valid(k, ncuisine, validx, fdata, fdata_label, param):

	param['num_class'] = ncuisine
	param['objective'] = 'multi:softmax'
	param['tree_method'] = 'gpu_hist'	# faster but approximate
	#param['tree_method'] = 'exact'
	num_round = param['num_round']
	del param['num_round']

	# train k models and evaluate on the k validation sets
	accvec = []
	for kdx in range(k):

		print("################################################")
		print("k-Model: "+str(kdx+1))
		print()
		vs = validx[kdx]
		dval = []
		dval_label = []
		dtrain = []
		dtrain_label = []
		for x in range(len(fdata)):
			if (x >= vs[0]) and (x < vs[1]):
				dval.append(fdata[x])
				dval_label.append(fdata_label[x])
			else:
				dtrain.append(fdata[x])
				dtrain_label.append(fdata_label[x])

		dval = np.asarray(dval)
		dval_label = np.asarray(dval_label)

		dtrain = np.asarray(dtrain)
		dtrain_label = np.asarray(dtrain_label)

		dval = xgb.DMatrix(dval, label=dval_label)
		dtrain = xgb.DMatrix(dtrain, label=dtrain_label)
		bst = xgb.train(param, dtrain, num_round)

		# make prediction
		preds = bst.predict(dval)
		preds = list(map(int,preds))

		l = len(dval.get_label())
		acc = 0
		for x in range(l):
			if int(dval.get_label()[x]) == preds[x]:
				acc += 1
		acc = acc/l
		print("Model Accuracy: ",acc)
		accvec.append(acc)

	return accvec

########################################################
dpath0 = "data/used/train_data_2.npz"
dpath1 = "data/used/training_data.npz"
dpath2 = "data/used/label_training_data.npz"
dspath = "data/used/train_ds.p"
pathvec = [dpath0,dpath1,dpath2,dspath]
train_data, train_label, train_nc, train_ni, train_cis, train_csi, train_isi = pl.load_data(pathvec,'train',-1,{},{})
print()
print()
print("Parsed training data....")
print()

# split training set into training and validation for 5 fold cross validation
k=5
validx = setup_val_sets(train_data,k)

########################################################
# Parameters to sweep
########################################################
param = {}

# Short test
param['eta'] = [0.1,0.3]
param['gamma'] = [0]
param['max_depth'] = [2,4]
param['lambda'] = [0]
param['subsample'] = [1]
param['num_round'] = [2]

# Sweep 1 parameters
#param['eta'] = [0.1,0.5,1]
#param['gamma'] = [0,0.3]
#param['max_depth'] = [2,4,6]
#param['lambda'] = [0,1,3]
#param['subsample'] = [0.7,1]
#param['num_round'] = [2,4]

# Sweep 2 parameters
#param['eta'] = [0.5]
#param['gamma'] = [0]
#param['max_depth'] = [8,10,12,14]
#param['lambda'] = [0,1]
#param['subsample'] = [0.7,1]
#param['num_round'] = [6,8,10,12]

# create parameter sets
vec2 = []
plist = []
for key in param:
	plist.append(key)
	vec1 = param[key]
	vec2.append(vec1)
pvec = list(itertools.product(*vec2))

# Sweep parameters
num = 0
results = {}
for params in pvec:

	print()
	print("#########################################################################")
	num += 1
	print('Parameter set: '+str(num)+" of "+str(len(pvec)))
	print("plist: ",plist)
	print("params: ",params)
	print()
	pdict = {}
	pdix = 0
	for el in plist:
		pdict[el] = params[pdix]
		if el == "num_round":			# cross_valid() deletes this parameter
			num_rounds = params[pdix]
		pdix += 1

	accvec = cross_valid(k, train_nc, validx, train_data, train_label, pdict)
	pdict['num_round'] = num_rounds
	pdict['accuracy'] = accvec
	accvec = np.asarray(accvec)
	print()
	print("Model Cross Validated Accuracy Mean: ", accvec.mean())
	#print("Model Cross Validated Accuracy Variance: ", accvec.var())
	print()
	results[str(num)] = pdict

print()
pickle.dump(results, open("data/used/results.p", 'wb'))
print("Done with parameter sweep!!!")
print()
