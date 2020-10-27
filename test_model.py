import xgboost as xgb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import project_lib as pl
np.set_printoptions(linewidth=150)

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
	preds = bst.predict(test_data)

	if param['objective'] == "multi:softmax":

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

		# confusion matrix
		pl.print_cm(train_nc, train_cis, test_data.get_label(), preds)

	elif param['objective'] == "multi:softprob":

		print()
		print("Predicting labels probabilities to plot ROC")
		print()

		tpr, fpr = pl.roc_ml1(test_data.get_label(), preds)

		plt.xlabel("False Positive Rate")
		plt.ylabel("True Positive Rate")
		plt.xlim(0,1)
		plt.ylim(0,1)
		plt.plot(fpr,tpr)
		plt.show()

		# save roc
		roc = {}
		roc['model_name'] = mname
		roc['tpr'] = tpr
		roc['fpr'] = fpr
		pickle.dump(roc, open('data/used/roc_'+mname+'.p','wb'))

########################################################
# Load training data to build model
########################################################

dpath0 = "data/used/train_data_2.npz"
dpath1 = "data/used/training_data.npz"
dpath2 = "data/used/label_training_data.npz"
dspath = "data/used/train_ds.p"
pathvec = [dpath0,dpath1,dpath2,dspath]

train_data, train_label, train_nc, train_ni, train_cis, train_csi, train_isi = pl.load_data(pathvec,'train',-1,{},{})
print()
print("Setup training data")

dpath0 = "data/used/test_data_2.npz"
dpath1 = "data/used/testing_data.npz"
dpath2 = "data/used/label_testing_data.npz"
dspath = "data/used/test_ds.p"
pathvec = [dpath0,dpath1,dpath2,dspath]

test_data, test_label, test_nc, test_ni, test_cis, test_csi, test_isi = pl.load_data(pathvec,'test',train_ni,train_csi,train_isi)
print()
print("Setup testing data")
print()

########################################################
# Parameters
########################################################
param = {}
ROC = True
if ROC == False:
	# Set objective for label prediction
	param['objective'] = 'multi:softmax'

else:
	# Set objective for ROC plots
	param['objective'] = 'multi:softprob'

# TEST
mname = "TEST"
print("Model: ",mname)
param['eta'] = 0.5
param['gamma'] = 0
param['max_depth'] = 4
param['lambda'] = 0
param['subsample'] = 1
param['num_round'] = 2

# S2M1
#mname = "S2M1"
#print("Model: ",mname)
#param['eta'] = 0.5
#param['gamma'] = 0
#param['max_depth'] = 14
#param['lambda'] = 0
#param['subsample'] = 1
#param['num_round'] = 12

# S2M2
#mname = "S2M2"
#print("Model: ",mname)
#param['eta'] = 0.5
#param['gamma'] = 0
#param['max_depth'] = 12
#param['lambda'] = 0
#param['subsample'] = 1
#param['num_round'] = 12

# S2M3
#mname = "S2M3"
#print("Model: ",mname)
#param['eta'] = 0.5
#param['gamma'] = 0
#param['max_depth'] = 14
#param['lambda'] = 0
#param['subsample'] = 0.7
#param['num_round'] = 12

# S2M4
#mname = "S2M4"
#print("Model: ",mname)
#param['eta'] = 0.5
#param['gamma'] = 0
#param['max_depth'] = 14
#param['lambda'] = 1
#param['subsample'] = 1
#param['num_round'] = 12

print()
print("params: ",param)
print()
test_model(train_data, train_label, test_data, test_label, param, train_nc, train_cis, mname)
