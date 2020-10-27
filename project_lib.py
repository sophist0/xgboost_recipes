# #####################################################
# Functions I reuse in training this model
#######################################################
import numpy as np
import random
import pickle
import os
import math
import csv
import matplotlib.pyplot as plt
import json


def part_data():
	ptrain = "data/used/train_data_2.npz"
	ptest = "data/used/test_data_2.npz"
	if (not os.path.isfile(ptrain)) or (not os.path.isfile(ptest)):
		data = open("./data/used/train.json")
		jdata = json.load(data)

		# random split saving 10% for testing
		lt = int(0.1*len(jdata))
		tmp = random.sample(range(len(jdata)),lt)
		test_idx = dict.fromkeys(tmp,None)	

		train = []
		test = []
		for x in range(len(jdata)):
			if x in test_idx:
				test.append(jdata[x])
			else:
				train.append(jdata[x])

	train = np.asarray(train)
	test = np.asarray(test)
	np.savez_compressed(ptrain,train)
	np.savez_compressed(ptest,test)


def get_ds(jdata):
	##################################
	# Setup initial data structures
	##################################

	# get cuisine labels
	cis = {} # int label : str label 
	csi = {} # str label : int label 
	cfreq = {} # str label: int label count
	ncuisine = 0
	for el in jdata:

		c = el['cuisine']
		if c not in csi:
			cfreq[c] = 1
			csi[c] = ncuisine
			cis[ncuisine] = c
			ncuisine += 1
		else:
			cfreq[c] += 1

	# get all ingredients
	isi = {} # str label : int label 
	ifreq = {} # str label: int label count
	ning = 0
	for el in jdata:

		ilist = el['ingredients']
		for el in ilist:
			if el not in isi:
				ifreq[el] = 1
				isi[el] = ning
				ning += 1
			else:
				ifreq[el] += 1

	return [ncuisine, ning, cis, csi, isi]


def format_data(ning,csi,isi,jdata,dpath1,dpath2):

	# randomize indexing to randomize order of the dataset
	l = len(jdata)
	randx = random.sample(range(l),l)
	jdx = 0
	datamat = []
	datalabel = []
	for rdx in randx:

		el = jdata[rdx]
		c = el['cuisine']
		tmp = [0 for x in range(ning)]
		ilist = el['ingredients']

		for ing in ilist:
			if ing in isi:
				idx = isi[ing]
				tmp[idx] = 1
			# else ingredient in the testing set is not in the training set.

		# Again speed up!!!!
		datamat.append(tmp)
		datalabel.append(csi[c])
		jdx += 1 

	fdata = np.asarray(datamat)
	fdata_label = np.asarray(datalabel)
	np.savez_compressed(dpath1,fdata)
	np.savez_compressed(dpath2,fdata_label)

	return fdata, fdata_label


def load_data(path_vec,flag,train_ni,train_csi,train_isi):

	dpath0 = path_vec[0]
	dpath1 = path_vec[1]
	dpath2 = path_vec[2]
	dspath = path_vec[3]

	if (not os.path.isfile(dpath1)) or (not os.path.isfile(dpath2)) or (not os.path.isfile(dspath)):
		tmp = np.load(dpath0, allow_pickle=True)
		data = tmp['arr_0']

		# setup data structures required for setting up training and testing sets
		ncuisine, ning, cis, csi, isi = get_ds(data)

		# save data structures
		ds = [ncuisine, ning, cis, csi, isi]
		pickle.dump(ds, open(dspath, 'wb'))

		# format data
		if flag == "train":
			fdata, fdata_label = format_data(ning,csi,isi,data,dpath1,dpath2)
		else:
			fdata, fdata_label = format_data(train_ni,train_csi,train_isi,data,dpath1,dpath2)
	else:
		fdict1 = np.load(dpath1)
		fdata = fdict1["arr_0"]

		fdict2 = np.load(dpath2)
		fdata_label = fdict2["arr_0"]

		ds = pickle.load(open(dspath, 'rb'))
		ncuisine = ds[0]
		ning = ds[1]
		cis = ds[2]
		csi = ds[3]
		isi = ds[4]

	return [fdata, fdata_label, ncuisine, ning, cis, csi, isi]


def format_data_freq(ning,csi,isi,jdata,dpath1,dpath2):

	# randomize indexing to randomize order of the dataset
	l = len(jdata)
	randx = random.sample(range(l),l)

	jdx = 0
	datamat = []
	datalabel = []
	for rdx in randx:

		el = jdata[rdx]
		c = el['cuisine']
		tmp = [0 for x in range(ning)]
		ilist = el['ingredients']

		for ing in ilist:
			if ing in isi:
				idx = isi[ing]
				tmp[idx] = 1
			# else ingredient in the testing set is not in the training set.

		datamat.append(tmp)
		datalabel.append(csi[c])
		jdx += 1 

	fdata = np.asarray(datamat)
	fdata_label = np.asarray(datalabel)
	print()

	return fdata, fdata_label


def get_ds_freq_1(jdata, mf):
	##################################
	# Setup data structures
	# Most frequent ingredients
	##################################

	# get cuisine labels
	cis = {} # int label : str label 
	csi = {} # str label : int label 
	cfreq = {} # str label: int label count
	ncuisine = 0
	for el in jdata:

		c = el['cuisine']
		if c not in csi:
			cfreq[c] = 1
			csi[c] = ncuisine
			cis[ncuisine] = c
			ncuisine += 1
		else:
			cfreq[c] += 1

	# get all ingredients
	isi = {} # str label : int label 
	ifreq = {} # str label: int label count
	ning = 0
	for el in jdata:

		ilist = el['ingredients']
		for el in ilist:
			if el not in isi:
				ifreq[el] = 1
				isi[el] = ning
				ning += 1
			else:
				ifreq[el] += 1

	# get ingredient frequencies as tuples (frequency,ingredient)
	fvec = []
	for key in ifreq:
		fvec.append((ifreq[key],key))

	# this sorts first by ingredient count then alphabetically ingredient name (could randomly select ingredients with the same count instead)
	fvec.sort(reverse=True)

	# reduce ingredient dicts
	isi2 = {}
	for x in range(mf):
		isi2[fvec[x][1]] = x

	return [ncuisine, cis, csi, isi2]

def get_ds_freq_2(jdata, mf):
	###################################################
	# Setup data structures
	# Most frequent mf ingredients for each cuisine
	###################################################

	# cuisine frequency
	cis, csi, cfreq, ncuisine = cl1.cuisine_freq(jdata)

	# count of ingredients in each cuisine
	alling, fc_dict = cl1.cnt_ingredients(cfreq,jdata)

	# most frequent ingredients in each cuisine
	mfc_dict, fi_list, fi_cnt = cl1.mfi_cuisine(mf,cfreq,fc_dict)
	

	# reduce ingredient dicts (str label: int label)
	isi2 = {}
	ning = 0
	for cuisine in mfc_dict:
		
		ilist = mfc_dict[cuisine]
		for el in ilist:
			if el[1] not in isi2:
				isi2[el[1]] = ning
				ning += 1

	return [ncuisine, ning, cis, csi, isi2]


def get_ds_freq_3(jdata, mf, th, rev):
	###################################################
	# Setup data structures
	# Most frequent mf ingredients for each cuisine and the ingredients shared least between cuisines
	###################################################

	# cuisine frequency
	cis, csi, cfreq, ncuisine = cuisine_freq(jdata)

	# count of ingredients in each cuisine
	alling, fc_dict = cnt_ingredients(cfreq,jdata)

	# most frequent ingredients in each cuisine
	mfc_dict, fi_list, fi_cnt = mfi_cuisine(th,cfreq,fc_dict)

	# order ingredients by the number of cuisines they are in (their degree)
	ic_dict = {}
	c = 0
	for ing in fi_list:
		ic_dict[ing] = []
		for recipe in jdata:
			if (ing in recipe['ingredients']) and (recipe['cuisine'] not in ic_dict[ing]):
				ic_dict[ing].append(recipe['cuisine'])
		c += 1
		if c % 100 == 0:
			print("Percent ordered: ",c/len(fi_list))

	# SHOULD PASS OUT iorder after first call and reuse!
	iorder = []
	for ing in ic_dict:
		iorder.append((len(ic_dict[ing]),ing))

	if rev == True:
		iorder.sort(reverse=True)
	else:
		iorder.sort()
	ning = int(mf*len(iorder))

	# reduce ingredient dicts (str label: int label)
	isi2 = {}
	for x in range(ning):
		isi2[iorder[x][1]] = x

	return [ncuisine, ning, cis, csi, isi2]

def load_data_freq(path_vec,flag,order,train_ni,train_csi,train_isi,mf,th):

	# Only looking at the mf most frequent ingredients
	dpath0 = path_vec[0]
	dpath1 = path_vec[1]
	dpath2 = path_vec[2]
	dspath = path_vec[3]
	tmp = np.load(dpath0, allow_pickle=True)
	data = tmp['arr_0']

	# setup data structures required for setting up training and testing sets
	if order == 1:
		ncuisine, ning, cis, csi, isi = get_ds_freq_1(data,mf)

	elif order == 2:
		ncuisine, ning, cis, csi, isi = get_ds_freq_2(data,mf)

	elif order == 3:
		ncuisine, ning, cis, csi, isi = get_ds_freq_3(data,mf,th,False)

	elif order == 4:
		ncuisine, ning, cis, csi, isi = get_ds_freq_3(data,mf,th,True)

	# format data
	if flag == "train":
		fdata, fdata_label = format_data_freq(ning,csi,isi,data,dpath1,dpath2)
	else:
		fdata, fdata_label = format_data_freq(train_ni,train_csi,train_isi,data,dpath1,dpath2)

	return [fdata, fdata_label, ncuisine, ning, cis, csi, isi]


# Count ingredients in each cuisine
def cnt_ingredients(cfreq,data):

	ing = {}	# all ingredients
	fc_dict = {}
	for el in cfreq:

		if el not in fc_dict:
			fc_dict[el] = {}

	cnt = 0
	for el in data:

		c = el['cuisine']
		ilist = el['ingredients']
		for ingredient in ilist:
			if ingredient not in fc_dict[c]:
				fc_dict[c][ingredient] = 1
			else:
				fc_dict[c][ingredient] += 1

			if ingredient not in ing:
				ing[ingredient] = True

	alling = list(ing.keys())
	return alling, fc_dict


def ingredient_freq(data):

	# get all ingredients
	iis = {} # int label : str label 
	isi = {} # str label : int label 
	ifreq = {} # str label: int label count
	ning = 0
	for el in data:

		ilist = el['ingredients']
		for el in ilist:
			if el not in isi:
				ifreq[el] = 1
				isi[el] = ning
				iis[ning] = el
				ning += 1
			else:
				ifreq[el] += 1

	return [iis, isi, ifreq, ning]


def cuisine_freq(data):

	# get cuisine labels
	cis = {} # int label : str label 
	csi = {} # str label : int label 
	cfreq = {} # str label: int label count
	ncuisine = 0
	for el in data:

		c = el['cuisine']
		if c not in csi:
			cfreq[c] = 1
			csi[c] = ncuisine
			cis[ncuisine] = c
			ncuisine += 1
		else:
			cfreq[c] += 1

	return [cis,csi,cfreq,ncuisine]


def mfi_cuisine(n, cfreq, fc_dict):

	########################################################################################
	# get set of the n most frequent ingredients in each cuisine
	########################################################################################
	mfc_dict = {}
	fi_cnt = 0
	fi_dict = {}
	for el in cfreq:

		flist = [(0,'none') for x in range(n)]
		m = 0
		for ingredient in fc_dict[el]:

			if fc_dict[el][ingredient] > m:
				flist[0] = (fc_dict[el][ingredient],ingredient)
				flist.sort()
				m = flist[0][0]

		mfc_dict[el] = flist

		for ingredient in flist:
			if ingredient[1] not in fi_dict:
				fi_cnt += 1
				fi_dict[ingredient[1]] = True
	fi_list = list(fi_dict.keys())

	return [mfc_dict, fi_list, fi_cnt]

def print_cm(train_nc, train_cis, test_label, test_pred):

	####################################
	# Print Confusion Matrix
	####################################
	print()
	print("Confusion Matrix:")
	cm = [[0 for x in range(train_nc)] for y in range(train_nc)]
	cm = np.asarray(cm)
	idx = 0
	for x in range(len(test_pred)):
		cm[int(test_label[x]),int(test_pred[idx])] += 1
		idx += 1
	
	print(cm)
	print()
	print(train_cis)
	print()

def roc_ml1(test_label, test_pred):

	# ROC for multi-label, computes roc for each class treating the rest as a single class and reducing it to a binary problem.
	# There are other ways to do this
	# https://scikit-learn.org/stable/modules/model_evaluation.html (one-vs-one, one-vs-rest)

	# x-axis: false positive rate -> FP/P where P is the actual number of positive labels
	# y-axis: true positives rate -> TP/N where N is the actual number of negative labels
	# Thresholds to assume class label is positive

	tmp1 = [-5/(10*x) for x in range(1,21)]
	tmp2 = [-x/16 for x in range(8,161)]
	tmp1.reverse()
	thvec = tmp1 + tmp2
	thvec = list(map(math.exp,thvec))

	test_label = list(map(int, test_label))
	nclass = len(test_pred[0])
	ndata = len(test_pred)
	results = {}
	for label in range(nclass):
		print("ROC for class: ",label)
		nl = test_label.count(label)
		results[label] = {'th':[],'fpr':[],'tpr':[]}
		for t in range(len(thvec)):
			fp = 0
			tp = 0
			for r in range(ndata):
				if test_pred[r][label] > thvec[t]:
					if label == test_label[r]:
						tp += 1
					else:
						fp += 1

			results[label]['th'].append(thvec[t])
			results[label]['fpr'].append(fp/(ndata-nl))
			results[label]['tpr'].append(tp/(nl))

	# average roc results over the classes
	d = len(results[0]['th'])
	tpr = [0 for x in range(d)]
	fpr = [0 for x in range(d)]
	for label in range(nclass):		
		for x in range(d):
			tpr[x] += results[label]['tpr'][x] / nclass
			fpr[x] += results[label]['fpr'][x] / nclass

	return [tpr,fpr]


def plot_roc_results():

	plt.figure(figsize=(6,6))

	# model names
	mnvec = ["S2M1","S2M2","S2M3","S2M4"]
	color = ['b','r','k','g']
	for x in range(len(mnvec)):
		data = pickle.load(open('data/used/roc_'+mnvec[x]+".p",'rb'))	
		plt.plot(data['fpr'],data['tpr'],label=mnvec[x], color=color[x])

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.xlim((0,1))
	plt.ylim((0,1))
	plt.legend()
	plt.savefig('figures/rocs.png',dpi=600)
	plt.show()

def write_sweep_results():

	#############################################################
	# Write the results of sweeping a set of parameters to CSV 
	# for the n most accurate models
	#
	n=4
	path = "data/used/results.p"
	csv_path = 'data/used/best_results.csv'
	#############################################################

	results = pickle.load(open(path,'rb'))
	order = []
	for key in results:
		rdict = results[key]
		acc = rdict['accuracy']
		acc = np.asarray(acc)
		order.append((acc.mean(),acc.var(),key))

	order.sort(reverse=True)
	ord_results = []	# (mean acc, var acc, eta, gamma, max_depth, lambda, subsample, num_round)
	for el in order:
		eta = results[el[2]]['eta']
		gamma = results[el[2]]['gamma']
		md = results[el[2]]['max_depth']
		lb = results[el[2]]['lambda']
		sub = results[el[2]]['subsample']
		num = results[el[2]]['num_round']
		ord_results.append((el[0],el[1],eta,gamma,md,lb,sub,num))

	print()
	print("Top "+str(n)+" models accuracy, variance, and parameters")
	print()
	print("(accuracy, variance, eta, gamma, max_depth, lambda, subsample, num_round)")
	print()
	print(ord_results[0:n])
	print()

	# form rows to output to csv
	arow = []
	vrow = []
	erow = []
	grow = []
	mrow = []
	lrow = []
	srow = []
	nrow = []
	for x in range(n):
		el = ord_results[x]
		arow.append(round(el[0],3))
		vrow.append(round(el[1],3))
		erow.append(el[2])
		grow.append(el[3])
		mrow.append(el[4])
		lrow.append(el[5])
		srow.append(el[6])
		nrow.append(el[7])

	with open(csv_path, 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile, delimiter=',',)
		csvwriter.writerow(arow)
		csvwriter.writerow(vrow)
		csvwriter.writerow(erow)
		csvwriter.writerow(grow)
		csvwriter.writerow(mrow)
		csvwriter.writerow(lrow)
		csvwriter.writerow(srow)
		csvwriter.writerow(nrow)


def plot_fselect():

	results_1 = pickle.load(open("data/used/features_order_1.p","rb"))
	results_2 = pickle.load(open("data/used/features_order_2.p","rb"))
	results_3 = pickle.load(open("data/used/features_order_3.p","rb"))
	results_4 = pickle.load(open("data/used/features_order_4.p","rb"))

	plt.plot(results_1['mf'],results_1['acc'],label="Most Frequent Ingredients (#)")
	plt.plot(results_2['mf'],results_2['acc'],label="Most Frequent Ingredients for each Cuisine ($)")
	plt.plot(results_3['mf'],results_3['acc'],label="Ingredients in the Fewest Cuisines (&)")
	plt.plot(results_4['mf'],results_4['acc'],label="Ingredients in the Most Cuisines (*)")

	plt.xlim((0,601))
	plt.xlabel("Number of Features")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.savefig('figures/fselect.png',dpi=600)
	plt.show()
