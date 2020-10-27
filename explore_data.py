#############################################################################
# Initial exploration of the data
#############################################################################
import xgboost as xgb
import json
from scipy.special import factorial
import igraph as ig
import matplotlib.pyplot as plt
import project_lib as pl

fdata = open("./data/used/train.json")
jdata = json.load(fdata)
print()
print("Number of recipes: ", len(jdata))
print()
cis, csi, cfreq, ncuisine = pl.cuisine_freq(jdata)

print("#######################################################")
print()
print("Number of cuisines: ", ncuisine)
print()

# bar plot number of recipes in each cuisine
tmp1 = []
for key in cfreq:
	tmp1.append((cfreq[key],key))
tmp1.sort(reverse=True)

# unzip tmp1 tuples
cstr = [j for i,j in tmp1]
ccnt = [i for i,j in tmp1]

xpos = [x for x in range(len(ccnt))]
plt.bar(xpos, ccnt)
plt.grid(axis="y",linewidth=2,alpha=0.5)
plt.ylabel("Number of Recipes", fontsize=12)
plt.xticks(xpos,cstr,rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# get all ingredients
iis, isi, ifreq, ning = pl.ingredient_freq(jdata)

print("#######################################################")
print()
print("Number of ingredients: ", ning)
print()

# 20 most frequent ingredients
flist = [(0,'none') for x in range(20)]
m = 0
for el in ifreq:

	if ifreq[el] > m:
		flist[0] = (ifreq[el],el)
		flist.sort()
		m = flist[0][0]

print("#######################################################")
print()
print("20 most frequent ingredients from all cuisines")
print()

# bar plot number of the top 20 ingredients in all cuisines
tmp1 = []
for el in flist:
	tmp1.append((el[0],el[1]))
tmp1.sort(reverse=True)
print(tmp1)
print()

# unzip tmp1 tuples
cstr = [j for i,j in tmp1]
ccnt = [i for i,j in tmp1]
xpos = [x for x in range(len(ccnt))]
plt.bar(xpos, ccnt)
plt.grid(axis="y",linewidth=2,alpha=0.5)
plt.ylabel("Number of Ingredients", fontsize=12)
plt.xticks(xpos,cstr,rotation=25, fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Count ingredients in each cuisine
alling, fc_dict = pl.cnt_ingredients(cfreq,jdata)

#n=30
n=8 # number of most frequent ingredients per cuisine to include

# most frequent ingredients in each cuisine
mfc_dict, fi_list, fi_cnt = pl.mfi_cuisine(n,cfreq,fc_dict)

print("#########################################################################")
print("frequent ingredients not in other cuisines frequent ingredient sets")
print("#########################################################################")
for el_1 in cfreq:
	fi_1 = mfc_dict[el_1]
	fi_22 = []
	for el_2 in cfreq:
		if el_1 != el_2:
			fi_2 = mfc_dict[el_2]
			for tmp in fi_2:
				fi_22.append(tmp[1])

	tmp = []
	for ing in fi_1:
		if ing[1] not in fi_22:
			tmp.append(ing[1])
	print()
	print(el_1)
	print(tmp)
print()

######################################################################
# Look at all set intersections between frequent cuisine ingredients

print("#################################################################")
print()
print("Number of potential intersections between cuisines and ingredients")
nint = 0
m = 20
for i in range(1,m+1):
	nint += int(factorial(m) / (factorial(i)*factorial(m-i)))

print(nint)
print()

# Luckily the data is sparse enough that we shouldn't have this many subsets. In fact max is 6714 the number of unique ingredients in the dataset and the number of frequent ingredients is much lower.
print("Number of frequent ingredients")
print(fi_cnt)
print()
print(fi_list)
print()

tdict = {}
for ingredient in fi_list:
	key = ""
	for el in cfreq:
		tmp = mfc_dict[el]
		fing = []
		for ing in tmp:
			fing.append(ing[1])

		if ingredient in fing:
			key += str(csi[el]) + "_"

	key = key.strip("_")
	if (key != "") and (key in tdict):
		tdict[key].append(ingredient)	
	elif key != "":
		tdict[key] = [ingredient]

print()
print("Number of actual intersections between cuisines and ingredients in the dataset")
print(len(tdict))
print()

# Plot the bipartite graph of cuisines and ingredients
g = ig.Graph()
g.add_vertices(20)
labels = [cis[x] for x in range(20)]
label_sizes = [36 for x in range(20)]
colors = ["white" for x in range(20)]
sizes = [180 for x in range(20)]

# Add ingredient set labels
g.add_vertices(len(tdict))
node = 20
for key in tdict:

	links = key.split("_")
	ilist = tdict[key]
	nlabel = ""
	for el in ilist:
		nlabel += el + "\n"
	nlabel = nlabel.strip("\n")

	labels.append(nlabel)
	colors.append("light blue")
	sizes.append(180)
	label_sizes.append(36)

	for x in range(len(links)):
		g.add_edges([(node,int(links[x]))])
	node += 1

layout = g.layout("kk")
visual_style = {}
visual_style["edge_width"] = 6
visual_style["vertex_label"] = labels
visual_style["vertex_label_size"] = label_sizes
visual_style["vertex_color"] = colors
visual_style["bbox"] = (4000,4000)
visual_style["layout"] = layout
visual_style["margin"] = 150
visual_style["vertex_size"] = sizes # right size for cuisine labels
ig.plot(g, **visual_style)
