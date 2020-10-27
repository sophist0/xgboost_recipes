import pickle
import matplotlib.pyplot as plt

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
