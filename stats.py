import pandas as pd





df = pd.read_csv("results.csv")

# Dropping the last column because it's useless
df.drop(df.columns[-1], axis = 1, inplace=True)



# Average train performance and test performance by train function

print(df.groupby(["train_fct"])[["best_train_performance", "best_test_performance"]].mean())


