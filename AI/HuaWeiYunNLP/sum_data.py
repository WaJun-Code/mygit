import pandas as pd
df0,df1=pd.read_csv('../data/small_data.csv',header=None),pd.read_csv('../data/large_data.csv',header=None)
df0,df1=df0.sort_values(2).reset_index(drop=True),df1.sort_values(2).reset_index(drop=True)
pd.concat((df1.iloc[:60000], df1.iloc[-60:]),axis=0).to_csv('../data/val1_data.csv',index=False,header=None)
#pd.concat((df0.iloc[:60000], df0.iloc[-60:]),axis=0).to_csv('../data/val_data.csv',index=False,header=None)
#df=pd.concat((df1.sample(frac=1), df0.iloc[60000:-60].sample(frac=1)),axis=0)
#df0.iloc[60000:-60].sample(frac=1).to_csv('../data/train_data.csv',index=False,header=None)
#df.sample(frac=1).to_csv('../data/sum_data.csv',index=False,header=None)
print(df.isnull().sum())
print(df.shape)