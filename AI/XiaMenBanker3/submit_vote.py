import numpy as np
import math
import pandas as pd
# lgb = pd.read_csv("preds_lgb.csv")
# ctb = pd.read_csv("preds_ctb.csv")
xgb = pd.read_csv("preds_xgb.csv")
submit = pd.read_csv("./Data_A/x_test.csv")['id'].to_frame()
# preds0 = pd.concat((xgb,ctb),axis=1)
preds0 = xgb
best_th = 0.25
preds = np.zeros([submit.shape[0]])
while(preds.sum()<270000):    #保证recall比较高
    preds = preds0.copy()
    #preds = preds>=best_th
    preds = preds.mean(axis=1).values
    #preds[preds>=0.5],preds[preds<0.5] = 1,0
    preds[preds>=best_th],preds[preds<best_th] = 1,0
    print(best_th,"预测出的正例数量：",preds.sum())
    best_th = best_th-0.0001

submit['y'] = preds
submit.to_csv(f"submit_all.csv",index=False)
