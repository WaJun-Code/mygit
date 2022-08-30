### 17_赛队8d8fd_代码运行步骤与说明

运行环境：
        四核i7-11370H @ 3.30GHz，机带RAM：40G，华硕天选air笔记本，3060显卡，6G显存，cuda11.4
        ！注意：本代码用的gensim版本是3.8.1，对4.0以上gensim版本需要注意更改一定格式。
        Ubuntu20.04，python==3.8.5、numpy==1.21.3、pandas==1.3.0、sklearn==0.24.2、gensim==3.8.1、xgboost==1.5.1

运行步骤：
        python prepare_data.py
        python xgb.py
    若需要运行伪标签，需要先用xgb.py生成一个submit_xgb_0.csv文件，然后将之重命名为submit_all.csv，然后将xgb.py里的第160行代码处注释里的 ,x_test[x_test['y']==0] 放进该行对应的concat里即可。表示伪标签加进去的伪负样本均在submit_all.csv里，且存在了x_test[x_test['y']==0]里面，粘进去则表示使用伪标签参与训练。
        再次python xgb.py即可

文件形式：
    ./Data_A/Data_main/      A榜主表数据
    ./Data_A/Data_other/      A榜副表数据
    ./Data_B/Data_main/      B榜主表数据
    ./Data_B/Data_other/      B榜副表数据