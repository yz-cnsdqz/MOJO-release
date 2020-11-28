import pandas as pd
import numpy as np
import pdb

## the results in tables for proxe
user_study_file = 'user_study_v1.csv'

df = pd.DataFrame()
df = pd.read_csv(user_study_file)

exps = ['vanilla','mojo', 'gt']
datasets = ['ACCAD', 'BMLhandball']
scores = {'strongly disagree': 1,
          'disagree':2,
          'slightly disagree':3,
          'slightly agree':4,
          'agree':5,
          'strongly agree':6}


for exp in exps:
    for data in datasets:
        df_0 = df.loc[  df['Input.video_url'].str.contains(exp)
                    & df['Input.video_url'].str.contains(data)
                    & ~df['Answer.your opinion.label'].str.contains('not playing')]

        results = df_0['Answer.your opinion.label'].to_list()
        ss = [scores[x] for x in results]
        mu = np.mean(ss)
        sigma = np.std(ss)

        print('-- model:{}, data:{}, num_ratings:{}. stats= {:03f} pm {:03f}'.format(exp, data, df_0.shape[0], mu, sigma))



