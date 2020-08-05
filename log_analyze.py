import argparse
import os
import pandas as pd

dirs_ = os.listdir('./logs')
dirs_ = ['./logs/'+ dir_ for dir_ in dirs_]

final_dic = {'model': [],'config': [], 'score': [], 'monotone': [], 'screenshot': [], 'unknown': []}
configs = []
scores = []
for dir_ in dirs_:
    print(dir_, '\n\n==============================\n')
    with open(dir_, 'r') as f :
        content = f.readlines()
        
        config_enter = 0
        config = ""
        
        score_count = 0
        scores = []
        monotones = []
        screenshots = []
        unknowns = []
        for i, line in enumerate(content):
            # line = line.replace('\n', '')
            if 'Loaded!' in line:
                final_dic['model'].append(line.split(' ')[0])

            if 'batch_size' in line:
                config_enter = 1
                config += line
            
            elif 'weight_decay' in line:
                config_enter = 0
                config += line
                
                final_dic['config'].append(config)
                config = ""
            
            elif config_enter:
                config += line
            
            if 'Final' in line:
                line = line.split(' ')
                scores.append(float(line[2]))
                monotones.append(float(line[4].replace('),', '')))
                screenshots.append(float(line[6].replace('),', '')))
                unknowns.append(float(line[-1].replace(')]\n', '')))

            if 'train finished' in line:
                final_dic['score'].append(scores)
                final_dic['monotone'].append(monotones)
                final_dic['screenshot'].append(screenshots)
                final_dic['unknown'].append(unknowns)

                scores = monotones = screenshots = unknowns = [] 

df = pd.DataFrame.from_dict(final_dic)
df['best_score'] = df['score'].apply(lambda x : max(x))
df['best_idx'] = df[['score', 'best_score']].apply(lambda x: x[0].index(x[1]), axis=1)
df['best_monotone'] = df[['monotone', 'best_idx']].apply(lambda x : x[0][x[1]], axis=1)
df['best_screenshot'] = df[['screenshot', 'best_idx']].apply(lambda x : x[0][x[1]], axis=1)
df['best_unknown'] = df[['unknown', 'best_idx']].apply(lambda x : x[0][x[1]], axis=1)

df = df.sort_values(by=['best_score'])
print(df[['model','config','best_score','best_monotone', 'best_screenshot', 'best_unknown']].iloc[-5:])
