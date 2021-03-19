# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
            print("arg error")
            sys.exit(0)

    path = sys.argv[1]
    
    if not os.path.exists(path):
        print("file not exist")
        sys.exit(0)

    df=pd.read_csv(path,sep=',') 
    metrics = df.columns.values.tolist()
    metrics = [x for x in metrics if '_diff' not in x and 'name' not in x]
    df = df[ ['name'] + metrics]
    base_data = df.loc[df['name']=='base']
    for metric in metrics:
        base_metric = base_data[metric].iloc[0]
        df[metric + "_diff"] = (df[metric] - base_metric)/base_metric

    df.to_csv(path,index=False,header=True)
    print("save {} success".format(path))