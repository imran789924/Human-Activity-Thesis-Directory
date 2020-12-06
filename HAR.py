import os
import csv
import pandas as pd


#for dirpath, dirnames, filenames in os.walk('/home/imran/Desktop/Project HAR/DATASET'):
for dirpath, dirnames, filenames in os.walk('/home/imran/Desktop/Project HAR/Sitting_final'):
    
    
    for file in filenames:
        
        path = os.path.join(dirpath, file)
        
        mystr = ""
        
        with open(path) as r:
            for line in r:
                if line and line[0].isdigit():  # no need to compare against True 
                    mystr = mystr + line
                else: 
                    print(line.strip())
        
        from io import StringIO
        
        TESTDATA = StringIO(mystr)
        
        
        df = pd.read_csv(TESTDATA, sep="\t", header = None, index_col=None)
        
        len = df.shape[0] % 50
        end = df.shape[0] - (len+150)
        
        df2 = df.iloc[151: , : ]
        
        if(len != 0):
            df2 = df2.iloc[ :end, :]
        
        path = '/home/imran/Desktop/Project HAR/Sitting_final'
        
        path = os.path.join(path,  file+'.csv')
            
        df2.to_csv(path, sep='\t', encoding='utf-8', index=False)

    
