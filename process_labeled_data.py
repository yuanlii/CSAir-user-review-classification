
# '../Source_Data/CSV_files/*.csv' => listing all csv files of the 10 classes
# 
import glob
import pandas as pd
from sklearn import preprocessing

class ProcessLabeledData(object):    
    def __init__(self):
        self.raw_labeled_data_lst = []
        self.all_raw_labeled_data = pd.Series()

    def concat_labeled_data(self, source_data_path):
        '''concatenate data from 10 classes into one dataframe '''
        files= glob.glob(source_data_path)
        for f in files:
            label = f.split('/')[-1][:2]
            raw_data = pd.read_csv(f,header=None)
            raw_data['label'] = label
            self.raw_labeled_data_lst.append(raw_data)
        self.all_raw_labeled_data = pd.concat(self.raw_labeled_data_lst)
        self.all_raw_labeled_data = self.all_raw_labeled_data.rename(columns = {0:'review'})
        return self.all_raw_labeled_data

    def encode_labeled_data(self, output_data_file):
        ''' encode classes (text) into digits; 
            return a df with '''
        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(self.all_raw_labeled_data.label)
        self.all_raw_labeled_data['label_encoded'] = targets
        # output concatenated labeled data to csv file
        self.all_raw_labeled_data.to_csv(output_data_file, index = False)
        return self.all_raw_labeled_data

    
