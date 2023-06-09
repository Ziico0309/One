import zipfile
from torch.utils.data import Dataset
import torch
import os
import pandas  as pd
import numpy as np
from zipfile import ZipFile
import requests
import sklearn
import random

class Download():
    def __init__(self,
                 root: str,
                 file_size: str = '100k',
                 download: bool = True,
                 ) -> None:
        self.root = root
        self.download = download
        self.file_size_url = file_size
        '''
        when file size if '100k' or '20m', dataframe is .csv file,
        other wise dataframe is .data file
        and when extract 10m.zip then, extracted directory is 10M100K.. so i have to consider it.
        '''
        if self.file_size_url == '100k' or self.file_size_url == '20m':
            self.file_url = 'ml-latest' if self.file_size_url == '20m' else 'ml-latest-small'
            self.fname = os.path.join(self.root, self.file_url, 'ratings.csv')
        else:
            if self.file_size_url == '10m':
                self.file_url = 'ml-' + self.file_size_url
                self.extracted_file_dir = 'ml-10M100K'
            if self.file_size_url =='1m':
                self.file_url = 'ml-' + self.file_size_url

            if self.file_size_url=='10m':
                self.fname = os.path.join(self.root, self.extracted_file_dir, 'ratings.dat')
            else:
                self.fname = os.path.join(self.root, self.file_url, 'ratings.dat')

        if self.download or not os.path.isfile(self.fname):
            self._download_movielens()
        self.df = self._read_ratings_csv()

    def _download_movielens(self) -> None:
        '''
        Download dataset from url, if there is no root dir, then mkdir root dir.
        After downloading, it wil be extracted
        :return: None
        '''
        file = self.file_url + '.zip'
        url = ("http://files.grouplens.org/datasets/movielens/" + file)
        req = requests.get(url, stream=True)
        print('Downloading MovieLens dataset')
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        with open(os.path.join(self.root, file), mode='wb') as fd:
            for chunk in req.iter_content(chunk_size=None):
                fd.write(chunk)
        with ZipFile(os.path.join(self.root, file), "r") as zip:
            # Extract files
            print("Extracting all the files now...")
            zip.extractall(path=self.root)
            print("Downloading Complete!")

    def _read_ratings_csv(self) -> pd.DataFrame:
        '''
        at first, check if file exists. if it doesn't then call _download().
        it will read ratings.csv, and transform to dataframe.
        it will drop columns=['timestamp'].
        :return:
        '''
        print("Reading file")
        if not os.path.isfile(self.fname):
            self._download_movielens()

        if self.file_size_url == '100k' or self.file_size_url == '20m':
            df = pd.read_csv(self.fname, sep=',')
        else:
            df = pd.read_csv(self.fname, sep="::", header=None,
                               names=['userId', 'movieId', 'ratings', 'timestamp'])
        df = df.drop(columns=['timestamp'])
        print("Reading Complete!")
        return df

    def split_train_test(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        '''
        pick each unique userid row, and add to the testset, delete from trainset.
        :return: (pd.DataFrame,pd.DataFrame,pd.DataFrame)
        '''
        train_dataframe = self.df
        test_dataframe = train_dataframe.sample(frac=1).drop_duplicates(['userId']) #随机从训练集中抽一些数据组成训练集，去除重复项
        train_dataframe = pd.concat([train_dataframe, test_dataframe])
        train_dataframe = train_dataframe.drop_duplicates(keep=False)

        # explicit feedback -> implicit feedback
        # ignore warnings
        ########np.warnings.filterwarnings('ignore')
        # positive feedback (interaction exists)
        train_dataframe.loc[:, 'rating'] = 1
        test_dataframe.loc[:, 'rating'] = 1

        test_dataframe = test_dataframe.sort_values(by=['userId'],axis=0)
        print(f"len(total): {len(self.df)}, len(train): {len(train_dataframe)}, len(test): {len(test_dataframe)}")
        return self.df, train_dataframe, test_dataframe, #将数据集中的评分列（rating）全部设为 1，并将测试集按照 userId 排序。同时，代码打印数据集的长度，并返回处理后的数据集。


class MovieLens(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 total_df: pd.DataFrame,
                 ng_ratio: int,
                 train:bool=False,
                 )->None:
        '''
        :param root: dir for download and train,test.
        :param file_size: large of small. if size if large then it will load(download) 20M dataset. if small then, it will load(download) 100K dataset.
        :param download: if true, it will down load from url.
        '''
        super(MovieLens, self).__init__()

        self.df = df
        self.total_df = total_df
        self.train = train
        self.ng_ratio = ng_ratio
        self.users, self.items = self._negative_sampling()
        print(f'len items:{self.items.shape}')

    def __len__(self) -> int:
        '''
        get lenght of data
        :return: len(data)
        '''
        return len(self.users)


    def __getitem__(self, index):
        '''
        transform userId[index], item[inedx] to Tensor.
        and return to Datalaoder object.
        :param index: idex for dataset.
        :return: user,item,rating
        '''

        # self.items[index][0]: positive feedback
        # self.items[index][1]: negative feedback
        if self.train:
            return self.users[index], self.items[index][0], self.items[index][1]
        else:
            return self.users[index], self.items[index]


    def _negative_sampling(self) :
        '''
        sampling one positive feedback per one negative feedback
        :return: dataframe
        '''
        df = self.df
        total_df = self.total_df
        users, items = [], []
        user_item_set = set(zip(df['userId'], df['movieId']))
        total_user_item_set = set(zip(total_df['userId'],total_df['movieId']))
        all_movieIds = total_df['movieId'].unique()
        # negative feedback dataset ratio
        for u, i in user_item_set:
            # positive instance
            visit = []
            item = []
            if not self.train:
                items.append(i)
                users.append(u)
            else:
                item.append(i)

            for k in range(self.ng_ratio):
                # negative instance
                negative_item = np.random.choice(all_movieIds)
                # check if item and user has interaction, if true then set new value from random
                while (u, negative_item) in total_user_item_set or negative_item in visit:
                    negative_item = np.random.choice(all_movieIds)

                if self.train:
                    item.append(negative_item)
                    visit.append(negative_item)
                else:
                    items.append(negative_item)
                    visit.append(negative_item)
                    users.append(u)

            if self.train:
                items.append(item)
                users.append(u)

        return torch.tensor(users), torch.tensor(items) #返回张量对象



