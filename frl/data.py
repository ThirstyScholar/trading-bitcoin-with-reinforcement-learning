import numpy as np
import pandas as pd


class Data(object):

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)  # read CSV into DataFrame
        self.feat = None

    def __len__(self):
        return len(self.data)

    def remove_outlier(self):
        idx = pd.datetime(2017, 4, 15, 23)
        self.data.drop(index=idx, inplace=True)
        self.feat.drop(index=idx, inplace=True)

    def preprocess(self):
        """
        Step 1. Create datetime index and select datatime range
        Step 2. Drop columns 'Timestamp', 'Volume_(Currency)' and 'Weighted_Price'
        Step 3. Rename 'Volume_(BTC)' as 'Volume'
        Step 4. Resample to 15-minute bars and drop NaN values

        :return: None
        """

        # Step 1
        self.data.index = pd.to_datetime(self.data['Timestamp'], unit='s')
        self.data = self.data.loc[self.data.index < pd.datetime(2017, 7, 1)]

        # Step 2
        self.data.drop(['Timestamp', 'Volume_(Currency)', 'Weighted_Price'], axis=1, inplace=True)

        # Step 3
        self.data.rename(columns={'Volume_(BTC)': 'Volume'}, inplace=True)

        # Step 4
        self.data = self.data.groupby(pd.Grouper(freq='15Min')).aggregate({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        self.data.dropna(inplace=True)

    def extract_feature(self):
        """
        Step 1. Create an empty feature DataFrame
        Step 2. Calculate features
        Step 3. Drop rows with NaN values
        Step 4. Remove outlier

        :return: None
        """

        # Step 1
        self.feat = pd.DataFrame(index=self.data.index)

        # Step 2
        cls = self.data['Close']
        vol = self.data['Volume']
        np_cls = np.log(cls)

        self.feat['r'] = np_cls.diff()
        self.feat['r_1'] = self.feat['r'].shift(1)
        self.feat['r_2'] = self.feat['r'].shift(2)

        r = self.feat['r']
        self.feat['rZ12'] = Data.zscore(r, 12)
        self.feat['rZ96'] = Data.zscore(r, 96)

        self.feat['pma12'] = Data.zscore(Data.ser2ma_ret(cls, 12), 96)
        self.feat['pma96'] = Data.zscore(Data.ser2ma_ret(cls, 96), 96)
        self.feat['pma672'] = Data.zscore(Data.ser2ma_ret(cls, 672), 96)

        self.feat['ma4/36'] = Data.zscore(Data.ma2ma_ret(cls, 4, 36), 96)
        self.feat['ma12/96'] = Data.zscore(Data.ma2ma_ret(cls, 12, 96), 96)

        self.feat['ac12/12'] = Data.zscore(Data.acceleration(cls, 12, 12), 96)
        self.feat['ac96/96'] = Data.zscore(Data.acceleration(cls, 96, 12), 96)

        self.feat['vZ12'] = Data.zscore(vol, 12)
        self.feat['vZ96'] = Data.zscore(vol, 96)
        self.feat['vZ672'] = Data.zscore(vol, 672)

        self.feat['vma12'] = Data.zscore(Data.ser2ma_ret(vol, 12), 96)
        self.feat['vma96'] = Data.zscore(Data.ser2ma_ret(vol, 96), 96)
        self.feat['vma672'] = Data.zscore(Data.ser2ma_ret(vol, 672), 96)

        vola_12 = Data.roll_std(r, 12)  # 12-period volatility
        vola_96 = Data.roll_std(r, 96)
        vola_672 = Data.roll_std(r, 672)
        self.feat['vol12'] = Data.zscore(vola_12, 96)
        self.feat['vol96'] = Data.zscore(vola_96, 96)
        self.feat['vol672'] = Data.zscore(vola_672, 96)

        self.feat['dv12/96'] = Data.zscore(Data.ser2ma_ret(vola_12, 96), 96)
        self.feat['dv96/672'] = Data.zscore(Data.ser2ma_ret(vola_96, 672), 96)

        # Step 3
        self.feat.dropna(inplace=True)
        self.data = self.data.loc[self.feat.index]  # select data where feat are available

        # Step 4
        self.remove_outlier()

    @staticmethod
    def roll_mean(s, window):
        """
        :param s: Pandas Series
        :param window: int
        :return: Pandas Series
        """
        return s.rolling(window).mean()

    @staticmethod
    def roll_std(s, window):
        """
        :param s: Pandas Series
        :param window: int
        :return: Pandas Series
        """
        return s.rolling(window).std()

    @staticmethod
    def zscore(s, window):
        """
        :param s: Pandas Series
        :param window: int
        :return: Pandas Series
        """
        roll_mean = s.rolling(window).mean()
        roll_std = s.rolling(window).std()
        return (s - roll_mean) / (roll_std + 1e-6)

    @staticmethod
    def ser2ma_ret(s, window):
        """
        Series-to-Moving Average return.
        :param s: Pandas Series
        :param window: int
        :return: Pandas Series
        """
        roll_mean = s.rolling(window).mean()
        return (s - roll_mean) - 1

    @staticmethod
    def ma2ma_ret(s, window_1, window_2):
        """
        Series-to-series return.
        :param s: Pandas Series
        :param window_1: int
        :param window_2: int
        :return: Pandas Series
        """
        return s.rolling(window_1).mean() / s.rolling(window_2).mean() - 1

    @staticmethod
    def acceleration(s, window_1, window_2):
        """
        See the definition from the original post "https://launchpad.ai/blog/trading-bitcoin"
        :param s: Pandas Series
        :param window_1: int
        :param window_2: int
        :return: Pandas Series
        """
        tmp = s / s.rolling(window_1).mean()
        return tmp / tmp.rolling(window_2).mean()


def test_data():
    data_path = './bitcoin-historical-data/coinbaseUSD_1-min_data.csv'
    data = Data(data_path)
    data.preprocess()
    data.extract_feature()


if __name__ == '__main__':
    test_data()
