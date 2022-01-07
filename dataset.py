import pandas as pd


class Dataset:
    def __init__(self, country='Poland', begin_date='2020-09-27',  # '2020-09-07' - '2020-11-25'
                 end_date='2020-11-26'):
        self.country = country
        self.begin_date = begin_date
        self.end_date = end_date
        self.raw_data = pd.read_csv('owid-covid-data.csv', parse_dates=True)
        self.data = self.load()
        self.load_recovered()
        #self.data_smooth = self.data.rolling('7D').mean()
        #self.data_smoother = self.data.rolling('14D').mean()

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        e = [
            'date', 'location', 'total_cases', 'new_cases', 'total_deaths',
            'new_deaths', 'reproduction_rate', 'population', 'population_density',
            'new_tests', 'total_tests', 'positive_rate', 'tests_per_case',
        ]
        return df[e]

    def extract_country(df: pd.DataFrame, *countries: str) -> pd.DataFrame:
        return df[df['location'] == countries]

    def load(self) -> pd.DataFrame:
        d = self.extract_features(self.raw_data)
        d = d[d['location'] == self.country]
        d['date'] = pd.to_datetime(d['date'], format='%Y-%m-%d')
        d.fillna(0.0, inplace=True)
        d = d.set_index('date')
        d = d.asfreq('D')
        date_range = pd.date_range(start=self.begin_date, end=self.end_date, freq='D')
        d = d.loc[date_range]
        return d

    def load_recovered(self):
        df = pd.read_csv('covid_recovered.csv', parse_dates=True, delimiter=',')
        df['Country/Region'].replace({'Korea, South': 'South Korea'}, inplace=True)
        df = df[df['Country/Region'] == self.country]
        df = df.drop(columns=['Province/State', 'Lat', 'Long', 'Country/Region'])
        df = df.fillna(0)
        df.reset_index(inplace=True, drop=True)
        df = df.transpose()
        df.rename(columns={0: 'recovered'}, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.diff().fillna(0)
        date_range = pd.date_range(start=self.begin_date, end=self.end_date, freq='D')
        df = df.loc[date_range]
        self.data = df.merge(self.data, left_index=True, right_index=True, how='inner')
