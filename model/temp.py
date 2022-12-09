import pandas as pd

df = pd.read_csv('cs551.csv', parse_dates=['date_time'])
df['weekday'] = df.date_time.map(lambda x: x.weekday())
df['year'] = df.date_time.map(lambda x: x.year)
df['month'] = df.date_time.map(lambda x: x.month)

print(df.temp.describe())
print(sum(df.temp < 200))
