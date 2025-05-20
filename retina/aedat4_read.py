# env: demo
import dv
file = dv.AedatFile('events.aedat4')
timestamp = []
x = []
y = []
polarity = []
for event in file['events']:
    timestamp.append(event.timestamp)
    x.append(event.x)
    y.append(event.y)
    polarity.append(event.polarity)

import pandas as pd
df = pd.DataFrame({'timestamp': timestamp, 'x': x, 'y': y, 'polarity': polarity})
df.to_csv('events.csv', index=False)
print(df[:10])
print("events.csv created")
print(df['polarity'].isin([0]).any())
file.close()