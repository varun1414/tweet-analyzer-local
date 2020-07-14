import twint
import nest_asyncio
from datetime import datetime
from datetime import timedelta
import pandas as pd
nest_asyncio.apply()
live = twint.Config()
live.Search = "lockdown"
live.Pandas = True
live.Limit = 5
live.Pandas_clean = True
live.Geo = "28.5934,77.2223,2000km"
# live.Since = str(datetime.today() - timedelta(hours=0.5))[:19]  # Do not change
# live.Until = str(datetime.now())[:19]  # Do not change
live.Lang = 'en'
live.Hide_output = True
for i in range(10):
    twint.run.Search(live)
    df_temp = pd.DataFrame(twint.storage.panda.Tweets_df)
    print(df_temp)
#
# df_temp = pd.DataFrame(twint.storage.panda.Tweets_df)
# columns = ['date', 'tweet']

# print(df_temp)


