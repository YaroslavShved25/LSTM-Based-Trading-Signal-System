import os
import json
import pandas as pd
from collections import defaultdict

def load_data(folder_path="data"):
    ticker_dfs = {}
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    files_by_ticker = defaultdict(list)

    for file in json_files:
        ticker = file.split("_")[0].strip().upper()
        files_by_ticker[ticker].append(file)

    for ticker, files in files_by_ticker.items():
        dfs = []
        for f in files:
            with open(os.path.join(folder_path, f)) as j:
                data = json.load(j)
            df = pd.DataFrame(data['response']['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            dfs.append(df.add_prefix(f"{f.split('_')[1].split('.')[0]}_"))

        merged = pd.concat(dfs, axis=1).ffill().dropna()
        merged = merged.apply(pd.to_numeric, errors='coerce').dropna()
        ticker_dfs[ticker] = merged

    return ticker_dfs