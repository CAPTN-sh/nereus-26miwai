import pandas as pd

def transform(path):
    raw = pd.read_pickle(path)

    data = []
    for traj_id, row in enumerate(raw):
        for point in row["traj"]:
            data.append({
                "time": point[4],
                "traj_id": traj_id,
                "lat": point[0],
                "long": point[1]
            })
    df = pd.DataFrame(data)

    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values(['traj_id', 'time'])
    df = df.set_index('time')

    resampled = (
        df.groupby('traj_id')
        .resample('1T')
        .mean()
        .interpolate()
        .drop('traj_id', axis = 1)
        .reset_index()
    )
    df = resampled[resampled['time'].dt.minute % 10 == 0]
    df.loc[:, 'time'] = (
        (df['time'] - df['time'].min())
        .dt.total_seconds() / 600
    ).astype(int) + 1

    df = df[(df["lat"] <= 0.25) & (df["long"] >= 0.75)]
    
    traj_counts = df['traj_id'].value_counts()
    valid_traj_ids = traj_counts[traj_counts >= 30].index
    df = df[df['traj_id'].isin(valid_traj_ids)].reset_index(drop=True)

    df.loc[:, "lat"] = df["lat"] * 40
    df.loc[:, "long"] = (df["long"] - 0.75) * 40
    df = df[["time", "traj_id", "lat", "long"]].sort_values(["time", "traj_id"])
    df.to_csv("./dataset/denmark/train/train.csv", index=False, header=False)

path = "./dataset/denmark/raw/train.pkl"
transform(path)