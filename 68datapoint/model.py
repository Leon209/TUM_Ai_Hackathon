import pandas as pd
from sklearn.covariance import EllipticEnvelope

df_hr = pd.read_csv("data/test_1.csv")
rhr = df_hr["heart_rate"].median()
 # heart rate data
df_hr = df_hr.set_index('timestamp')
df_hr.index.name = None
df_hr.index = pd.to_datetime(df_hr.index)
# fit the model  ------------------------------------------------------
model = EllipticEnvelope(random_state=10,
                            support_fraction=0.7,
                            contamination=0.1)
model.fit(df_hr)
# predict the test set
preds = pd.DataFrame(model.predict(df_hr))
preds = preds.rename(lambda x: 'anomaly' if x == 0 else x, axis=1)
data = df_hr.reset_index()
data = data.join(preds)
data.to_csv("data.csv")
