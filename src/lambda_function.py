import pandas as pd
from sklearn.covariance import EllipticEnvelope


def anomaly_detection(dataset_path: str):
    """
    Returns a dataframe of anomalies for given timestamps. 
    Anomalies have the value -1. 
    """
    df_hr = pd.read_csv(dataset_path)
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
    data.reset_index().to_csv("data.csv")
    # return data.reset_index().to_json(orient="records")
    return data

def automata(df):
    pass


def filter_anomalies(df):
    df = df.loc[lambda df: df['anomaly'] == -1]
    df.reset_index().to_csv("anomalies.csv")
    return df


def lambda_handler(event, context):
    """
    Main method of the lambda function. Gets called when you send a json to the api. 
    Json will be accessible in the event variable. 
    """
    dataset_path = "test_1.csv"
    return {
        'statusCode': 200,
        'body': anomaly_detection(dataset_path)
    }


if __name__ == "__main__":
    # ret = lambda_handler(None, None)
    dataset_path = "test_1.csv"
    df = anomaly_detection(dataset_path)
    filter_anomalies(df)