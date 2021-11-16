# imports
from re import T
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import set_config; set_config(display='diagram')
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse
from sklearn.model_selection import train_test_split

class Trainer():
    def __init__(self, X, y):
        """
                X: pandas DataFrame
                y: pandas Series
            """
        self.pipeline = None
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y)

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
            ])

        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ])

        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        # Add the model of your choice to the pipeline
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        pipe = self.set_pipeline()
        pipe = pipe.fit(self.X_train, self.y_train)
        return pipe


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        pipe = self.run()
        y_pred = pipe.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    X = df.drop(columns='fare_amount')
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    trainer = Trainer(X, y)
    print(trainer.evaluate(X_test, y_test))
