import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.compose import make_column_selector

def pipline(data):

    cat_cols = data.select_dtypes(include=['object', 'category']).columns.to_list()

    depth_transformer = StandardScaler()

    # Se escalan, se discretizan con KMeans y se transforman con OneHotEncoder
    table_transformer = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('discretizer', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]     
    )

    # Se transforman con operador log, se escalan y se crean variables polinomicas de grado 2
    carat_transformer = Pipeline(
        steps=[
            ('log', FunctionTransformer(np.log1p)),
            ('scaler', StandardScaler())
        ]
    ) 

    # Se transforman con OneHotEncoder
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
                        transformers=[
                            ('carat', carat_transformer, ['carat']),
                            ('depth', depth_transformer, ['depth']),
                            ('table', table_transformer, ['table']),
                            ('categoric', categorical_transformer, cat_cols),
                        ],
                        remainder='passthrough',
                        verbose_feature_names_out = True
                )
    
    return preprocessor




# data = pd.read_csv('data/cleaned_data/cleaned_diamonds.csv')
# data = data.drop(columns=['x', 'y', 'z', 'latitude', 'longitude', 'price'])
# print(data.head())
# preprocessor = pipline(data)
# preprocessor.fit(data)

# model = keras.models.load_model('models/mlp/diamonds_model_mlp.keras')

