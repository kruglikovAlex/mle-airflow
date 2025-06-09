from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import CatBoostEncoder

from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv

def create_connection():

    load_dotenv()
    host = 'rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net' # os.environ.get('rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net')
    port = 6432 # os.environ.get('6432')
    db = 'playground_mle_20250507_60d03b0a2f' # os.environ.get('playground_mle_20250507_60d03b0a2f')
    username = 'mle_20250507_60d03b0a2f_freetrack' # os.environ.get('mle_20250507_60d03b0a2f_freetrack')
    password = 'c2538958c7974067a843c0a10811d6db' # os.environ.get('c2538958c7974067a843c0a10811d6db')
    
    print(f'postgresql://{username}:{password}@{host}:{port}/{db}')
    conn = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{db}')
    return conn

# устанавливаем соединение с базой
conn = create_connection()
data = pd.read_sql('select * from clean_users_churn', conn)
print('data.head(): /n' ,data.head())

#  В процессе обучения модели уже не потребуются колонки
#  id, customer_id, begin_date и end_date, поэтому уберём их из датасета.
data.drop(columns=['id','customer_id','begin_date','end_date'], inplace=True)

X_tr, X_val, y_tr, y_val = train_test_split(
    data,
    data['target'],
    stratify=data['target']
)

# Тренировочная выборка
cat_features_tr = X_tr.select_dtypes(include='object')
potential_binary_features_tr = cat_features_tr.nunique() == 2

binary_cat_features_tr = cat_features_tr[potential_binary_features_tr[potential_binary_features_tr].index]
other_cat_features_tr = cat_features_tr[potential_binary_features_tr[~potential_binary_features_tr].index]
num_features_tr = X_tr.select_dtypes(['float'])

# Валидационная выборка
cat_features_val = X_val.select_dtypes(include='object')
potential_binary_features_val = cat_features_val.nunique() == 2

binary_cat_features_val = cat_features_val[potential_binary_features_val[potential_binary_features_val].index]
other_cat_features_val = cat_features_val[potential_binary_features_val[~potential_binary_features_val].index]
num_features_val = X_val.select_dtypes(['float'])

binary_cols = binary_cat_features_tr.columns.tolist()
non_binary_cat_cols = other_cat_features_tr.columns.tolist()
num_cols = num_features_tr.columns.tolist()

preprocessor = ColumnTransformer(
    [
    ('binary', OneHotEncoder(drop='if_binary'), binary_cols),
    ('cat', CatBoostEncoder(), non_binary_cat_cols),
    ('num', StandardScaler(), num_cols)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)
model = CatBoostClassifier(auto_class_weights='Balanced')

# создайте пайплайн
pipeline = Pipeline(
	[
        ('transformer1', preprocessor),
        ('model', model)
    ]
)

# обучите пайплайн
pipeline.fit(X_tr, y_tr)
y_pred = pipeline.predict(X_val)

# получите предсказания для тестовой выборки
y_pred_proba = pipeline.predict_proba(X_val)[:, 1] 

print('f1:', f1_score(y_val, y_pred))
print('roc_auc:', roc_auc_score(y_val, y_pred_proba))

import joblib
pipeline.fit(X_tr, y_tr)

with open('fitted_model.pkl', 'wb') as fd:
    joblib.dump(pipeline, fd)