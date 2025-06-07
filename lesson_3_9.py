# Спринт 1/11: 1 спринт. Разработка пайплайнов подготовки данных
# и обучения модели → Тема 3/5: Создание пайплайна обучения 
# ML-модели → Урок 3/9
#
# ПОИСК И РЕШЕНИЕ ПРОБЛЕМ В ДАННЫХ
#
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
print('conn: ', conn)

# выгружаем датасет
data = pd.read_sql('select * from public.users_churn', conn)
print('data.shape; ', data.shape)
print('data.head(): ', data.head())