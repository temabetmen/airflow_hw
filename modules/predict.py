import os

import dill
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '~/airflow_hw')

def last_model(path):
    # Получим список имен всего содержимого папки
    # и превратим их в абсолютные пути
    dir_list = [os.path.join(path, x) for x in os.listdir(path)]

    if dir_list:
        # Создадим список из путей к файлам и дат их создания.
        date_list = [[x, os.path.getctime(x)] for x in dir_list]

        # Возьмем максимальный
        sort_date_list = max(date_list, key=lambda x: x[1])

        # Выведем первый элемент
        return sort_date_list[0]


def files_folder(path):
    # Получим список имен всего содержимого папки
    # и превратим их в абсолютные пути
    dir_list = [os.path.join(path, x) for x in os.listdir(path)]

    if dir_list:
        return dir_list


# Загружаем последнию по дате модель

with open(last_model(f'{path}/data/models/'), 'rb') as file:
    model = dill.load(file)


def predict():
    list_predict = list()
    list_id = list()
    result = pd.DataFrame()
    for file_json in files_folder(f'{path}/data/test'):
        if Path(file_json).suffix == '.json':  # Проверка на разширение файла
                                                # избавится от краша если что-то пападет в папку не то
            with open(file_json, 'r') as fh:
                data_json = json.load(fh)
                df = pd.DataFrame.from_dict([data_json])
                y = model.predict(df)
                list_predict.append(y[0])
                list_id.append(df.id[0])

    result['car_id'] = list_id
    result['pred'] = list_predict
    result_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    return result.to_csv(result_filename, index=False)


if __name__ == '__main__':
    predict()
