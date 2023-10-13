import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
from tqdm import tqdm
from collections import Counter
from df_addons import df_to_excel

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from df_addons import memory_compression
from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')

GATES_DIR = Path(r'D:\python-datasets\gates-2')

if not GATES_DIR.exists():
    GATES_DIR = Path('.')
    __file__ = Path('.')

PREDICTIONS_DIR = GATES_DIR.joinpath('predictions')

# Проверяем, существует ли каталог -> Если каталог не существует, создаем его
if not PREDICTIONS_DIR.exists():
    PREDICTIONS_DIR.mkdir(parents=True)


def read_all_df(file_dir=None):
    """
    Чтение трейна и теста и объединение их в один ДФ
    :param file_dir: путь к файлам
    :return: объединенный ДФ
    """
    if file_dir is None:
        file_dir = Path(__file__).parent
    file_dir = Path(file_dir)

    file_train = file_dir.joinpath('train.csv')
    file_test = file_dir.joinpath('test.csv')

    train_df = pd.read_csv(file_train, parse_dates=['ts'], index_col=0)
    test_df = pd.read_csv(file_test, parse_dates=['ts'], index_col=0)
    test_df.insert(0, 'user_id', -1)

    all_df = pd.concat([train_df, test_df])
    return all_df


def get_max_num(file_logs=None):
    """Получение максимального номера итерации обучения моделей
    :param file_logs: имя лог-файла с полным путем
    :return: максимальный номер
    """
    if file_logs is None:
        file_logs = GATES_DIR.joinpath('scores_local.logs')

    if not file_logs.is_file():
        with open(file_logs, mode='a') as log:
            log.write('num;mdl;roc_auc;acc_train;acc_valid;acc_full;score;WF1;'
                      'model_columns;exclude_columns;cat_columns;comment\n')
        max_num = 0
    else:
        df = pd.read_csv(file_logs, sep=';')
        if 'acc_train' not in df.columns:
            df.insert(2, 'acc_train', 0)
            df.insert(3, 'acc_valid', 0)
            df.insert(4, 'acc_full', 0)
            df.num = df.index + 1
            df.to_csv(file_logs, index=False, sep=';')
        if 'roc_auc' not in df.columns:
            df.insert(2, 'roc_auc', 0)
            df.num = df.index + 1
            df.to_csv(file_logs, index=False, sep=';')
        df.num = df.index + 1
        max_num = df.num.max() if len(df) else 0
    return max_num


def predict_train_valid(model, datasets, label_enc=None):
    """Расчет метрик для модели: accuracy на трейне, на валидации, на всем трейне, roc_auc
    и взвешенная F1-мера на валидации
    :param model: обученная модель
    :param datasets: кортеж с тренировочной, валидационной и полной выборками
    :param label_enc: используемый label_encоder для target'а
    :return: accuracy на трейне, на валидации, на всем трейне, roc_auc и взвешенная F1-мера
    """
    X_train, X_valid, y_train, y_valid, train, target, test_df = datasets
    valid_pred = model.predict(X_valid)
    train_pred = model.predict(X_train)
    train_full = model.predict(train)

    if len(valid_pred.shape) > 1 and valid_pred.shape[1] > 1:
        predict_proba = valid_pred.copy()
        valid_pred = np.argmax(valid_pred, axis=1)
    else:
        predict_proba = model.predict_proba(X_valid)

    if len(train_pred.shape) > 1 and train_pred.shape[1] > 1:
        train_pred = np.argmax(train_pred, axis=1)
    if len(train_full.shape) > 1 and train_full.shape[1] > 1:
        train_full = np.argmax(train_full, axis=1)

    f1w = f1_score(y_valid, valid_pred, average='weighted')
    acc_valid = accuracy_score(y_valid, valid_pred)
    acc_train = accuracy_score(y_train, train_pred)
    acc_full = accuracy_score(target, train_full)
    try:
        roc_auc = roc_auc_score(y_valid, predict_proba,
                                average='weighted', multi_class='ovr')
    except:
        roc_auc = 0

    # print(classification_report(y_valid, valid_pred))
    return acc_train, acc_valid, acc_full, roc_auc, f1w


def find_predict_words(file_submit_csv, test_df, user_id_max=60):
    """
    Процедура формирования предсказания user_id на основа самого частотного во всей массе
    предсказаний и охранение в файл xxx.tst.csv
    :param file_submit_csv: имя файл сабмита с предсказанными user_id
    :param test_df: тестовый ДФ
    :param user_id_max: максимальный номер user_id из трейна
    :return: новый сабмит с user_word
    """
    file_dir = Path(__file__).parent

    all_df = read_all_df(file_dir)
    data_cls = DataTransform2()
    grp = data_cls.fit_days_mask(all_df, show_messages=False, remove_double_gate=True)
    # TO DO


def find_predict_words_new(file_submit_csv, test_df, user_id_max=60):
    """
    Новая версия с удалением дублей user_id в сабмите
    Процедура формирования предсказания user_id на основа самого частотного во всей массе
    предсказаний и исключение дублей user_id для разных слов и сохранение в файл xxx.tst.csv
    :param file_submit_csv: имя файл сабмита с предсказанными user_id
    :param test_df: тестовый ДФ
    :param user_id_max: максимальный номер user_id из трейна
    :return: новый сабмит с user_word
    """
    file_dir = Path(__file__).parent

    all_df = read_all_df(file_dir)
    data_cls = DataTransform2()
    grp = data_cls.fit_days_mask(all_df, show_messages=False, remove_double_gate=True,
                                 drop_december=True)

    if data_cls.out_user_id is None:
        out_user_id = [4, 51, 52]
    else:
        out_user_id = data_cls.out_user_id

    # пока зафиксируем эти user_id для удаления
    out_user_id = [4, 51, 52]
    out_user_id = []

    print(f'data_cls.out_user_id: len={len(out_user_id)} -> {out_user_id}')

    submit = pd.read_csv(file_submit_csv, index_col=0)
    submit['user_word'] = test_df['user_word'].astype('object')

    # print(submit)
    # print(test_df['user_word'])
    # print('user_id_max:', user_id_max)

    # print('out_user_id == target')
    # print(submit[submit['target'].isin(out_user_id)].target.value_counts())
    # print(submit[submit['target'].isin(out_user_id)].user_word.value_counts())

    # без user_id, которых не было в декабре
    words = submit[~submit.target.isin(out_user_id)] \
        .groupby('user_word', as_index=False) \
        .agg(p_values=('target', lambda x: x.value_counts(normalize=True)
                       .reset_index().values.tolist()))
    words.insert(1, 'p_value', 0)
    words.insert(1, 'pred', -999)
    # value_counts выдал в виде кортежа ('user_id', 'p_value'), user_id преобразуем в int
    words['p_values'] = words['p_values'].map(
        lambda x: [*map(lambda z: (int(z[0]), z[1]), x)])
    words['p_values'] = words['p_values'].map(
        lambda x: sorted(x, key=lambda k: (-k[1], k[0])))
    # сортируем кортежи в порядке убывания p_value и возрастанию user_id
    words['p_dbl'] = words['p_values'].map(
        lambda x: False if len(x) < 2 else x[0][1] == x[1][1])
    # выделим в отдельную колонку первый pred user_id
    words['pred'] = words['p_values'].map(lambda x: x[0][0]).astype(int)
    # выделим в отдельную колонку первое p_value
    words['p_value'] = words['p_values'].map(lambda x: x[0][1])
    # сохраним для истории эти значения, т.к. дальше будем их затирать
    words['pred_old'] = words['pred']
    words['p_values_old'] = words['p_values']

    # преобразуем в ДФ максимальные p_value для каждого user_id
    p_values = sum(words['p_values'].tolist(), [])
    p_values = pd.DataFrame(p_values, columns=['pred', 'p_value'])
    p_values = p_values.groupby('pred', as_index=False).p_value.max() \
        .sort_values('p_value', ascending=False)

    df_to_excel(p_values, file_dir.joinpath('p_values.xlsx'), float_cells=[2])
    # print(words)
    df_to_excel(words, file_dir.joinpath('words.xlsx'), float_cells=[3])

    copy_words = words.copy()

    uses_preds = {}  # словарь для хранения обработанных user_id для user_word
    iteration = 0  # порядковый номер итерации
    user_pred = set(words['pred'].tolist())
    prev_user_pred = set()
    while len(user_pred) and len(p_values) and prev_user_pred != user_pred:
        iteration += 1
        df_pv = p_values.merge(words[['user_word', 'pred', 'p_value']],
                               on=['pred', 'p_value'], how='left')
        # print(df_pv)

        # запишем в ДФ найденные user_id для user_word
        pv = df_pv.dropna()
        # преобразуем ДФ в словарь
        pv_dict = {row['pred']: (row['user_word'], row['p_value']) for _, row in
                   pv.iterrows()}
        # найденные user_word запишем в список
        pv_word = pv.user_word.tolist()
        # добавим найденные user_id для user_word в словарь  uses_preds обработанных user_id
        uses_preds.update(pv_dict)

        # print(uses_preds)
        print(sorted(pv_word), sorted(pv_dict.keys()), sep='\n')

        prev_user_pred = user_pred
        # удалим из исходного ДФ найденные слова
        words = words[~words.user_word.isin(pv_word)]
        # удалим из списка вероятностей те user_id, которые уже обработаны, есть в uses_preds
        words['p_values'] = words['p_values'].apply(
            lambda x: [(k, v) for k, v in x if k not in uses_preds])
        # запишем в 'pred' следующий user_id по порядку, если список пуст -> -999
        words['pred'] = words['p_values'].apply(lambda x: x[0][0] if len(x) > 0 else -999)
        # запишем в 'pred' следующую вероятность по порядку, если список пуст -> 0
        words['p_value'] = words['p_values'].apply(lambda x: x[0][1] if len(x) > 0 else 0)
        # теперь надо повторить действия с "запишем в ДФ найденные user_id для user_word"

        df_to_excel(words, file_dir.joinpath(f'words{iteration:02}.xlsx'), float_cells=[3])
        user_pred = set(words['pred'].tolist())
        if prev_user_pred == user_pred:
            p_values = words[words.pred.gt(-9)].groupby('pred', as_index=False).p_value.max()
            print('p_values по новому алгоритму:\n', p_values)
            prev_user_pred = set()

    print('Длина uses_preds:', len(uses_preds), uses_preds, sep='\n')
    print(words)

    # Для неопределенных user_id
    no_user_id = words[words.pred == -999]
    if len(no_user_id):
        print('no_user_id:\n', no_user_id)
        for index, row in no_user_id.iterrows():
            tmp = grp[grp.user_id == row.user_word]
            print('row.user_word:', row.user_word)
            time_start = tmp.time_start.min() * 0.8
            time_end = tmp.time_end.max() * 1.2
            temp = grp[
                (pd.to_datetime(grp['first_show']).dt.date > pd.to_datetime(
                    '2022-12-01').date())
                & (pd.to_datetime(grp['date']).dt.date < pd.to_datetime('2023-01-01').date())
                & ~grp.user_id.isin(uses_preds.keys())
                & (grp.time_start > time_start) & (grp.time_end < time_end)
                ]
            if not len(temp):
                temp = grp[
                    (pd.to_datetime(grp['date']).dt.date < pd.to_datetime(
                        '2023-01-01').date())
                    & ~grp.user_id.isin(uses_preds.keys())
                    & (grp.time_start > time_start) & (grp.time_end < time_end)
                    ]
            temp = temp.sort_values('first_show', ascending=False).reset_index(drop=True)
            user_id = temp.at[0, 'user_id']
            words.at[index, 'pred'] = user_id

            # добавим найденные user_id для user_word в словарь uses_preds
            uses_preds[user_id] = (row['user_word'], row['p_value'])

            # print(temp)
            df_to_excel(temp, file_dir.joinpath('grp.xlsx'))

        iteration += 1
        df_to_excel(words, file_dir.joinpath(f'words{iteration:02}.xlsx'), float_cells=[3])

    print('Длина uses_preds:', len(uses_preds), uses_preds, sep='\n')

    new_words = {val[0]: key for key, val in uses_preds.items()}
    new_words = pd.DataFrame(sorted(new_words.items()), columns=['user_word', 'preds'])
    print(new_words)

    new_words.to_csv(file_submit_csv.with_suffix('.words.tst_new.csv'), index=False)
    return words


def predict_test(idx_fold, model, datasets, max_num=0, submit_prefix='lg_', label_enc=None,
                 save_predict_proba=True):
    """Предсказание для тестового датасета.
    Расчет метрик для модели: accuracy на трейне, на валидации, на всем трейне, roc_auc
    и взвешенная F1-мера на валидации
    :param idx_fold: номер фолда при обучении
    :param model: обученная модель
    :param datasets: кортеж с тренировочной, валидационной и полной выборками
    :param max_num: максимальный порядковый номер обучения моделей
    :param submit_prefix: префикс для файла сабмита для каждой модели свой
    :param label_enc: используемый label_encоder для target'а
    :param save_predict_proba: сохранять файл с вероятностями предсказаний
    :return: accuracy на трейне, на валидации, на всем трейне, roc_auc и взвешенная F1-мера
    """
    X_train, X_valid, y_train, y_valid, train, target, test_df = datasets

    test = test_df.copy()

    # выделим колонку с user_word
    if 'user_word' in test.columns:
        user_words = test['user_word'].astype('object')
        test.drop('user_word', axis=1, inplace=True)
    else:
        print("Ошибка!!! В test.columns нет 'user_word'!!!")
        user_words = pd.DataFrame(index=test.index)

    # постфикс если было обучение на отдельных фолдах
    nfld = f'_{idx_fold}' if idx_fold else ''
    predictions = model.predict(test)
    predict_train = model.predict(train)

    if label_enc:
        predictions = label_enc.inverse_transform(predictions)
        predict_train = label_enc.inverse_transform(predict_train)

    # печать размерности предсказаний и списка меток классов
    classes = model.classes_.tolist()
    print('predict_proba.shape:', predictions.shape, 'classes:', classes)

    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predict_proba = predictions.copy()
        predictions = np.argmax(predictions, axis=1)
        train_proba = predict_train.copy()
        predict_train = np.argmax(predict_train, axis=1)
    else:
        predict_proba = model.predict_proba(test)
        train_proba = model.predict_proba(train)

    submit_csv = f'{submit_prefix}submit_{max_num:03}{nfld}_local.csv'
    file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
    file_proba_csv = PREDICTIONS_DIR.joinpath(submit_csv.replace('submit_', 'proba_'))
    file_train_csv = PREDICTIONS_DIR.joinpath(submit_csv.replace('submit_', 'train_'))

    # Сохранение предсказаний в файл
    submit = test[test.columns.to_list()[5:7]]
    # print('submit.columns:', submit.columns)  # ['last_day', 'is_weekend']
    submit['target'] = predictions
    submit[['target']].to_csv(file_submit_csv)

    print('submit:')
    print(submit)
    print('user_words:')
    print(user_words)

    try:
        # формирование ответа
        submit = pd.read_csv(file_submit_csv, index_col=0)
        submit['user_word'] = user_words
        words = submit.groupby('user_word', as_index=False).agg(
            preds=('target', lambda x: x.value_counts().index[0])
        )
        words.to_csv(file_submit_csv.with_suffix('.words.csv'), index=False)

        find_predict_words_new(file_submit_csv, test_df, target.max())

    except IndexError:
        print('Ошибка!!! Что-то пошло не так при формировании words.csv')

    if save_predict_proba:
        train_sp = pd.DataFrame(target)
        train_sp['target'] = predict_train
        train_sp.to_csv(file_train_csv)

        try:
            train_sp[classes] = train_proba
            train_sp.to_csv(file_train_csv)
        except:
            pass

        try:
            proba = submit[['target']]
            proba[classes] = predict_proba
            proba.to_csv(file_proba_csv)
        except:
            pass

    acc_train, acc_valid, acc_full, roc_auc, f1w = predict_train_valid(model, datasets,
                                                                       label_enc=label_enc)
    print(f'Accuracy = {acc_valid:.6f}')
    print(f'Weighted F1-score = {f1w:.6f}')
    print(f'Accuracy train:{acc_train} valid:{acc_valid} full:{acc_full} roc_auc:{roc_auc}')
    return acc_train, acc_valid, acc_full, roc_auc, f1w


class DataTransform:
    def __init__(self, use_catboost=False, numeric_columns=None, category_columns=None,
                 drop_first=False, scaler=None, args_scaler=None):
        """
        Преобразование данных
        :param use_catboost: данные готовятся для catboost
        :param numeric_columns: цифровые колонки
        :param category_columns: категориальные колонки
        :param drop_first: из dummy переменных удалить первую колонку
        :param scaler: какой скейлер будем использовать
        :param degree: аргументы для скейлера, например: степень для полином.преобразования
        """
        self.use_catboost = use_catboost
        self.category_columns = [] if category_columns is None else category_columns
        self.numeric_columns = [] if numeric_columns is None else numeric_columns
        self.drop_first = drop_first
        self.exclude_columns = []
        self.new_columns = []
        self.comment = {'drop_first': drop_first}
        self.train_months = (7, 8, 9, 10, 11)
        self.valid_months = (12,)
        self.train_idxs = None
        self.valid_idxs = None
        self.transform_columns = None
        self.scaler = scaler
        self.args_scaler = args_scaler
        self.preprocess_path_file = None
        self.beep_outlet = None
        self.gates_mask = [(-1, -1, -1), (-1, -1, 10), (-1, -1, 11), (3, 3, 4), (3, 3, 10),
                           (3, 3, 10, 11), (4, 4, 3), (4, 4, 4), (4, 4, 5), (4, 4, 7),
                           (4, 4, 8), (4, 4, 9, 9), (4, 4, 9, 9, 5, 5), (4, 7, 3), (4, 9, 9),
                           (5, 5, 10), (5, 10, 11), (6, 3, 3), (6, 6, 5), (6, 6, 7),
                           (6, 6, 9, 9), (6, 7, 3), (6, 9, 9), (7, 3, 3), (7, 3, 3, 10),
                           (7, 3, 3, 10, 11), (7, 3, 3, 11), (7, 3, 10), (7, 5, 5),
                           (7, 5, 5, 10), (7, 5, 5, 10, 11), (8, 8, 5), (7, 8, 8), (7, 9, 9),
                           (7, 9, 9, 3, 3), (7, 9, 9, 5, 5), (7, 9, 9, 5, 5, 5),
                           (7, 9, 9, 5, 5, 10), (9, 5, 5), (9, 5, 5, 10), (9, 9, 3),
                           (9, 9, 5), (9, 9, 5, 5), (9, 9, 5, 5, 10), (9, 9, 15),
                           (10, 11, 4, 4), (10, 11, 6, 6), (10, 11, 10), (10, 13, 13),
                           (11, 4, 4), (11, 4, 4, 3), (11, 4, 4, 3, 3), (11, 4, 4, 3, 3, 10),
                           (11, 4, 4, 4), (11, 4, 4, 4, 4), (11, 4, 4, 5), (11, 4, 4, 5, 5),
                           (11, 4, 4, 5, 5, 10), (11, 4, 4, 7), (11, 4, 4, 7, 3, 3),
                           (11, 4, 4, 7, 5), (11, 4, 4, 8, 8), (11, 4, 4, 9, 9),
                           (11, 4, 4, 9, 9, 5), (11, 4, 4, 9, 9, 15), (11, 4, 4, 15),
                           (11, 6, 6), (11, 6, 6, 5), (11, 6, 6, 6), (11, 6, 6, 9, 9),
                           (11, 10, 11), (11, 10, 11, 4), (11, 11, 4, 4), (11, 11, 4, 4, 9),
                           (11, 11, 10), (12, 12, 11), (12, 12, 11, 4), (13, 13, 4, 4),
                           (13, 13, 6, 6), (13, 13, 10), (13, 13, 11), (13, 13, 12, 12),
                           (13, 13, 12, 12, 11), (15, 3, 3), (15, 3, 3, 10),
                           (15, 3, 3, 10, 11), (15, 9, 9), (15, 9, 9, 5, 5),
                           ]
        self.gates_M_V2 = [(-1, -1, -1), (-1, -1, -1, -1), (-1, -1, 10), (-1, -1, 11),
                           (3, 3, 4), (3, 3, 10), (3, 3, 10, 11), (3, 3, 10, 11, 4),
                           (3, 3, 10, 11, 6), (3, 4, 4), (3, 10, 11), (3, 10, 11, 4, 4),
                           (3, 10, 11, 6), (4, 3, 3), (4, 3, 3, 10), (4, 4, 3), (4, 4, 4),
                           (4, 4, 3, 3, 10), (4, 4, 4, 9), (4, 4, 5), (4, 4, 5, 5, 10),
                           (4, 4, 7), (4, 4, 7, 3, 3), (4, 4, 7, 5), (4, 4, 8), (4, 4, 9, 9),
                           (4, 4, 9, 9, 5, 5), (4, 4, 9, 9, 15), (4, 4, 15), (4, 5, 5),
                           (4, 5, 5, 10), (4, 7, 3), (4, 7, 3, 3, 10), (4, 7, 5), (4, 8, 8),
                           (4, 9, 9), (4, 9, 9, 5, 5), (4, 9, 9, 5, 5, 10), (4, 9, 9, 15),
                           (5, 5, 5), (5, 5, 5, 10), (5, 5, 10), (5, 5, 10, 11), (5, 10, 11),
                           (5, 5, 10, 11, 4, 4), (5, 5, 10, 11, 6), (5, 5, 10, 13),
                           (5, 10, 11, 4, 4), (5, 10, 11, 4, 4, 9), (5, 10, 11, 6, 6),
                           (5, 10, 13), (6, 3, 3), (6, 5, 5), (6, 6, 5), (6, 6, 6),
                           (6, 6, 6, 6, 9), (6, 6, 7), (6, 6, 9, 9), (6, 7, 3), (6, 9, 9),
                           (6, 9, 9, 5), (7, 3, 3), (7, 3, 3, 10), (7, 3, 3, 10, 11),
                           (7, 3, 3, 10, 11, 4), (7, 3, 10), (7, 5, 5), (7, 5, 5, 10),
                           (7, 5, 5, 10, 11), (7, 8, 8), (7, 9, 9), (7, 9, 9, 3, 3),
                           (7, 9, 9, 5, 5), (7, 9, 9, 5, 5, 5), (7, 9, 9, 5, 5, 10),
                           (8, 8, 3), (8, 8, 5), (9, 3, 3), (9, 5, 5), (9, 5, 5, 5, 5),
                           (9, 5, 5, 10), (9, 5, 5, 10, 11), (9, 5, 5, 10, 11, 4), (9, 9, 3),
                           (9, 5, 5, 10, 11, 6), (9, 5, 5, 10, 13), (9, 9, 5), (9, 9, 5, 5),
                           (9, 9, 5, 5, 5, 5), (9, 9, 5, 5, 10), (9, 9, 15), (10, 11, 4, 4),
                           (10, 11, 4, 4, 9, 9), (10, 11, 6, 6), (10, 11, 10), (10, 13, 13),
                           (11, 4, 4), (11, 4, 4, 3), (11, 4, 4, 3, 3), (11, 4, 4, 3, 3, 10),
                           (11, 4, 4, 4), (11, 4, 4, 4, 4), (11, 4, 4, 5), (11, 4, 4, 5, 5),
                           (11, 4, 4, 5, 5, 10), (11, 4, 4, 7), (11, 4, 4, 7, 3, 3),
                           (11, 4, 4, 7, 5), (11, 4, 4, 8, 8), (11, 4, 4, 9, 9), (11, 6, 6),
                           (11, 4, 4, 9, 9, 5), (11, 4, 4, 9, 9, 15), (11, 4, 4, 15),
                           (11, 6, 6, 5), (11, 6, 6, 6), (11, 6, 6, 9, 9), (11, 10, 11),
                           (11, 10, 11, 4), (11, 11, 4, 4), (11, 11, 4, 4, 9), (12, 12, 11),
                           (12, 11, 4, 4), (12, 12, 11, 4), (13, 4, 4), (13, 12, 12),
                           (13, 13, 4, 4), (13, 13, 6, 6), (13, 13, 10), (13, 13, 11),
                           (13, 13, 12, 12), (13, 13, 12, 12, 11), (15, 3, 3),
                           (15, 3, 3, 10), (15, 3, 3, 10, 11), (15, 9, 9), (15, 9, 9, 5, 5),
                           ]
        self.gates_mask_count_2_4 = None
        self.gates_mask_count_ge5 = None

    def initial_preparation(self, df, out_five_percent=False):
        """
        Общая первоначальная подготовка данных
        :param df: исходный ДФ
        :param out_five_percent: граница 5% при определении выбросов
        :return: обработанный ДФ
        """
        df["date"] = df['ts'].dt.date
        df["time"] = df['ts'].dt.time
        df["day"] = df['ts'].dt.day
        df["week"] = df['ts'].dt.week
        df["month"] = df['ts'].dt.month

        df["hour"] = df['ts'].dt.hour
        df["min"] = df['ts'].dt.minute
        df["sec"] = df['ts'].dt.second

        df['minutes'] = df["hour"] * 60 + df["min"]
        df['seconds'] = df.minutes * 60 + df["sec"]

        # 1-й день месяца
        df["1day"] = df['ts'].dt.is_month_start.astype(int)
        # 2-й день месяца
        df["2day"] = (df.day == 2).astype(int)
        # предпоследний день месяца
        df["last_day-1"] = (df.day == df.ts.dt.daysinmonth - 1).astype(int)
        # Последний день месяца
        df["last_day"] = df['ts'].dt.is_month_end.astype(int)

        df["weekday"] = df['ts'].dt.dayofweek  # День недели от 0 до 6
        df["dayofweek"] = df["weekday"] + 1  # День недели от 1 до 7

        # Метка выходного дня
        df["is_weekend"] = df["weekday"].map(lambda x: 1 if x in (5, 6) else 0)

        # метки "график 2 через 2"
        df["DofY1"] = (df['ts'].dt.dayofyear % 4).apply(lambda x: int(x in (1, 2)))
        df["DofY2"] = (df['ts'].dt.dayofyear % 4).apply(lambda x: int(x < 2))

        # df['morning'] = df['hour'].map(lambda x: 1 if 6 <= x <= 10 else 0)
        # df['daytime'] = df['hour'].map(lambda x: 1 if 11 <= x <= 17 else 0)
        # df['evening'] = df['hour'].map(lambda x: 1 if 18 <= x <= 22 else 0)
        # df['night'] = df['hour'].map(lambda x: 1 if 0 <= x <= 5 or x == 23 else 0)

        # Подсчет количества срабатываний за день
        df["beep_count"] = df.groupby("date").ts.transform("count")
        # Подсчет количества срабатываний за день по каждому gate_id
        df["beep_gate"] = df.groupby(["date", "gate_id"]).ts.transform("count")

        # данные для устранения выбросов, где рабочий день помечен как выходной и наоборот
        tmp = df[['date', 'weekday', 'beep_count', 'is_weekend']].drop_duplicates()
        tmp["weekend"] = tmp["weekday"].map(lambda x: 1 if x in (5, 6) else 0)
        beep_cnt = tmp[tmp["weekend"] == 1].beep_count
        if out_five_percent:
            self.beep_outlet = beep_cnt.quantile(0.975)  # 69.75
        else:
            self.beep_outlet = beep_cnt.quantile(0.75) + beep_cnt.std() * 1.5  # 98.7

        if self.beep_outlet:
            weekend_to_work = df["is_weekend"].eq(1) & df["beep_count"].gt(self.beep_outlet)
            work_to_weekend = df["is_weekend"].eq(0) & df["beep_count"].lt(self.beep_outlet)
            df.loc[weekend_to_work, 'is_weekend'] = 0
            df.loc[work_to_weekend, 'is_weekend'] = 1

        return df

    def cat_dummies(self, df):
        """
        Отметка категориальных колонок --> str для catboost
        OneHotEncoder для остальных
        :param df: ДФ
        :return: ДФ с фичами
        """
        # если нет цифровых колонок --> заполним их
        if self.category_columns and not self.numeric_columns:
            self.numeric_columns = [col_name for col_name in df.columns
                                    if col_name not in self.category_columns]
        # если нет категориальных колонок --> заполним их
        if self.numeric_columns and not self.category_columns:
            self.category_columns = [col_name for col_name in df.columns
                                     if col_name not in self.numeric_columns]

        for col_name in self.category_columns:
            if col_name in df.columns:
                if self.use_catboost:
                    df[col_name] = df[col_name].astype(str)
                else:
                    print(f'Трансформирую колонку: {col_name}')
                    # Create dummy variables
                    df = pd.get_dummies(df, columns=[col_name], drop_first=self.drop_first)

                    self.new_columns.extend([col for col in df.columns
                                             if col.startswith(col_name)])
        return df

    def apply_scaler(self, df):
        """
        Масштабирование цифровых колонок
        :param df: исходный ДФ
        :return: нормализованный ДФ
        """
        if not self.transform_columns:
            self.transform_columns = self.numeric_columns

        self.transform_columns = [col for col in self.transform_columns if col in df.columns]

        print('self.transform_columns:', self.transform_columns)

        if self.scaler and self.transform_columns:
            print(f'Применяю scaler: {self.scaler.__name__} '
                  f'с аргументами: {self.args_scaler}')
            args = self.args_scaler if self.args_scaler else tuple()
            scaler = self.scaler(*args)
            scaled_data = scaler.fit_transform(df[self.transform_columns])
            if scaled_data.shape[1] != len(self.transform_columns):
                print(f'scaler породил: {scaled_data.shape[1]} колонок')
                new_columns = [f'pnf_{n:02}' for n in range(scaled_data.shape[1])]
                df = pd.concat([df, pd.DataFrame(scaled_data, columns=new_columns)], axis=1)
                self.exclude_columns.extend(self.transform_columns)
            else:
                df[self.transform_columns] = scaled_data

            self.comment.update(scaler=self.scaler.__name__, args_scaler=self.args_scaler)
        return df

    def fit_gate_times(self, df, remake_gates_mask=False, use_gates_mask_V2=False):
        """
        Получение паттернов прохода через турникеты
        :param df: тренировочный ДФ
        :param remake_gates_mask: получить шаблоны масок из трейна, иначе взять из класса
        :param use_gates_mask_V2: использовать расширенный набор масок из класса
        :return: ДФ с паттернами
        """
        print('Ищу паттерны в данных...\n')
        current_user_id = prev_time = None
        current_gate_times, current_gates = [], []
        res_gate_times, result_gates = [], []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            if current_user_id != row["user_id"]:
                if len(current_gate_times) >= 3:
                    res_gate_times.append((current_user_id, current_gate_times))
                    result_gates.append((current_user_id, current_gates))
                current_gate_times, current_gates = [], []
                current_user_id = row["user_id"]
                prev_time = row['ts']
            delta = int((row['ts'] - prev_time).total_seconds()) if prev_time else 0
            prev_time = row['ts']
            current_gate_times.append((row["gate_id"], delta))
            current_gates.append(row["gate_id"])

        if len(current_gate_times) >= 3:
            res_gate_times.append((current_user_id, current_gate_times))
            result_gates.append((current_user_id, current_gates))

        gates_times = [tuple(zip(*gt)) for gt in
                       [*map(lambda x: tuple(x[1]), res_gate_times)]]

        if remake_gates_mask:
            res_gate = []
            for user_gates in result_gates:
                gates = user_gates[1]
                start_range = 3 if len(gates) < 5 else 4
                for len_mask in range(start_range, 7):
                    res_gate.extend([*zip(*[gates[i:] for i in range(len_mask)])])
            res_cnt = Counter(res_gate)
            prev_key = prev_cnt = None
            find_gates_mask = []
            for key, cnt in sorted(res_cnt.items()):
                # количество проходов по шаблону 1 или 2 - игнорируем,
                # количество проходов по шаблону 3 и 4 берем только шаблоны длиной 3 и 4,
                # количество проходов по шаблону = 5 берем длины 3, 4 и 5,
                # количество проходов по шаблону = 6 и более берем длины 4, 5 и 6
                if (cnt in (3, 4) and len(key) in (3, 4)) or (cnt == 5 and len(key) < 6) or (
                        cnt > 5 and len(key) < cnt):
                    # убираем дубли когда след шаблон отличается на последним турникетом
                    # и количество шаблонов различается на 2 и менее
                    if len(key) > 4 and prev_key == key[:-1] and abs(prev_cnt - cnt) < 3:
                        prev_key, prev_cnt = key, cnt
                        continue
                    prev_key, prev_cnt = key, cnt
                    find_gates_mask.append(key)
            # заменим ручной отбор шаблонов на автоматический
            self.gates_mask = find_gates_mask
            # print(*self.gates_mask, sep='\n')
        if use_gates_mask_V2:
            # заменим ручной отбор шаблонов на ручной отбор V2
            self.gates_mask = self.gates_M_V2
            # print(*self.gates_mask, sep='\n')
        print('Количество паттернов:', len(self.gates_mask))

        df_gt = pd.DataFrame(columns=['mask'] + [f'dt_{i}' for i in range(6)])
        for gates, times in tqdm(gates_times):
            for mask in self.gates_mask:
                for idx, sub_gates in enumerate(zip(*[gates[i:] for i in range(len(mask))])):
                    if sub_gates == mask:
                        row_df_gt = [mask] + [*times[idx:idx + len(mask)]] + [0] * 3
                        df_gt.loc[len(df_gt)] = row_df_gt[:7]
        df_gt['dt_0'] = 0
        return df_gt

    def group_gate_times(self, df_gt, replace_gates_mask=False):
        """
        Получение временных интервалов для паттернов
        :param df_gt: ДФ с паттернами
        :param replace_gates_mask: заменить атрибут self.gates_mask
        :return: список паттернов с временными интервалами
        """

        # диапазон границ интервалов расширим вниз на 50% и вверх 20% - это сработало лучше,
        # чем расширение границ вниз и вверх на 5%
        def make_min_max(col):
            min_col = min(col)
            min_col = 0 if min_col < 10 else int(min_col * 0.5)
            return min_col, int(max(col) * 1.2)

        grp = df_gt.groupby('mask', as_index=False).agg(
            min_max_0=('dt_0', lambda x: make_min_max(x)),
            min_max_1=('dt_1', lambda x: make_min_max(x)),
            min_max_2=('dt_2', lambda x: make_min_max(x)),
            min_max_3=('dt_3', lambda x: make_min_max(x)),
            min_max_4=('dt_4', lambda x: make_min_max(x)),
            min_max_5=('dt_5', lambda x: make_min_max(x)),
        )
        result = []
        grp_columns = grp.columns.to_list()
        for _, row in grp.iterrows():
            mask = row['mask']
            result.append((mask, tuple(row[col] for col in grp_columns[1:len(mask) + 1])))

        if replace_gates_mask:
            self.gates_mask = result

        return result

    @staticmethod
    def find_gates(row, mask, times=None):
        """
        Поиск паттернов по шаблонам
        :param row: строка датафрейма
        :param mask: шаблон
        :param times: временные границы прохода через турникеты шаблона
        :return: True / False --> найден паттерн по шаблону или нет
        """
        shift_gates = [f'g{i}' for i in range(len(mask) - 1, -len(mask), -1)]
        gates = row[shift_gates].values
        gates_times = None
        if times:
            shift_times = [f't{i}' for i in range(len(mask) - 1, -len(mask), -1)]
            gates_times = row[shift_times].values
        index_mask = -1
        for idx, sub_gates in enumerate(zip(*[gates[i:] for i in range(len(mask))])):
            # найден паттерн турникетов
            if sub_gates == mask:
                # если есть проверка по дельта времени прохода --> смотрим,
                # чтобы дельта попадала в границы диапазона обученных паттернов
                if times is None or (
                        times and all(times[i][0] <= gates_times[idx + i] <= times[i][1]
                                      for i in range(1, len(mask)))):
                    # print('паттерн найден')
                    index_mask = idx
                    break
                print('паттерн не найден, user_id:', row['user_id'], 'idx =', idx)
                print('mask, gates:', mask, gates)
                print('times, gates_times:', times, gates_times)
        return index_mask >= 0

    def fit(self, df, file_df=None, out_five_percent=False, remake_gates_mask=False,
            drop_december=False):
        """
        Формирование фич --> очень временнозатратная операция ~2 часа
        :param df: исходный ФД
        :param file_df: Предобработанный Файл .pkl с полным путём
        :param out_five_percent: граница 5% при определении выбросов
        :param remake_gates_mask: получить шаблоны масок из трейна, иначе взять из класса
        :param drop_december: удалить тех, кто в декабре не появлялся
        :return: обработанный ДФ
        """
        if file_df and file_df.suffix == '.pkl' and file_df.is_file():
            df = pd.read_pickle(file_df)
            return df

        df = self.initial_preparation(df, out_five_percent=out_five_percent)

        # # эту секцию перенес в self.initial_preparation
        # # данные для устранения выбросов, где рабочий день помечен как выходной и наоборот
        # tmp = df[['date', 'weekday', 'beep_count', 'is_weekend']].drop_duplicates()
        # tmp["weekend"] = tmp["weekday"].map(lambda x: 1 if x in (5, 6) else 0)
        # beep_cnt = tmp[tmp["weekend"] == 1].beep_count
        # if out_five_percent:
        #     self.beep_outlet = beep_cnt.quantile(0.975)  # 69.75
        # else:
        #     self.beep_outlet = beep_cnt.quantile(0.75) + beep_cnt.std() * 1.5  # 98.7

        # выделил shift по датам, чтобы случайно не зацепить переход между сутками
        result = pd.DataFrame()
        for flt_date in sorted(df["date"].unique()):
            tmp = df[df["date"] == flt_date]
            # формирование колонок с gate_id для 5 предыдущих и следующих строк
            for i in range(5, -6, -1):
                tmp[f'g{i}'] = tmp['gate_id'].shift(i, fill_value=-9)

            # формирование колонок с timestamp для 5 предыдущих и следующих строк
            for i in range(5, -6, -1):
                tmp[f'ts{i}'] = tmp['ts'].shift(i)
            tmp[f'ts6'] = tmp[f'ts5']
            for i in range(5, -6, -1):
                tmp[f't{i}'] = tmp[f'ts{i}'] - tmp[f'ts{i + 1}']
                tmp[f't{i}'] = tmp[f't{i}'].map(lambda x: x.total_seconds())
                tmp[f't{i}'].fillna(0, inplace=True)
                tmp[f't{i}'] = tmp[f't{i}'].astype(int)

            # удалить временные колонки с timestamp.shift()
            tmp.drop([f'ts{i}' for i in range(6, -6, -1)], axis=1, inplace=True)

            if not len(result):
                result = tmp
            else:
                result = pd.concat([result, tmp])

        df = result

        # df.to_csv('df_ts.csv', sep=';')

        if remake_gates_mask:
            # получим паттерны
            tmp_columns = ['user_id', 'ts', 'gate_id']
            train_tmp = df[df.user_id > -1][tmp_columns]
            self.fit_gate_times(train_tmp, remake_gates_mask=True)
            print('Количество паттернов:', len(self.gates_mask))

        start_time = print_msg('Поиск по шаблонам...')

        tqdm.pandas()
        for mask in self.gates_mask:
            times = None
            if len(mask) == 2:
                mask, times = mask
            mask_col = 'G' + '_'.join(map(str, mask))
            print(f'Шаблон: {mask} колонка: {mask_col}')
            df[mask_col] = df.progress_apply(
                lambda row: self.find_gates(row, mask, times=times), axis=1).astype(int)

        if self.preprocess_path_file:
            df.to_pickle(self.preprocess_path_file)
            df.to_csv(self.preprocess_path_file.with_suffix('.csv'))

        print_time(start_time)
        return df

    def transform(self, df, model_columns=None, out_five_percent=False, drop_december=False):
        """
        Формирование остальных фич
        :param df: ДФ
        :param model_columns: список колонок, которые будут использованы в модели
        :param out_five_percent: граница 5% при определении выбросов
        :param drop_december: удалить тех, кто в декабре не появлялся
        :return: ДФ с фичами
        """
        df = self.initial_preparation(df, out_five_percent=out_five_percent)

        # заполнение колонки user_word на трейне значением user_id
        idx_isna_words = df['user_word'].isna()
        df.loc[idx_isna_words, 'user_word'] = df.loc[idx_isna_words, 'user_id']
        # количество проходов 'user_word', т.е. user_id за день + среднее медиана по ним
        grp = df.groupby(['user_word', 'date'], as_index=False).agg(
            list_gates_full=('gate_id', tuple),
        )
        grp['user_day_beep'] = grp['list_gates_full'].map(len)
        grp['user_day_mean'] = grp.groupby('user_word').user_day_beep.transform('mean')
        grp['user_day_median'] = grp.groupby('user_word').user_day_beep.transform('median')
        grp['counter'] = grp['list_gates_full'].map(Counter)

        df = df.merge(grp, on=['user_word', 'date'], how='left')
        # print(grp.columns)
        # df_to_excel(grp, file_dir.joinpath('user_day_beep.xlsx'), float_cells=[3, 4])

        # выделение временных лагов между проходами через gate_id
        lags = {'lag0': lambda x: not x, 'lag1': lambda x: x == 1, 'lag2': lambda x: x == 2,
                'lag3': lambda x: x <= 3,
                'lag4': lambda x: 2 < x <= 5,
                'lag5': lambda x: 5 < x <= 15,
                'lag6': lambda x: 15 < x <= 25,
                'lag7': lambda x: 25 < x <= 36,
                'lag8': lambda x: 36 < x <= 49,
                'lag9': lambda x: 49 < x <= 79,
                }

        for col_name, lag_func in lags.items():
            df[col_name] = (df['t0'].map(lag_func) | df['t-1'].map(lag_func)).astype(int)

        # выделение временных лагов между одинаковыми gate_id
        for col_name, lag_func in lags.items():
            # (g1 g0 g-1) & (t0 t-1)
            gate_prev = (df['g1'] == df['g0']) & df['t0'].map(lag_func)
            gate_next = (df['g0'] == df['g-1']) & df['t-1'].map(lag_func)
            df[col_name.replace('lag', 'dbl')] = (gate_prev | gate_next).astype(int)

        # группировки для подсчета кол-ва --------------------------------
        grp_month = df.groupby(['month'], as_index=False).agg(
            counts=('ts', 'count'),
            user_id_unique=('user_id', lambda x: x.nunique()),
            date_unique=('date', lambda x: x.nunique())
        )
        grp_month['prs'] = grp_month['counts'] / grp_month['counts'].sum()

        grp_week = df.groupby(['week'], as_index=False).agg(
            counts=('ts', 'count'),
            user_id_unique=('user_id', lambda x: x.nunique())
        )
        grp_week['prs'] = grp_week['counts'] / grp_week['counts'].sum()

        grp_date = df.groupby(['date'], as_index=False).agg(
            date_cnt=('ts', 'count')
        )
        grp_gate = df.groupby(['date', 'gate_id'], as_index=False).agg(
            gate_cnt=('ts', 'count'),
        )
        # группировки -------------------------------------------------

        if model_columns is None:
            model_columns = df.columns.to_list()

        if "user_id" not in model_columns:
            model_columns.insert(0, "user_id")

        # добавление колонок с количеством турникетов за день по user_id
        if 'counter' not in self.exclude_columns and 'counter' in df.columns:
            unique_gates = sorted(df['gate_id'].unique())
            for gate in unique_gates:
                mask_col = f'gate_{gate}'
                df[mask_col] = df['counter'].map(lambda x: x.get(gate, 0))
                self.numeric_columns.append(mask_col)
                model_columns.append(mask_col)
            self.exclude_columns.append('counter')

        self.train_idxs = df[df.month.isin(self.train_months)].index
        self.valid_idxs = df[df.month.isin(self.valid_months)].index

        df = self.cat_dummies(df)

        df = self.apply_scaler(df)

        model_columns.extend(self.new_columns)

        exclude_columns = [col for col in self.exclude_columns if col in df.columns]
        exclude_columns.extend(col for col in df.columns if col not in model_columns)

        if exclude_columns:
            df.drop(exclude_columns, axis=1, inplace=True)

        self.exclude_columns = exclude_columns

        # Переводим типы данных в минимально допустимые - экономим ресурсы
        df = memory_compression(df, exclude_columns=['counter'])

        return df

    def fit_transform(self, df, file_df=None, out_five_percent=False,
                      remake_gates_mask=False, model_columns=None, drop_december=False):
        """
        fit + transform
        :param df: исходный ФД
        :param file_df: Предобработанный Файл .pkl с полным путём
        :param out_five_percent: граница 5% при определении выбросов
        :param remake_gates_mask: получить шаблоны масок из трейна, иначе взять из класса
        :param model_columns: список колонок, которые будут использованы в модели
        :param drop_december: удалить тех, кто в декабре не появлялся
        :return: ДФ с фичами
        """
        df = self.fit(df, file_df=file_df,
                      out_five_percent=out_five_percent,
                      remake_gates_mask=remake_gates_mask,
                      drop_december=drop_december)
        df = self.transform(df, model_columns=model_columns,
                            out_five_percent=out_five_percent,
                            drop_december=drop_december)
        return df

    def train_test_split(self, df, y=None, *args, **kwargs):
        """
        Деление на обучающую и валидационную выборки
        :param df: ДФ
        :param y: целевая переменная
        :param args: аргументы
        :param kwargs: именованные аргументы
        :return: x_train, x_valid, y_train, y_valid
        """
        if any(key in kwargs for key in ('train_size', 'test_size')):
            if 'test_size' in kwargs:
                train_size = 1 - kwargs['test_size']
            else:
                train_size = kwargs['train_size']
            if train_size > 0.99:
                train_size = 0.99
            train_rows = int(len(df) * train_size)
            x_train = df.iloc[:train_rows]
            x_valid = df.iloc[train_rows:]
            if y is None:
                y_train = y_valid = None
            else:
                y_train = y.iloc[:train_rows]
                y_valid = y.iloc[train_rows:]
        else:

            x_train = df.loc[self.train_idxs]
            x_valid = df.loc[self.valid_idxs]
            if y is None:
                y_train = y_valid = None
            else:
                y_train = y.loc[self.train_idxs]
                y_valid = y.loc[self.valid_idxs]

            self.comment.update(train_months=self.train_months,
                                valid_months=self.valid_months)

        return x_train, x_valid, y_train, y_valid

    @staticmethod
    def make_sample(df, days=5):
        """
        Для опытов оставим небольшой сэмпл из данных и виде первых дней days
        :param df: ДФ
        :param days: количество дней для обучения и +1 день для теста, чтобы код не падал
        :return: ДФ сэмпла данных
        """
        dates = sorted(df['date'].unique())[:days + 1]
        temp = df[df['date'].isin(dates)]
        temp.loc[temp['date'] == dates[-1], 'user_id'] = -1
        return temp

    def train_valid_split(self, df, test_size=0.2, SEED=17, drop_outlets=False):
        """
        Деление на обучающую и валидационную выборки штатным train_test_split
        :param df: ДФ
        :param test_size: test_size
        :param SEED: SEED
        :param drop_outlets: удалить редких юзеров и гейты
        :return: x_train, x_valid, y_train, y_valid
        """
        train = df.drop(['user_id'], axis=1)
        target = df['user_id']

        # Split the train_df into training and testing sets
        X_train, X_valid, y_train, y_valid = train_test_split(train, target,
                                                              test_size=test_size,
                                                              stratify=target,
                                                              random_state=SEED)
        if drop_outlets:
            if 'gate_id' in df.columns:
                outlet_user_gate = df.user_id.isin([4, 51]) | df.gate_id.isin([0, 16])
                self.comment.update(drop_users='4,51', drop_gates='0,16')
            else:
                outlet_user_gate = df.user_id.isin([4, 51])
                self.comment.update(drop_users='4,51')
            outlet_index = df[outlet_user_gate].index
            X_train = X_train[~X_train.index.isin(outlet_index)]
            X_valid = X_valid[~X_valid.index.isin(outlet_index)]
            y_train = y_train[~y_train.index.isin(outlet_index)]
            y_valid = y_valid[~y_valid.index.isin(outlet_index)]

        return X_train, X_valid, y_train, y_valid


class DataTransform2(DataTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gates_mask = self.gates_M_V2 = []
        self.gate_value_counts = None
        self.drop_outlet_users = None
        self.drop_outlet_gates = None
        self.make_gate_pattern = True
        self.make_gate_counter = True
        self.out_user_id = None
        self.file_dir = Path(kwargs.get('file_dir', Path(__file__).parent))

    @staticmethod
    def almost_equal(mask1, mask2, compare_first=2, last_different=True,
                     user_find=False, sorted_set=False):
        """
        Проверка похожести двух списков турникетов:
        в двух масках может отличаться один элемент, если они разной длины,
        или элементы, стоящие на одинаковой позиции, если маски одной длины.
        :param mask1: список 1 - список турникетов
        :param mask2: список 2 - маска турникетов
        :param compare_first: необходимо сравнивать первые N элементов
        :param last_different: сравниваем списки без последнего элемента
        :param user_find: поиск для user_id - критерии сравнения разниц увеличиваются на 1
        :param sorted_set: будем сравнивать сортированные множества
        :return: True / False
        """
        if compare_first and mask1[:compare_first] != mask2[:compare_first]:
            return False
        len_delta = 0
        if user_find:
            len_gates, len_mask = len(mask1), len(mask2)
            # Длина списка турникетов не больше, чем на 2 элемента маски
            # или не меньше, чем на 1 элемент маски
            # для обоих Списка и Маски <= 8 элементов.
            # Для большей длины элементов разница в длинах не учитывается.
            if (len_gates - len_mask > 2 or len_mask - len_gates > 1) and all(
                    x <= 8 for x in (len_gates, len_mask)):
                return False
            if sorted_set:
                mask1, mask2 = sorted(set(mask1)), sorted(set(mask2))
                len_gates, len_mask = len(mask1), len(mask2)
            len_delta = max(len_gates - len_mask, len_mask - len_gates)

        if len(mask1) > len(mask2):
            mask1, mask2 = mask2, mask1
        len1, len2 = len(mask1), len(mask2)

        # тут проверяем 2 маски на отличие последнего элемента
        if last_different:
            if len2 - len1 == 1 and mask1 == mask2[:-1]:
                return True
            return False

        if not user_find and abs(len2 - len1) > 1:
            return False

        i = j = true_elements = 0
        differences = []
        while i < len1 and j < len2:
            if mask1[i] == mask2[j]:
                true_elements += 1
            else:
                if len1 == len2 and not user_find:
                    differences.extend([mask1[i], mask2[j]])

                elif (i + 1 < len1 and mask1[i + 1] == mask2[j]) or (
                        i + 2 < len1 and mask1[i + 2] == mask2[j]):
                    j -= 1
                    differences.append(mask1[i])
                elif (j + 1 < len2 and mask1[i] == mask2[j + 1]) or (
                        j + 2 < len2 and mask1[i] == mask2[j + 2]):
                    i -= 1
                    differences.append(mask2[j])
                else:
                    differences.extend([mask1[i], mask2[j]])
            i += 1
            j += 1
        differences.extend(mask1[i:])
        differences.extend(mask2[j:])
        # print(differences, len(differences), true_elements)

        if user_find:
            if len(differences) - len_delta > 2 and len(differences) > true_elements / 2:
                return False
        else:
            if (len(differences) > 1 and len2 - len1 == 1) or (
                    len(differences) > 2 and len2 == len1):
                return False
        return True

    def fit_days_mask(self, df, out_five_percent=False, remove_double_gate=False,
                      show_messages=True, drop_december=False):
        """
        Построение маски турникетов для user_id по целому дню
        :param df: Объединенный датафрейм трейн и тест
        :param out_five_percent: граница 5% при определении выбросов
        :param remove_double_gate: удалить повторяющиеся подряд турникеты
        :param show_messages: Выводить сообщения о ходе процесса
        :param drop_december: удалить тех, кто в декабре не появлялся
        :return: Сгруппированный ДФ с маской турникетов по дням
        """

        def remove_doubles(list_gates):
            res = [list_gates[i] for i in range(len(list_gates))
                   if i == 0 or list_gates[i] != list_gates[i - 1]]
            return tuple(res)

        df = self.initial_preparation(df.copy(), out_five_percent=out_five_percent)

        # присвоение отсутствующим user_id из теста закодированных слов
        df.loc[df['user_id'] < 0, 'user_id'] = df.loc[df['user_id'] < 0, 'user_word']

        df.user_word.fillna('', inplace=True)  # заполнение пропусков user_word на трейне

        group_period = 'seconds'
        # group_period = 'minutes'
        # список турникетов за день для user_id
        grp = df.groupby(['user_id', 'date'], as_index=False).agg(
            time_start=(group_period, min),
            time_end=(group_period, max),
            time_delta=(group_period, lambda x: x.max() - x.min()),
            list_gates_full=('gate_id', tuple),
        )
        # всего посещений по user_id
        grp['total_visits'] = grp.groupby('user_id').list_gates_full.transform('count')

        # количество турникетов за день для одного user_id + среднее медиана по ним
        grp['len_gates_full'] = grp.list_gates_full.map(len)
        grp['len_gates_full_mean'] = grp.groupby('user_id') \
            .len_gates_full.transform('mean')
        grp['len_gates_full_median'] = grp.groupby('user_id') \
            .len_gates_full.transform('median')

        gates_col = 'list_gates_full'
        # удаляем повторяющиеся подряд турникеты
        double_gate = ''
        if remove_double_gate:
            gates_col = 'list_gates'
            double_gate = '_remove_double_gate'
            grp[gates_col] = grp.list_gates_full.map(remove_doubles)
            # количество турникетов за день для одного user_id без повторяющихся турникетов
            name_col = 'len_gates'
            grp[name_col] = grp[gates_col].map(len)
            grp['len_gates_mean'] = grp.groupby('user_id')[name_col].transform('mean')
            grp['len_gates_median'] = grp.groupby('user_id')[name_col].transform('median')

        grp['train'] = grp.user_id.map(lambda x: int(str(x).isnumeric()))
        # когда первый и последний раз проходил user_id
        grp['first_show'] = grp.groupby('user_id')['date'].transform("min")
        grp['last_show'] = grp.groupby('user_id')['date'].transform("max")
        grp['no_december'] = grp['last_show'].map(
            lambda x: pd.to_datetime(x).month not in (1, 2, 12)).astype(int)

        if drop_december:
            # Попробовать убрать те user_id, которых не было в декабре
            out_user_id = grp[grp.train.eq(1) & grp['no_december'].eq(1)].user_id.unique()
            grp = grp[~grp.user_id.isin(out_user_id)]
            self.comment.update(drop_december=str(drop_december))
            self.out_user_id = sorted(out_user_id)

        # подсчет количества раз использования последовательности турникетов
        gate_value_counts = grp[gates_col].value_counts().to_frame()
        gate_value_counts.columns = ['cnt_use_gates']
        gate_value_counts.insert(0, gates_col, gate_value_counts.index)
        grp = grp.merge(gate_value_counts, on=gates_col, how='left')
        # отношение: сколько раз встречалась эта последовательность на кол-во визитов user_id
        grp['ratio_gate'] = grp['cnt_use_gates'] / grp['total_visits']

        gate_value_counts.sort_values(gates_col, inplace=True)
        gate_value_counts['len_gates'] = gate_value_counts[gates_col].map(len)
        # df_to_excel(gate_value_counts, self.file_dir.joinpath(f'gate_val_cnt{double_gate}.xlsx'))

        tmp = grp.value_counts(subset=[gates_col, 'train']).to_frame()
        tmp.columns = ['cnt_use_gates']
        tmp = tmp.reset_index().sort_values([gates_col, 'train'], ascending=[True, False])
        tmp['all_use_gates'] = tmp.groupby(gates_col).cnt_use_gates.transform(sum)
        tmp['len_gates'] = tmp[gates_col].map(len)
        tmp = tmp.merge(grp[[gates_col, 'total_visits']].drop_duplicates(),
                        on=gates_col, how='left')
        # df_to_excel(tmp, self.file_dir.joinpath(f'train_value_counts{double_gate}.xlsx'))

        # маски турникетов, которые встречаются 5 и более раз
        self.gates_mask_count_ge5 = tmp[tmp.all_use_gates.ge(5)][
            gates_col].unique().tolist()

        # маски турникетов, которые встречаются 2-4 раза, user_id приходил менее 9 раз и
        # список турникетов есть в трейне и тесте
        tmp = tmp[tmp.all_use_gates.gt(1) & tmp.all_use_gates.le(4) & tmp.total_visits.le(9)]
        tmp['train_count'] = tmp.groupby(gates_col).train.transform('count')
        self.gates_mask_count_2_4 = tmp[tmp.train_count.eq(2)][gates_col].unique().tolist()

        self.gates_mask = sorted(set(self.gates_mask_count_ge5 + self.gates_mask_count_2_4))
        if show_messages:
            print('Шаблонов gates_mask_count_ge5:', len(self.gates_mask_count_ge5))
            print('Шаблонов gates_mask_count_2_4:', len(self.gates_mask_count_2_4))
            print('Всего шаблонов:', len(self.gates_mask))

        # маски турникетов, которые не удовлетворяют двум вышеуказанным условиям
        gate_value_counts['outlet'] = gate_value_counts[gates_col].map(
            lambda x: x not in self.gates_mask).astype(int)
        # df_to_excel(gate_value_counts, self.file_dir.joinpath(f'gate_val_cnt{double_gate}.xlsx'))

        gvc = gate_value_counts

        gvc['gates'] = gvc[gates_col]

        if gates_col == 'list_gates':
            # обрежем все маски до 12 элементов
            gvc['gates'] = gvc[gates_col].apply(lambda x: tuple(x[:12]))
        else:
            # Условие для фильтрации строк
            condition = gvc['outlet'] == 1
            # Берем первые 12 элементов из колонки gates_col по строкам с условием
            gvc.loc[condition, 'gates'] = gvc.loc[
                condition, 'gates'].apply(lambda x: tuple(x[:12]))
        gvc['len_g12'] = gvc['gates'].map(len)
        gvc['counts'] = gvc.groupby('gates').cnt_use_gates.transform(sum)

        # Условие для фильтрации: маски длиной от 9 турникетов, встречающиеся 5<= X <= 50 раз
        condition = gvc['len_g12'].ge(9) & gvc['counts'].ge(5) & gvc['counts'].le(50)
        mask_ge10 = sorted(gvc[condition].gates.unique(), key=len, reverse=True)

        mapping = pd.DataFrame(columns=['mask', 'len_mask', 'gates', 'use_gates'])
        for mask in mask_ge10:
            condition = gvc.gates.apply(lambda x: str(x).startswith(str(mask)[:-1]))
            tmp = gvc[condition & ~gvc.gates.isin(mask_ge10)]
            if len(tmp):
                mapping.loc[len(mapping)] = [mask, len(mask),
                                             sorted(tmp.gates.unique()),
                                             tmp['cnt_use_gates'].sum()]
        mapping = mapping.explode('gates')
        mapping.insert(3, 'len_gates', mapping['gates'].map(len))
        # mapping['len_gates'] = mapping['gates'].map(len)
        df_to_excel(mapping, self.file_dir.joinpath(f'mapping.xlsx'))

        mapping_dict = mapping.drop_duplicates('gates', keep='last').set_index('gates')[
            'mask'].to_dict()

        # замена исходных турникетов, на новые маски
        gvc['gates'] = gvc['gates'].map(lambda x: mapping_dict.get(x, x))
        gvc['counts'] = gvc.groupby('gates').cnt_use_gates.transform(sum)
        gvc['outlet'] = gvc.apply(
            lambda row: int(row.counts < 5 and row.gates not in self.gates_mask), axis=1)

        # ищем маски, которые отличаются друг от друга на 1 последний турникет
        if show_messages:
            print(f'Старый self.gates_mask: {len(self.gates_mask)} элементов')
        # Условие для фильтрации: маски встречающиеся 5<= X <= 50 раз
        condition = gvc['counts'].ge(5) & gvc['counts'].le(50)
        self.gates_mask = gvc[condition]['gates'].unique().tolist()
        self.gates_mask = sorted(set(self.gates_mask + self.gates_mask_count_2_4))
        if show_messages:
            print(f'Новый self.gates_mask:  {len(self.gates_mask)} элементов')
        masks = sorted(gvc.gates.unique())
        # print(masks)
        mapping_dict2 = {}
        for idx, mask in enumerate(masks[:-1]):
            for next_mask in masks[idx + 1:]:
                if mask == next_mask or len(mask) < 3 or len(next_mask) < 3:
                    continue
                if self.almost_equal(mask, next_mask, last_different=True):
                    # если next_mask из списка, а mask нет -> пропускаем
                    if next_mask in self.gates_mask and mask not in self.gates_mask:
                        continue
                    # если mask есть в ключах (заменяемых масках) - это не имеет смысла
                    if mask in mapping_dict2.keys() and mask not in self.gates_mask:
                        continue
                    mapping_dict2[next_mask] = mask
                    # print(mask, masks[idx + 1], sep='\n')
        # замена исходных турникетов, на новые маски
        gvc['gates'] = gvc['gates'].map(lambda x: mapping_dict2.get(x, x))
        gvc['counts'] = gvc.groupby('gates').cnt_use_gates.transform(sum)
        gvc['outlet'] = gvc.apply(
            lambda row: int(row.counts < 5 and row.gates not in self.gates_mask), axis=1)

        gvc['len_g12'] = gvc['gates'].map(len)

        # ищем маски, которые отличаются друг от друга на 1 турникет:
        # при равной длине, 1 отличие на одинаковых позициях,
        # при длине mask2 (х) больше на 1 чем mask1, один элемент в mask2 на любой позиции
        # almost_equal_masks_dict = {}
        # gvc_gts = gvc[['gates']]
        # for mask in masks[:3]:
        #     if len(mask) < 3:
        #         continue
        #     flt_gvc = gvc_gts.gates.map(
        #         lambda x: len(x) > 2 and self.almost_equal(mask, x, last_different=False))
        #     print(gvc_gts[flt_gvc].gates.tolist())

        self.gate_value_counts = gvc

        almost_equal_masks = pd.DataFrame(list(mapping_dict2.items()),
                                          columns=['from_gates', 'gates'])
        almost_equal_masks['len_from_gates'] = almost_equal_masks['from_gates'].map(len)
        almost_equal_masks['len_mask'] = almost_equal_masks['gates'].map(len)
        almost_equal_masks = almost_equal_masks.merge(
            gvc[['gates', 'counts']].drop_duplicates(), on='gates', how='left')

        df_to_excel(almost_equal_masks.drop_duplicates().sort_values('gates'),
                    self.file_dir.joinpath('almost_equal_masks.xlsx'))

        # df_to_excel(gvc, self.file_dir.joinpath(f'gate_val_cnt{double_gate}.xlsx'))

        # return grp
        # exit()

        # маски турникетов, которые встречаются 5 и более раз
        gvc['counts'] = gvc.groupby('gates').cnt_use_gates.transform(sum)
        gvc['outlet'] = gvc.apply(
            lambda row: int(row.counts < 5 and row.gates not in self.gates_mask), axis=1)

        self.gates_mask = gvc[gvc.counts.ge(5)]['gates'].unique().tolist()
        self.gates_mask = sorted(set(self.gates_mask + self.gates_mask_count_2_4))
        if show_messages:
            print(f'Новый self.gates_mask:  {len(self.gates_mask)} элементов')

        df_to_excel(gvc, self.file_dir.joinpath(f'gate_value_counts{double_gate}.xlsx'))

        tmp = df[['user_id', 'date', 'month', '1day', '2day', 'last_day-1', 'last_day',
                  'weekday', 'is_weekend', 'DofY1', 'DofY2', 'user_word']].drop_duplicates()
        grp = grp.merge(tmp, on=['user_id', 'date'], how='left')

        # Количество различных вариантов турникетов для user_id
        grp['user_diff_gates'] = grp.groupby('user_id')[gates_col] \
            .transform(lambda x: len(set(x)))
        # Добавить группировки для user_id по таким колонкам
        add_group_columns = ['1day', '2day', 'last_day-1', 'last_day', 'weekday',
                             'is_weekend', 'DofY1', 'DofY2']
        for grp_col in add_group_columns:
            # Количество различных вариантов турникетов по дням недели
            grp[f'{grp_col}_diff_gates'] = grp.groupby(['user_id', grp_col])[gates_col] \
                .transform(lambda x: len(set(x)))

        return grp

    @staticmethod
    def get_mapping_dict(df, len_mask, min_elements=5, one_parent=True):
        """
        Создание словаря для маппинга масок заданной подпоследовательности
        :param df: ДФ
        :param len_mask: длина маски
        :param min_elements: мин.кол-во элементов, найденных по данной маске
        :param one_parent: брать маски только от одного родителя
                           (только для условия, что родитель сам маска)
        :return: словарь и ДФ
        """
        cdf = df.copy()
        seq_name = f'sequence{len_mask}'
        # Создаем новый столбец с подпоследовательностью заданной длины
        cdf[seq_name] = cdf['gates'].map(lambda x: x[:len_mask])
        # Фильтруем строки, удовлетворяющие условиям
        flt = cdf[['gates', seq_name]].groupby(seq_name).filter(
            lambda x: len(x) >= min_elements)
        flt = flt[flt[seq_name].map(len) == len_mask].drop_duplicates()
        # Удаляем временную колонку seq_name, объединяем ДФ с новыми масками из seq_name
        cdf = pd.merge(cdf.drop(columns=[seq_name]), flt, on='gates', how='left')
        # оставим только те маски у которых только одна маска-предок
        tmp = cdf[cdf.outlet == 0].groupby(seq_name).agg(
            parent=('gates', lambda x: sorted(x.unique())))
        tmp['parent'] = tmp['parent'].map(lambda x: x[1:])
        # список gates, которые нужно убрать из замены
        out_gates = sum(tmp['parent'].tolist(), [])
        if not one_parent:
            out_gates = []
        # Фильтруем по маскам, которые нужно исключить
        flt = cdf[~cdf.gates.isin(out_gates)][['gates', seq_name]].dropna()
        gates_mapping_dict = flt.drop_duplicates('gates').set_index('gates')[
            seq_name].to_dict()
        return gates_mapping_dict, cdf

    def update_gates(self, dfg, mapping_dict, len_mask):
        """
        Обновление последовательностей турникетов
        :param dfg: ДФ
        :param mapping_dict: словарь маппинга
        :param len_mask: длина маски
        :return: измененный ДФ
        """
        dfg['gates'] = dfg['gates'].map(lambda x: mapping_dict.get(x, x))
        dfg['len_g12'] = dfg['gates'].map(len)
        dfg['counts'] = dfg.groupby('gates')['gate_cnt'].transform(sum)
        dfg['outlet'] = dfg.apply(
            lambda row: int(row.counts < 5 and row.gates not in self.gates_mask), axis=1)
        self.gates_mask = dfg[dfg.counts.ge(5)]['gates'].unique().tolist()
        self.gates_mask = sorted(set(self.gates_mask + self.gates_mask_count_2_4))
        print(f'Новый self.gates_mask {len_mask:2}:  {len(self.gates_mask)} элементов')
        return dfg

    def find_gates_pattern(self, all_df, debug=False):
        """
        Поиск паттернов по шаблонам для каждого user_id
        :param all_df: датафрейм
        :param debug: выводить на экран отладочную информацию
        :return: датафрейм c заполненными шаблонами
        """

        dfg = all_df.groupby(['user_id', 'list_gates'], as_index=False).agg(
            gate_cnt=('date', 'count'))
        dfg['gate_cnt_all'] = dfg.groupby('list_gates').gate_cnt.transform(sum)
        dfg['len_gates'] = dfg.list_gates.map(len)

        gvc_columns = ['list_gates', 'outlet', 'gates', 'len_g12', 'counts']
        dfg = dfg.merge(self.gate_value_counts[gvc_columns], on='list_gates', how='left')

        dfg.sort_values(['user_id', 'gates', 'list_gates'], inplace=True)
        df_to_excel(dfg, self.file_dir.joinpath('user_gates.xlsx'))

        dfg = dfg.sort_values(['gates', 'list_gates', 'user_id']).reset_index(drop=True)
        df_to_excel(dfg, self.file_dir.joinpath('gate_users.xlsx'))

        mapping_dict, _ = self.get_mapping_dict(dfg, 11, one_parent=False)
        dfg = self.update_gates(dfg, mapping_dict, 11)

        # # замена исходных турникетов, на новые маски
        for len_mask in range(10, 5, -2):
            mapping_dict, _ = self.get_mapping_dict(dfg, len_mask, one_parent=True)
            dfg = self.update_gates(dfg, mapping_dict, len_mask)
        #
        # ОПЫТЫ с этой частью кода ведутся в tst_mask.py !!!
        #
        # ДФ с заполненной инфой по user_id
        users = pd.DataFrame()
        users_idx = dfg.user_id.unique()
        for user_id in tqdm(users_idx, total=len(users_idx)):
            if debug:
                print(f'user_id: {user_id}')

            df = dfg[dfg.user_id.eq(user_id)]
            # print(cdf)

            first_gates = df.gates.map(lambda x: x[0]).value_counts()
            common_first_gate = first_gates.index[0]
            first_gates = first_gates.to_dict()
            # если первый турникет не типичный для этого user_id встречаемость < 5%
            # и длина > 1 -> добавим в начало его самый любимый номер турникета
            df.gates = df.gates.apply(
                lambda x: (common_first_gate,) + x
                if first_gates.get(x[0], 0) / len(df) * 100 <= 5 and len(x) > 1 else x)

            df['found_gates'] = ''
            for index, row in df.iterrows():
                for mask in self.gates_mask:
                    if row.gates == mask:
                        df.at[index, 'found_gates'] = mask
                        break

            df['replace'] = ''

            no_found = df[df['found_gates'].map(len) < 1]

            user_masks = sorted(df[df['found_gates'].map(len) > 0].found_gates.unique(),
                                key=lambda x: (len(x), x), reverse=True)
            # print(*user_masks, sep='\n')

            # no_found_gates = no_found.gates.tolist()
            # print(no_found_gates)

            for index, row in no_found.iterrows():
                for mask in filter(lambda x: x[:2] == row.gates[:2], user_masks):
                    if self.almost_equal(row.gates, mask, last_different=False,
                                         user_find=True):
                        df.at[index, 'found_gates'] = mask
                        df.at[index, 'replace'] = mask
                        break
                    if self.almost_equal(row.gates, mask, last_different=False,
                                         user_find=True,
                                         sorted_set=True):
                        df.at[index, 'found_gates'] = mask
                        df.at[index, 'replace'] = mask
                        break
                    if str(row.gates).startswith((str(mask)[:-1], str(mask[:-1])[:-1])):
                        df.at[index, 'found_gates'] = mask
                        break

            # Если не смогли подобрать последовательность турникетов - тогда заполним самым
            # популярным для этого user_id, который начинается с такого же турникета
            df_copy = df.copy()
            no_found = df[df['found_gates'].map(len) < 1]
            for index, row in no_found.iterrows():
                gate1 = str(tuple(row.gates[:1]))[:-1]
                if debug:
                    print('gate1:', gate1)
                df_copy['start'] = df_copy.found_gates.map(
                    lambda x: str(x).startswith(gate1))
                tmp = df_copy[df_copy['start']]
                # print(tmp)
                if len(tmp):
                    # print('tmp.gate_cnt.argmax()', tmp.gate_cnt.argmax())
                    most_common = tmp.iloc[tmp.gate_cnt.argmax()].found_gates
                    if debug:
                        print('most_common', most_common)
                    df.at[index, 'found_gates'] = most_common
                    df.at[index, 'replace'] = most_common

            # Если не смогли подобрать последовательность турникетов - тогда заполним самой
            # популярной последовательностью
            no_found = df[df['found_gates'].map(len) < 1]
            if len(no_found):
                most_common = df.iloc[df.gate_cnt.argmax()].found_gates
                if debug:
                    print('Наиболее часто используемая последовательность', most_common)
                for index, row in no_found.iterrows():
                    df.at[index, 'found_gates'] = most_common
                    df.at[index, 'replace'] = most_common

            # Если не смогли подобрать последовательность турникетов - тогда заполним самой
            # популярной последовательностью из всего, которая начинается с такого турникета
            df_copy = dfg[dfg.outlet == 0].copy()
            no_found = df[df['found_gates'].map(len) < 1]
            for index, row in no_found.iterrows():
                gate1 = str(tuple(row.gates[:2]))[:-1]
                if debug:
                    print('gate1:', gate1)
                df_copy['start'] = df_copy['gates'].map(lambda x: str(x).startswith(gate1))
                tmp = df_copy[df_copy['start'] & (df_copy['len_g12'] <= len(row.gates) + 1)]
                # print(tmp)
                if len(tmp):
                    # print('tmp.gate_cnt.argmax()', tmp.gate_cnt.argmax())
                    most_common = tmp.iloc[tmp.gate_cnt.argmax()]['gates']
                    if debug:
                        print('most_common', most_common)
                    df.at[index, 'found_gates'] = most_common
                    df.at[index, 'replace'] = most_common

            # Если не смогли подобрать последовательность турникетов - тогда заполним самым
            # популярным для этого user_id
            no_found = df[df['found_gates'].map(len) < 1]
            most_common = df[df['found_gates'].map(len) > 0]['found_gates'].value_counts()
            if len(set(most_common.tolist())) > 1:
                most_common = most_common.index[0]
            else:
                most_common = df.loc[df['counts'].idxmax(), 'found_gates']
            if debug:
                print('most_common', most_common)
            for index, row in no_found.iterrows():
                df.at[index, 'found_gates'] = most_common
                df.at[index, 'replace'] = most_common

            users = pd.concat([users, df])

        cwd = [(0, 14), (1, 52)] + [(2, 10)] * 4 + [(6, 40)] + [(7, 10)] * 2 + [(9, 40)] * 2
        df_to_excel(users, self.file_dir.joinpath('find_gates_patt.xlsx'), ins_col_width=cwd)

        all_df = all_df.merge(users.drop(columns=['gate_cnt', 'len_gates']),
                              on=['user_id', 'list_gates'], how='left')
        return all_df

    def fit(self, df, file_df=None, out_five_percent=False, remake_gates_mask=True,
            drop_december=False):
        """
        Формирование фич
        :param df: исходный ФД
        :param file_df: Предобработанный Файл .pkl с полным путём
        :param out_five_percent: граница 5% при определении выбросов
        :param remake_gates_mask: получить шаблоны масок из трейна, иначе взять из класса
        :param drop_december: удалить тех, кто в декабре не появлялся
        :return: обработанный ДФ
        """
        if file_df and file_df.suffix == '.pkl' and file_df.is_file():
            df = pd.read_pickle(file_df)
            return df

        # удаление user_id с выбросами
        if self.drop_outlet_users is not None and isinstance(self.drop_outlet_users,
                                                             (list, tuple)):
            df = df[~df.user_id.isin(self.drop_outlet_users)]
        # удаление gate_id с выбросами, которых нет в тесте
        if self.drop_outlet_gates is not None and isinstance(self.drop_outlet_gates,
                                                             (list, tuple)):
            df = df[~df.gate_id.isin(self.drop_outlet_gates)]

        # ищем шаблоны последовательности турникетов
        df = self.fit_days_mask(df, out_five_percent=out_five_percent,
                                remove_double_gate=True, drop_december=drop_december)

        self.gates_mask = sorted(set(self.gates_mask_count_2_4 + self.gates_mask_count_ge5))

        start_time = print_msg('Поиск по шаблонам...')
        df = self.find_gates_pattern(df)

        df = df.sort_values(['date', 'time_start', 'user_id']).reset_index(drop=True)

        if self.preprocess_path_file:
            df.to_pickle(self.preprocess_path_file)

            wd = [(0, 12)] * len(df.columns)
            df_to_excel(df, self.preprocess_path_file.with_suffix('.xlsx'), ins_col_width=wd)

        print_time(start_time)
        return df

    def transform(self, df, model_columns=None, out_five_percent=False, mem_compress=True,
                  drop_december=False):
        """
        Формирование остальных фич
        :param df: ДФ
        :param model_columns: список колонок, которые будут использованы в модели
        :param out_five_percent: граница 5% при определении выбросов
        :param mem_compress: Переводить типы данных в минимально допустимые
        :param drop_december: удалить тех, кто в декабре не появлялся
        :return: ДФ с фичами
        """
        if self.drop_outlet_users is not None and isinstance(self.drop_outlet_users,
                                                             (list, tuple)):
            self.comment.update(drop_users=str(self.drop_outlet_users))
        if self.drop_outlet_gates is not None and isinstance(self.drop_outlet_gates,
                                                             (list, tuple)):
            self.comment.update(drop_gates=str(self.drop_outlet_gates))

        self.gates_mask = df[df.outlet.eq(0)]['found_gates'].unique().tolist()

        # сортировка почти как в исходном ДФ
        df = df.sort_values(['date', 'time_start', 'user_id']).reset_index(drop=True)

        if drop_december:
            # Попробовать убрать те user_id, которых не было в декабре
            out_user_id = df[df.train.eq(1) & df['no_december'].eq(1)].user_id.unique()
            df = df[~df.user_id.isin(out_user_id)]
            self.comment.update(drop_december=str(drop_december))

        if self.make_gate_pattern:
            # Проверить гипотезу со сравнением маски с последовательностью турникетов
            # не на равенство, а по str(x).startswith(str(mask)[:-1])
            for mask in tqdm(sorted(self.gates_mask)):
                mask_col = 'G' + '_'.join(map(str, mask))
                # df[mask_col] = df.found_gates.map(lambda x: int(x == mask))
                df[mask_col] = df.found_gates.map(
                    lambda x: int(str(x).startswith(str(mask)[:-1])))

        if self.make_gate_counter:
            # Исходный список
            # df['counter'] = df.list_gates_full.map(Counter)
            # Список турникетов за день для одного user_id без повторяющихся турникетов
            df['counter'] = df['list_gates'].map(Counter)
            unique_gates = df['list_gates'].unique().tolist()
            unique_gates = sorted(set(sum(map(list, unique_gates), [])))
            for gate in unique_gates:
                mask_col = f'gate_{gate}'
                df[mask_col] = df['counter'].map(lambda x: x.get(gate, 0))
                self.numeric_columns.append(mask_col)

        if model_columns is None:
            model_columns = df.columns.to_list()

        if "user_id" not in model_columns:
            model_columns.insert(0, "user_id")

        self.train_idxs = df[df.month.isin(self.train_months)].index
        self.valid_idxs = df[df.month.isin(self.valid_months)].index

        df = self.cat_dummies(df)

        df = self.apply_scaler(df)

        model_columns.extend(self.new_columns)

        exclude_columns = [col for col in self.exclude_columns if col in df.columns]
        exclude_columns.extend(col for col in df.columns if col not in model_columns)

        if exclude_columns:
            df.drop(exclude_columns, axis=1, inplace=True)

        self.exclude_columns = exclude_columns

        if 'train' in df.columns:
            df.loc[df.train < 1, 'user_id'] = -1
            df.user_id = df.user_id.astype(int)

        # Переводим типы данных в минимально допустимые - экономим ресурсы
        df = memory_compression(df)

        file_dir = Path(__file__).parent
        wd = [(0, 12)] * len(df.columns)
        df_to_excel(df, self.file_dir.joinpath('df_to_model.xlsx'), ins_col_width=wd)

        return df


if __name__ == "__main__":
    # тут разные опыты с классом...

    file_dir = Path(__file__).parent

    # Чтение трейна и теста и объединение их в один ДФ
    all_df = read_all_df(file_dir)

    numeric_columns = ['min', 'minutes', 'seconds', 'beep_count', 'beep_gate', 'row_id']
    cat_columns = ['gate_id', 'weekday', 'hour']

    data_cls = DataTransform(category_columns=cat_columns, drop_first=False,
                             # numeric_columns=numeric_columns, scaler=StandardScaler,
                             )
    prefix_preprocess = '_MV2'
    data_cls.preprocess_path_file = GATES_DIR.joinpath(
        f'preprocess_df{prefix_preprocess}.pkl')

    data_cls.beep_outlet = 98.7
    all_df = data_cls.fit(all_df, file_df=data_cls.preprocess_path_file,
                          remake_gates_mask=False)

    # Добавление номера строки вместе с scaler=StandardScaler чуть увеличивает скор
    all_df['row_id'] = all_df.index

    all_df = data_cls.transform(all_df)
    all_df.to_csv(GATES_DIR.joinpath(f'all_df{prefix_preprocess}.csv'))

    gate_col = sorted([col for col in all_df.columns if col.startswith('gate_')],
                      key=lambda x: (int(x.split('_')[-1]), x))

    df = all_df[['user_id', 'ts', 'date'] + gate_col]
    wd = [(0, 16)] * len(df.columns)
    df_to_excel(df, GATES_DIR.joinpath(f'all_df_gates{prefix_preprocess}.xlsx'),
                ins_col_width=wd)
