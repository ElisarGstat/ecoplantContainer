import pandas as pd
from os import listdir
from create_data_list import *
import numpy as np
from sklearn.model_selection import train_test_split


def read_data_file():
    path = './data/'
    dfs = []
    filenames = [path + f for f in listdir(path) if f.endswith('.csv')]
    for filename in filenames:
        dfs.append(pd.read_csv(filename, usecols=lambda x: x not in not_read_in_file))
    print('loaded files')

    df = pd.concat(dfs, ignore_index=True)
    df.loc[:, 'SEG'] = 0
    df.query('dd>=20190101 & ind_shutdown_mal==0', inplace=True)

    df.loc[:, 'enabling_hours'] = df['high_part_enabling_hours'] + df['low_part_enabling_hours']
    df.loc[:,'running_hours'] = df['high_part_running_hours'] + df['low_part_running_hours']
    df.loc[:,'load_hours'] = df['high_part_load_hours'] + df['low_part_load_hours']
    df.loc[:,'load_hours'] = df['high_part_load_hours'] + df['low_part_load_hours']
    df.loc[:,'oil_operating_hours_service'] = df['high_part_oil_operating_hours_service'] + df[
        'low_part_oil_operating_hours_service']
    df.loc[:,'oil_filter_operating_hours_service'] = df['high_part_oil_filter_operating_hours_service'] + df[
        'low_part_oil_filter_operating_hours_service']
    df.loc[:,'separator_filter_operating_hours_service'] = df['high_part_separator_filter_operating_hours_service'] + df[
        'low_part_separator_filter_operating_hours_service']
    df.loc[:,'oil_hours_threshold_service'] = df['high_part_oil_hours_threshold_service'] + df[
        'low_part_oil_hours_threshold_service']
    df.loc[:,'oil_filter_hours_threshold_service'] = df['high_part_oil_filter_hours_threshold_service'] + df[
        'low_part_oil_filter_hours_threshold_service']
    df.loc[:,'separator_filter_hours_threshold_service'] = df['high_part_separator_filter_hours_threshold_service'] + df[
        'low_part_separator_filter_hours_threshold_service']

    df.loc[:,'min_enabling_hours_10M'] = df['min_high_part_enabling_hours_10M'] + df['min_low_part_enabling_hours_10M']
    df.loc[:,'min_running_hours_10M'] = df['min_high_part_running_hours_10M'] + df['min_low_part_running_hours_10M']
    df.loc[:,'min_load_hours_10M'] = df['min_high_part_load_hours_10M'] + df['min_low_part_load_hours_10M']
    df.loc[:,'min_load_hours_10M'] = df['min_high_part_load_hours_10M'] + df['min_low_part_load_hours_10M']
    df.loc[:,'min_oil_operating_hours_service_10M'] = df['min_high_part_oil_operating_hours_service_10M'] + df[
        'min_low_part_oil_operating_hours_service_10M']
    df.loc[:,'min_oil_filter_operating_hours_service_10M'] = df['min_high_part_oil_filter_operating_hours_service_10M'] + df[
        'min_low_part_oil_filter_operating_hours_service_10M']
    df.loc[:,'min_separator_filter_operating_hours_service_10M'] = df[
                                                                 'min_high_part_separator_filter_operating_hours_service_10M'] + \
                                                             df[
                                                                 'min_low_part_separator_filter_operating_hours_service_10M']
    df.loc[:,'min_oil_hours_threshold_service_10M'] = df['min_high_part_oil_hours_threshold_service_10M'] + df[
        'min_low_part_oil_hours_threshold_service_10M']
    df.loc[:,'min_oil_filter_hours_threshold_service_10M'] = df['min_high_part_oil_filter_hours_threshold_service_10M'] + df[
        'min_low_part_oil_filter_hours_threshold_service_10M']
    df.loc[:,'min_separator_filter_hours_threshold_service_10M'] = df[
                                                                 'min_high_part_separator_filter_hours_threshold_service_10M'] + \
                                                             df[
                                                                 'min_low_part_separator_filter_hours_threshold_service_10M']

    df.loc[:,'min_enabling_hours_1H'] = df['min_high_part_enabling_hours_1H'] + df['min_low_part_enabling_hours_1H']
    df.loc[:,'min_running_hours_1H'] = df['min_high_part_running_hours_1H'] + df['min_low_part_running_hours_1H']
    df.loc[:,'min_load_hours_1H'] = df['min_high_part_load_hours_1H'] + df['min_low_part_load_hours_1H']
    df.loc[:,'min_load_hours_1H'] = df['min_high_part_load_hours_1H'] + df['min_low_part_load_hours_1H']
    df.loc[:,'min_oil_operating_hours_service_1H'] = df['min_high_part_oil_operating_hours_service_1H'] + df[
        'min_low_part_oil_operating_hours_service_1H']
    df.loc[:,'min_oil_filter_operating_hours_service_1H'] = df['min_high_part_oil_filter_operating_hours_service_1H'] + df[
        'min_low_part_oil_filter_operating_hours_service_1H']
    df.loc[:,'min_separator_filter_operating_hours_service_1H'] = df[
                                                                'min_high_part_separator_filter_operating_hours_service_1H'] + \
                                                            df[
                                                                'min_low_part_separator_filter_operating_hours_service_1H']
    df.loc[:,'min_oil_hours_threshold_service_1H'] = df['min_high_part_oil_hours_threshold_service_1H'] + df[
        'min_low_part_oil_hours_threshold_service_1H']
    df.loc[:,'min_oil_filter_hours_threshold_service_1H'] = df['min_high_part_oil_filter_hours_threshold_service_1H'] + df[
        'min_low_part_oil_filter_hours_threshold_service_1H']
    df.loc[:,'min_separator_filter_hours_threshold_service_1H'] = df[
                                                                'min_high_part_separator_filter_hours_threshold_service_1H'] + \
                                                            df[
                                                                'min_low_part_separator_filter_hours_threshold_service_1H']

    df.drop(drop_list_high_low, axis=1, inplace=True)
    df.fillna(0, inplace=True)
    df.sort_values(['device_id', 'ts'], inplace=True)
    return df

def highest_20_pct(clf, df, y):
    tmp = pd.DataFrame(y)
    tmp['prob'] = clf.predict_proba(df)[:, 1]
    tmp = tmp.loc[(df.tsHour == 0) & (df.tsMinute == 0), :]
    tmp.sort_values('prob', ascending=False, inplace=True)
    return tmp.head(int(0.2*(len(tmp))))['ind_shutdown_mal_3D_next'].sum() / tmp['ind_shutdown_mal_3D_next'].sum()


def asses_raking(clf, y, X, df):
    tmp = pd.DataFrame(y)
    tmp['prob'] = clf.predict_proba(X)[:, 1]
    tmp['pct'] = pd.qcut(tmp['prob'], 10, labels=False, duplicates='drop')
    tmp = tmp.loc[(df.tsHour == 0) & (df.tsMinute == 0), :]
    tmp['rank'] = tmp['prob'].rank(method='first', ascending=1)
    tmp['rank_pct'] = pd.qcut(tmp['rank'], 10, labels=False, duplicates='drop' )

    tmp2 = tmp.groupby('rank_pct', as_index=False).agg(
        {'prob': 'mean', 'ind_shutdown_mal_3D_next': ['count', 'sum']})
    tmp2[('ind_shutdown_mal_3D_next', 'cumsum')] = tmp2[('ind_shutdown_mal_3D_next', 'sum')].cumsum()
    tmp2[('ind_shutdown_mal_3D_next', 'cumsumpct')] = tmp2[('ind_shutdown_mal_3D_next', 'cumsum')] / tmp2[
        ('ind_shutdown_mal_3D_next', 'sum')].sum()
    tmp2.set_index('rank_pct')
    return tmp2


def split_dates(df, test_pct, device_id):
    tmp_df = df.query('device_id == @device_id').sort_values('ts')
    train_size = round(len(tmp_df) * (1 - test_pct))

    X = tmp_df.drop(non_model_drops, axis=1)
    y = tmp_df['ind_shutdown_mal_3D_next']

    dfeco_train, dfeco_test = X.iloc[:train_size, :], X.iloc[train_size:, :]
    y_train, y_test = y[:train_size], y[train_size:]

    try:
        cnt_y_train = tmp_df.iloc[:train_size, :]
        cnt_y_label = cnt_y_train[(cnt_y_train.tsHour == 0) & (cnt_y_train.tsMinute == 0)]['ind_shutdown_mal_3D_next']
        print(
            f' train label mean {np.round(cnt_y_label.mean(), 4)}, label count {cnt_y_label.sum()}, size {cnt_y_label.count()}, max date {cnt_y_train.ts.max()[:10]}')

    except:
        pass

    try:
        cnt_y = tmp_df.iloc[train_size:, :]
        cnt_y_label = cnt_y[(cnt_y.tsHour == 0) & (cnt_y.tsMinute == 0)]['ind_shutdown_mal_3D_next']
        print(
            f' test label mean {np.round(cnt_y_label.mean(), 4)}, label count {cnt_y_label.sum()}, size {cnt_y_label.count()} max date {cnt_y.ts.max()[:10]}')
    except:
        pass
    return dfeco_train, dfeco_test, y_train, y_test


def split_set(df, device_id, test_size):
    tmp_df = df.query('device_id == @device_id').sort_values('ts')

    X = tmp_df.drop(non_model_drops, axis=1)
    y = tmp_df['ind_shutdown_mal_3D_next']

    train, test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return train, test, y_train, y_test


def highest_20_pct(clf, df, y):
    tmp = pd.DataFrame(y)
    tmp['prob'] = clf.predict_proba(df)[:, 1]
    tmp = tmp.loc[(df.tsHour == 0) & (df.tsMinute == 0), :]
    tmp.sort_values('prob', ascending=False, inplace=True)
    return tmp.head(int(0.2*(len(tmp))))['ind_shutdown_mal_3D_next'].sum() / tmp['ind_shutdown_mal_3D_next'].sum()