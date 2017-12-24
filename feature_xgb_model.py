__author__ = 'Alex Rodin'

import pandas as pd
import numpy as np
import us_holidays
import plt_cr
import xgboost as xgb
import logging
from logging.config import fileConfig
from sklearn.metrics import log_loss, f1_score, classification_report
from datetime import datetime
from datetime import timedelta
from sklearn.externals import joblib

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
fileConfig('xgb_model.log.ini')
logger = logging.getLogger()


def to_dt(dt_str, right=0):
    dt_str = dt_str.strip()[:-6]
    dt = datetime.strptime(dt_str, TIME_FORMAT)
    return np.datetime64((dt + timedelta(days=right)).strftime('%Y-%m-%d'))


def dt_range_unroll(amin, amax, weekmask='1111100', holidays=None):
    if not holidays:
        holidays = []
    dt_min = to_dt(amin)
    dt_max = to_dt(amax, 1)
    return np.busday_count(dt_min, dt_max, weekmask, holidays)


def bd_num(amin, amax):
    return dt_range_unroll(amin, amax)


def hd_num(amin, amax):
    no_hd = dt_range_unroll(amin, amax, weekmask='1111111', holidays=[])
    hd = dt_range_unroll(amin, amax, weekmask='1111111', holidays=us_holidays.US_HOLIDAYS)
    return np.absolute(no_hd - hd)


def wd_num(amin, amax):
    return dt_range_unroll(amin, amax, weekmask='0000011', holidays=[])


def to_sec(dt_str):
    dt_str = dt_str.strip()[:-6]
    dt = datetime.strptime(dt_str, TIME_FORMAT)
    return (dt - datetime(1970, 1, 1)).total_seconds()


def prepare_data(order_lines_src='global_mt_2.csv',
                 unit_costs_src='unit_costs.csv',
                 prd_attr='prods_fashion_fit.csv',
                 sku_attr='skus_fashion_fit.csv',
                 cycle_size=3 * 7 * 24 * 60 * 60,
                 test_set_size=0.2,
                 data_set_dst=None):
    ol = pd.read_csv(order_lines_src)
    ol = ol[ol['status'] == 20]
    ol['promo_ratio'] = np.where(ol.loc[:, 'list_price'] == 0, 1, ol.loc[:, 'promo_price'] / ol.loc[:, 'list_price'])
    ol.fillna(0, inplace=True)
    
    ol.sort_values(['prod_id', 'sku_id', 'order_date', 'promo_ratio'], inplace=True)
    ol['cons'] = (ol.loc[:, 'promo_ratio'].shift(1) != ol.loc[:, 'promo_ratio']).astype(int).cumsum()
    
    del ol['status']
    del ol['tax_flag']
    del ol['promo_id']
    del ol['promo_schema']
    del ol['promo_price']
    
    unit_costs = pd.read_csv(unit_costs_src)
    
    ol = pd.merge(left=ol, right=unit_costs, left_on=['sku_id'], right_on=['ITEM_ID'])
    ol.rename(columns={'UNIT_COST': 'unit_cost'}, inplace=True)
    
    del ol['ITEM_ID']
    
    ##ADDING PRODUCT/SKU FEATURES############
    prd_attr=pd.read_csv(prd_attr)
    sku_attr=pd.read_csv(sku_attr)
    prd_attr=prd_attr   #feel free to choose features u need
    sku_attr=sku_attr  #feel free to choose features u need
    ol1 = pd.merge(left=ol[['sku_id','prod_id','cons']], right=sku_attr, left_on=['sku_id','prod_id'], right_on=['s_code','s_prnt_prd'], how='left')
    del ol1['s_code']
    del ol1['sku_id']
    del ol1['s_prnt_prd']
    ol1 = pd.merge(left=ol1, right=prd_attr, left_on=['prod_id'], right_on=['prd_id'], how='left')
    del ol1['prd_id']
    del ol1['prod_id']
    #########################################
    
    ol['price_ratio'] = np.where(ol.loc[:, 'list_price'] == 0, 1, ol.loc[:, 'price'] / ol.loc[:, 'list_price'])
    ol['uc_ratio'] = np.where(ol.loc[:, 'list_price'] == 0, 1, ol.loc[:, 'unit_cost'] / ol.loc[:, 'list_price'])
    ol_demand = ol.groupby(['cons'], as_index=False).agg({'ordered_qty': np.sum})
    ol = pd.merge(left=ol, right=ol_demand, left_on=['cons'], right_on=['cons'])
    ol['eff'] = ol.loc[:, 'ordered_qty_x'] * ol.loc[:, 'price_ratio'] / ol.loc[:, 'ordered_qty_y']
    
    del ol['ordered_qty_x']
    del ol['price']
    del ol['list_price']
    del ol['unit_cost']
    del ol['ordered_qty_y']
    
    ol_grp = ol.groupby(['cons'], as_index=False).agg(
                                                      {'price_ratio': ['first', 'last', np.min, np.max], 'promo_ratio': np.mean, 'uc_ratio': np.mean, 'eff': np.sum,
                                                      'order_date': [np.min, np.max]})
    ol=ol_grp
    ol['uc_ratio_mean'] = ol.loc[:, 'uc_ratio']['mean']
                                                      
    ol['promo_ratio_mean'] = ol.loc[:, 'promo_ratio']['mean']
    ol['price_ratio_first'] = ol.loc[:, 'price_ratio']['first']
    ol['price_ratio_last'] = ol.loc[:, 'price_ratio']['last']
    ol['price_ratio_min'] = ol.loc[:, 'price_ratio']['amin']
    ol['price_ratio_max'] = ol.loc[:, 'price_ratio']['amax']
    ol['eff_sum'] = ol.loc[:, 'eff']['sum']
    ol['order_date_min'] = ol.loc[:, 'order_date']['amin']
    ol['order_date_max'] = ol.loc[:, 'order_date']['amax']

    del ol['uc_ratio']
    del ol['promo_ratio']
    del ol['price_ratio']
    del ol['eff']
    del ol['order_date']

    ol['bd'] = np.vectorize(bd_num)(ol.loc[:, 'order_date_min'], ol.loc[:, 'order_date_max'])
    ol['wd'] = np.vectorize(wd_num)(ol.loc[:, 'order_date_min'], ol.loc[:, 'order_date_max'])
    ol['hd'] = np.vectorize(hd_num)(ol.loc[:, 'order_date_min'], ol.loc[:, 'order_date_max'])
    ol['start'] = ol.loc[:, 'order_date_min'].map(to_sec)
    ol['label'] = np.where(ol.loc[:, 'eff_sum'] > ol.loc[:, 'uc_ratio_mean'], 1, 0)
        
    del ol['eff_sum']
    del ol['order_date_min']
    del ol['order_date_max']

    ol['start_cycle'] = ol.loc[:, 'start'] % cycle_size
                                                      
    del ol['start']
                                                      
    ol['promo_ratio_mean'] = np.round_(ol.loc[:, 'promo_ratio_mean'], decimals=3)
    ol['uc_ratio_mean'] = np.round_(ol.loc[:, 'uc_ratio_mean'], decimals=3)
    ol = ol.sort_values(['uc_ratio_mean', 'promo_ratio_mean', 'label', 'start_cycle'])
    ol_grp = ol.groupby(['uc_ratio_mean', 'promo_ratio_mean'], as_index=False)
                                                      
    ol['num'] = ol_grp.cumcount() + 1
    ol_grp_max = ol_grp.agg({'num': 'max'})
    ol_grp_max.rename(columns={'num': 'num_max'}, inplace=True)
    ol = pd.merge(left=ol, right=ol_grp_max, left_on=['uc_ratio_mean', 'promo_ratio_mean'], right_on=['uc_ratio_mean', 'promo_ratio_mean'])
                                                      
    ol.rename(columns={
              ('price_ratio_first', '') : 'price_ratio_first',
              ('price_ratio_last', '') : 'price_ratio_last',
              ('price_ratio_min', '') : 'price_ratio_min',
              ('price_ratio_max', '') : 'price_ratio_max',
              ('bd', '') : 'bd',
              ('wd', '') : 'wd',
              ('hd', '') : 'hd',
              ('label', '') : 'label',
              ('start_cycle', '') : 'start_cycle',
              ('num', '') : 'num',
              ('cons', '') : 'cons',
                        }, inplace=True)
                        
    #MERGING WITH SKU ATTRIBUTES##########################################
    ol = pd.merge(left=ol, right=ol1, left_on=['cons'], right_on=['cons'])
    ######################################################################
    del ol['cons']
                                                      
    del ol['hd']
                                                      
    ol['data_set_marker'] = np.where(ol.loc[:, 'num_max'] - ol.loc[:, 'num'] <= ol.loc[:, 'num_max'] * test_set_size - 1, 0, 1)
                                                      
    del ol[('promo_ratio_mean', '')]
    del ol[('uc_ratio_mean', '')]
    del ol['num']
    del ol['num_max']
                                                      
    del ol['uc_ratio_mean']
                                                      
    if data_set_dst:
        ol.to_csv(data_set_dst, index=False)
    return ol

def shape_data(inp, inp_file=None):
    logger.info("Shaping of data-set ...")
    if inp is None and inp_file is None:
        logger.error("No data-set for XGBoost model")
        exit(1)

    X = pd.read_csv(inp_file) if inp is None else inp
    y = np.array(X['label'].tolist())

    good = y.sum()
    bad = len(y) - good

    trn_X = X[X['data_set_marker'] == 1]
    tst_X = X[X['data_set_marker'] == 0]
    trn_Y = np.array(X[X['data_set_marker'] == 1]['label'].tolist())
    tst_Y = np.array(X[X['data_set_marker'] == 0]['label'].tolist())

    del trn_X['label']
    del tst_X['label']
    del trn_X['data_set_marker']
    del tst_X['data_set_marker']

    bln_Y = np.zeros(len(tst_Y))
    bln_Y[:] = 1.0 * good / (bad + good)

    logger.info("Data-set has been shaped.")
    return trn_X, trn_Y, tst_X, tst_Y, bln_Y


def train(trn_X, trn_Y, model_file_dst=None):
    logger.info('XGB Model training ...')
    model = xgb.XGBClassifier(max_depth=4,
                              n_estimators=400,
                              learning_rate=0.08,
                              objective="binary:logistic")
    model.fit(trn_X, trn_Y)
    if model_file_dst:
        joblib.dump(model, model_file_dst)
        logger.info('Trained XGB Model has been saved to {}'.format(model_file_dst))
    logger.info('XGB Model has been trained.')
    return model


def test(trn_X, tst_X, tst_Y, bln_Y, model, saved_model_file=None):
    logger.info('XGB Model testing ...')
    if model is None:
        if saved_model_file is None:
            logger.error('XGB Model file was not passed')
            exit(1)
        model = joblib.load(saved_model_file)
    logger.info("Base line: {}".format(log_loss(tst_Y, bln_Y)))
    p_prb = model.predict_proba(tst_X)
    logger.info("Test predict log loss: {}".format(log_loss(tst_Y, p_prb)))
    p = model.predict(tst_X)
    logger.info("Test predict f1-score: {}".format(f1_score(tst_Y, p)))
    plt_cr.plot_classification_report(classification_report(tst_Y, p), 'Effectiveness', 'Score Report for Submission')

    sp = list(zip(model.feature_importances_, list(trn_X)))
    sp = sorted(sp, key=lambda x: -x[0])
    plt_cr.plot_features(sp, 'Effectiveness', 'Features for Effectiveness')
    logger.info('XGB Model has been tested.')

ds = prepare_data(
     order_lines_src='./global_mt_2.csv',
     unit_costs_src='./unit_costs.csv',
     data_set_dst='xgb_ds_01.csv'
 )

trn_X, trn_Y, tst_X, tst_Y, bln_Y = shape_data(None, inp_file='xgb_ds_01.csv')
train(trn_X, trn_Y, model_file_dst='xgb_trained_01.pkl')
test(trn_X, tst_X, tst_Y, bln_Y, None, 'xgb_trained_01.pkl')






