import os

import numpy as np
import pandas as pd

from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names


def get_xy_fd():
    # 文件名列表
    file_names = [
        'amazon_sales_url.csv',
        'amazon_sales_ad.csv',
        'amazon_sales_all.csv',
        'amazon_sales_detail.csv',
        'amazon_sales_stock.csv'
    ]

    # 获取当前目录下的data文件夹路径
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # 读取所有文件
    data_frames = {}
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path)
        data_frames[file_name] = df

    # 数据预处理和合并（根据实际需求调整）
    # 假设以`amazon_sales_url.csv`为基础进行合并
    url_data = data_frames['amazon_sales_url.csv']
    ad_data = data_frames['amazon_sales_ad.csv']
    all_data = data_frames['amazon_sales_all.csv']
    detail_data = data_frames['amazon_sales_detail.csv']
    stock_data = data_frames['amazon_sales_stock.csv']

    # 合并数据 (这里以asin和date为主键合并, 请根据实际情况调整)
    merged_data = url_data.merge(ad_data, on=['country', 'asin'], how='left', suffixes=('', '_ad')) \
        .merge(all_data, left_on=['insert_date', 'country'], right_on=['date', 'country'], how='left',
               suffixes=('', '_all')) \
        .merge(detail_data, left_on=['insert_date', 'country', 'asin'], right_on=['update_day', 'country', 'asin'],
               how='left', suffixes=('', '_detail')) \
        .merge(stock_data, left_on=['insert_date', 'country', 'asin'], right_on=['update_day', 'country', 'asin'],
               how='left', suffixes=('', '_stock'))

    # 确保数据按需要的类型读取
    merged_data['insert_date'] = pd.to_datetime(merged_data['insert_date'])

    # 将所有数值特征转为浮点型
    num_cols = merged_data.select_dtypes(include=[np.number]).columns
    merged_data[num_cols] = merged_data[num_cols].astype(float)

    # 定义特征列
    feature_columns = [
        SparseFeat('country', vocabulary_size=merged_data['country'].nunique(), embedding_dim=4),
        SparseFeat('asin', vocabulary_size=merged_data['asin'].nunique(), embedding_dim=8),
        DenseFeat('sales_rank', 1),
        DenseFeat('daily_units', 1),
        DenseFeat('price', 1),
        DenseFeat('stars', 1),
        DenseFeat('reviews', 1),
        DenseFeat('rank_small', 1),
        DenseFeat('clicks', 1),
        DenseFeat('cost', 1),
        DenseFeat('impressions', 1),
        DenseFeat('gmv_ad', 1),
        DenseFeat('sales_ad', 1),
        DenseFeat('read', 1),
        DenseFeat('buyer', 1),
        DenseFeat('on_sell', 1),
        DenseFeat('sales_refund', 1),
        DenseFeat('refund_ratio', 1),
        DenseFeat('gmv_total', 1),
        DenseFeat('sales_total', 1),
        DenseFeat('orders_count', 1),
        DenseFeat('units_count', 1),
        DenseFeat('avg_price', 1),
        DenseFeat('gmv', 1),
        DenseFeat('profit', 1),
        DenseFeat('reserved_fc_transfer', 1),
        DenseFeat('inbound_receiving', 1),
        DenseFeat('in_stock', 1)
    ]

    # 提取特征和标签
    feature_dict = {
        'country': merged_data['country'].astype('category').cat.codes.values,
        'asin': merged_data['asin'].astype('category').cat.codes.values,
        'sales_rank': merged_data['sales_rank'].values,
        'daily_units': merged_data['daily_units'].values,
        'price': merged_data['price'].values,
        'stars': merged_data['stars'].values,
        'reviews': merged_data['reviews'].values,
        'rank_small': merged_data['rank_small'].values,
        'clicks': merged_data['clicks'].values,
        'cost': merged_data['cost'].values,
        'impressions': merged_data['impressions'].values,
        'gmv_ad': merged_data['gmv_ad'].values,
        'sales_ad': merged_data['sales_ad'].values,
        'read': merged_data['read'].values,
        'buyer': merged_data['buyer'].values,
        'on_sell': merged_data['on_sell'].values,
        'sales_refund': merged_data['sales_refund'].values,
        'refund_ratio': merged_data['refund_ratio'].values,
        'gmv_total': merged_data['gmv_total'].values,
        'sales_total': merged_data['sales_total'].values,
        'orders_count': merged_data['orders_count'].values,
        'units_count': merged_data['units_count'].values,
        'avg_price': merged_data['avg_price'].values,
        'gmv': merged_data['gmv'].values,
        'profit': merged_data['profit'].values,
        'reserved_fc_transfer': merged_data['reserved_fc_transfer'].values,
        'inbound_receiving': merged_data['inbound_receiving'].values,
        'in_stock': merged_data['in_stock'].values
    }

    # 生成特征字典
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}

    # 生成标签数组，这里假设标签为二分类，并简单地生成示例标签
    y = np.random.randint(0, 2, merged_data.shape[0])

    return x, y, feature_columns, []




if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    # 输出特征和标签以验证
    print("Features: ", x)
    print("Labels: ", y)
    print("Feature Columns: ", feature_columns)
    model = DIN(feature_columns, behavior_feature_list)
    # model = BST(feature_columns, behavior_feature_list,att_head_num=4)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
