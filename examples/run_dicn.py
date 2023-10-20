import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names


def get_xy_fd():
    # 读取数据
    data = pd.read_csv('./data/UserBehavior_small.csv', header=None, names=["user", "item_id", "cate_id", "behavior", "timestamp"])

    # 对类别特征进行Label Encoding
    sparse_features = ["user", "item_id", "cate_id", "behavior"]
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 构建特征列
    feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=8) for feat in sparse_features]

    feature_columns += [
        VarLenSparseFeat(
            SparseFeat('hist_item_id', data['item_id'].nunique(), embedding_dim=8, embedding_name='item_id'),
            maxlen=4, length_name="seq_length"),
        VarLenSparseFeat(
            SparseFeat('hist_cate_id', data['cate_id'].nunique(), embedding_dim=4, embedding_name='cate_id'),
            maxlen=4, length_name="seq_length")]

    # 提取用户ID、性别等其他特征
    user = np.array(data['user'])
    item_id = np.array(data['item_id'])
    cate_id = np.array(data['cate_id'])
    behavior = np.array(data['behavior'])
    timestamp = np.array(data['timestamp'])
    seq_length = np.array([1] * len(data))  # 您需要根据实际情况计算序列长度

    feature_dict = {'user': user, 'item_id': item_id, 'cate_id': cate_id, 'behavior': behavior,
                    'hist_item_id': item_id, 'hist_cate_id': cate_id, 'seq_length': seq_length}

    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1] * len(data))  # 你需要根据实际情况提供目标标签

    return x, y, feature_columns, ['item_id', 'cate_id']


if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    model = DIN(feature_columns, behavior_feature_list)
    # model = BST(feature_columns, behavior_feature_list,att_head_num=4)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
