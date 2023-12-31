import pytest

from deepctr.models import PNN
from ..utils import check_model, get_test_data, SAMPLE_SIZE, get_test_data_estimator, check_estimator, TEST_Estimator


@pytest.mark.parametrize(
    'use_inner, use_outter,sparse_feature_num',
    [(True, True, 3), (False, False, 1)
     ]
)
def test_PNN(use_inner, use_outter, sparse_feature_num):
    model_name = "PNN"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)
    model = PNN(feature_columns, dnn_hidden_units=[4, 4], dnn_dropout=0.5, use_inner=use_inner, use_outter=use_outter)
    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'use_inner, use_outter,sparse_feature_num',
    [(True, True, 2)
     ]
)
def test_PNNEstimator(use_inner, use_outter, sparse_feature_num):
    if not TEST_Estimator:
        return
    from deepctr.estimator import PNNEstimator

    sample_size = SAMPLE_SIZE
    _, dnn_feature_columns, input_fn = get_test_data_estimator(sample_size,
                                                               sparse_feature_num=sparse_feature_num,
                                                               dense_feature_num=sparse_feature_num)

    model = PNNEstimator(dnn_feature_columns, dnn_hidden_units=[4, 4], dnn_dropout=0.5, use_inner=use_inner,
                         use_outter=use_outter)

    check_estimator(model, input_fn)


if __name__ == "__main__":
    pass
