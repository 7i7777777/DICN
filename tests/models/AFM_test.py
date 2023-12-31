import pytest

from deepctr.models import AFM
from ..utils import check_model, check_estimator, get_test_data, get_test_data_estimator, SAMPLE_SIZE, \
    TEST_Estimator


@pytest.mark.parametrize(
    'use_attention,sparse_feature_num,dense_feature_num',
    [(True, 3, 0),
     ]
)
def test_AFM(use_attention, sparse_feature_num, dense_feature_num):
    model_name = "AFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=dense_feature_num)

    model = AFM(feature_columns, feature_columns, use_attention=use_attention, afm_dropout=0.5)

    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'use_attention,sparse_feature_num,dense_feature_num',
    [(True, 3, 0),
     ]
)
def test_AFMEstimator(use_attention, sparse_feature_num, dense_feature_num):
    if not TEST_Estimator:
        return
    from deepctr.estimator import AFMEstimator

    sample_size = SAMPLE_SIZE

    linear_feature_columns, dnn_feature_columns, input_fn = get_test_data_estimator(sample_size,
                                                                                    sparse_feature_num=sparse_feature_num,
                                                                                    dense_feature_num=dense_feature_num)
    model = AFMEstimator(linear_feature_columns, dnn_feature_columns, use_attention=use_attention, afm_dropout=0.5)
    check_estimator(model, input_fn)


if __name__ == "__main__":
    pass
