from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline
import onnx

pipeline = Pipeline([
    ("scaler", scaler),
    ("isoforest", iso)
])

initial_type = [
    ("float_input", FloatTensorType([None, X.shape[1]]))
]
onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_type
)

