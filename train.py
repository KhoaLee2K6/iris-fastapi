from sklearn import datasets
from sklearn.svm import SVC
import joblib

# Load dữ liệu Iris
iris = datasets.load_iris()

X = iris.data
y = iris.target

# Huấn luyện mô hình
model = SVC(kernel="linear")
model.fit(X, y)

# Lưu mô hình
joblib.dump(model, "svm_model.pkl")

print("Model save.pkl")