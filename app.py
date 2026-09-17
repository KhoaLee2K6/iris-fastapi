from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

app = FastAPI(title="Iris Classifier API")

# Tải model SVM
model = joblib.load("svm_model.pkl")

# Ánh ánh dữ liệu hoa kèm icon và màu sắc giao diện
SPECIES_MAP = {
    0: {"name": "Iris-setosa", "icon": "🪻", "color": "#8B5CF6"},      # Diên vĩ hoa tím nhẹ
    1: {"name": "Iris-versicolor", "icon": "🌺", "color": "#EC4899"},  # Diên vĩ hoa rực rỡ
    2: {"name": "Iris-virginica", "icon": "🪷", "color": "#3B82F6"}   # Diên vĩ hoa lam
}

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nhận Diện Loài Hoa Iris</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 min-h-screen flex items-center justify-center p-4">
        <div class="bg-white/80 backdrop-blur-md rounded-2xl shadow-xl p-8 max-w-md w-full border border-white/50">
            <h1 class="text-2xl font-bold text-center text-indigo-900 mb-1">🌸 Phân Loại Hoa Iris</h1>
            <p class="text-xs text-center text-gray-500 mb-6">Nhập thông số chiều dài & chiều rộng (cm)</p>
            
            <form id="irisForm" class="space-y-4">
                <div class="grid grid-cols-2 gap-3">
                    <div>
                        <label class="block text-xs font-semibold text-gray-600 mb-1">Đài hoa (Dài)</label>
                        <input type="number" step="0.1" id="sepal_length" value="5.1" required class="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-400 outline-none text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-gray-600 mb-1">Đài hoa (Rộng)</label>
                        <input type="number" step="0.1" id="sepal_width" value="3.5" required class="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-400 outline-none text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-gray-600 mb-1">Cánh hoa (Dài)</label>
                        <input type="number" step="0.1" id="petal_length" value="1.4" required class="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-400 outline-none text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-gray-600 mb-1">Cánh hoa (Rộng)</label>
                        <input type="number" step="0.1" id="petal_width" value="0.2" required class="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-400 outline-none text-sm">
                    </div>
                </div>
                <button type="submit" class="w-full py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg shadow-md transition duration-200 active:scale-95">
                    Dự Đoán
                </button>
            </form>

            <div id="result" class="mt-6 hidden p-4 rounded-xl text-center border transition-all duration-300">
                <div id="icon" class="text-5xl mb-2 animate-bounce"></div>
                <div id="speciesName" class="text-lg font-bold"></div>
                <div id="classId" class="text-xs text-gray-500 mt-1"></div>
            </div>
        </div>

        <script>
            document.getElementById('irisForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const payload = {
                    sepal_length: parseFloat(document.getElementById('sepal_length').value),
                    sepal_width: parseFloat(document.getElementById('sepal_width').value),
                    petal_length: parseFloat(document.getElementById('petal_length').value),
                    petal_width: parseFloat(document.getElementById('petal_width').value)
                };

                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                
                const resultDiv = document.getElementById('result');
                resultDiv.classList.remove('hidden');
                document.getElementById('icon').textContent = data.icon;
                document.getElementById('speciesName').textContent = data.prediction;
                document.getElementById('speciesName').style.color = data.color;
                document.getElementById('classId').textContent = `Mã lớp: ${data.class_id}`;
                resultDiv.style.backgroundColor = `${data.color}15`;
                resultDiv.style.borderColor = data.color;
            });
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length, 
        data.sepal_width, 
        data.petal_length, 
        data.petal_width
    ]]
    
    pred_id = int(model.predict(features)[0])
    species_info = SPECIES_MAP.get(pred_id, {"name": "Chưa xác định", "icon": "❓", "color": "#6B7280"})
    
    return {
        "class_id": pred_id,
        "prediction": species_info["name"],
        "icon": species_info["icon"],
        "color": species_info["color"]
    }
