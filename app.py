from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

app = FastAPI(title="Iris Creative AI Classifier")

# Load model SVM
try:
    model = joblib.load("svm_model.pkl")
except Exception:
    model = None

SPECIES_MAP = {
    0: {"name": "Iris-setosa", "icon": "🪻", "color": "#8B5CF6", "desc": "Cánh hoa ngắn, đài hoa rộng"},
    1: {"name": "Iris-versicolor", "icon": "🌺", "color": "#EC4899", "desc": "Kích thước trung bình, màu sắc sặc sỡ"},
    2: {"name": "Iris-virginica", "icon": "🪷", "color": "#3B82F6", "desc": "Cánh hoa dài & rộng, kích thước lớn nhất"}
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
    <html lang="vi" class="light">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris AI Studio - Phân Loại Hoa Thông Minh</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
        <script>
            tailwind.config = { darkMode: 'class' }
        </script>
    </head>
    <body class="bg-slate-50 dark:bg-slate-900 text-slate-800 dark:text-slate-100 min-h-screen transition-colors duration-300 p-4 md:p-8">
        <div class="max-w-4xl mx-auto space-y-6">
            
            <div class="flex justify-between items-center bg-white dark:bg-slate-800 p-4 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700">
                <div class="flex items-center gap-3">
                    <span class="text-3xl">🌸</span>
                    <div>
                        <h1 class="text-xl font-bold bg-gradient-to-r from-indigo-500 to-pink-500 bg-clip-text text-transparent">Iris AI Classifier Studio</h1>
                        <p class="text-xs text-slate-400">Dự đoán loài hoa sinh động & trực quan hóa dữ liệu</p>
                    </div>
                </div>
                <button onclick="toggleDarkMode()" class="p-2 rounded-xl bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 transition">
                    <span id="themeIcon">🌙</span>
                </button>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-12 gap-6">
                
                <div class="md:col-span-7 bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-md border border-slate-200 dark:border-slate-700 space-y-5">
                    
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">Mẫu hoa điển hình</label>
                        <div class="grid grid-cols-3 gap-2">
                            <button onclick="applyPreset(5.1, 3.5, 1.4, 0.2)" class="py-1.5 px-2 bg-purple-50 dark:bg-purple-950/40 text-purple-600 dark:text-purple-300 text-xs rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/50 font-medium transition">🪻 Setosa</button>
                            <button onclick="applyPreset(6.0, 2.7, 5.1, 1.6)" class="py-1.5 px-2 bg-pink-50 dark:bg-pink-950/40 text-pink-600 dark:text-pink-300 text-xs rounded-lg hover:bg-pink-100 dark:hover:bg-pink-900/50 font-medium transition">🌺 Versicolor</button>
                            <button onclick="applyPreset(6.5, 3.0, 5.5, 2.0)" class="py-1.5 px-2 bg-blue-50 dark:bg-blue-950/40 text-blue-600 dark:text-blue-300 text-xs rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/50 font-medium transition">🪷 Virginica</button>
                        </div>
                    </div>

                    <form id="irisForm" class="space-y-4">
                        <div class="space-y-3">
                            <div>
                                <div class="flex justify-between text-xs font-semibold mb-1">
                                    <span>Chiều dài đài hoa (Sepal Length)</span>
                                    <span id="sl_val" class="text-indigo-500 font-bold">5.1 cm</span>
                                </div>
                                <input type="range" min="4.0" max="8.0" step="0.1" id="sepal_length" value="5.1" oninput="updateUI()" class="w-full accent-indigo-600">
                            </div>

                            <div>
                                <div class="flex justify-between text-xs font-semibold mb-1">
                                    <span>Chiều rộng đài hoa (Sepal Width)</span>
                                    <span id="sw_val" class="text-indigo-500 font-bold">3.5 cm</span>
                                </div>
                                <input type="range" min="2.0" max="4.5" step="0.1" id="sepal_width" value="3.5" oninput="updateUI()" class="w-full accent-indigo-600">
                            </div>

                            <div>
                                <div class="flex justify-between text-xs font-semibold mb-1">
                                    <span>Chiều dài cánh hoa (Petal Length)</span>
                                    <span id="pl_val" class="text-indigo-500 font-bold">1.4 cm</span>
                                </div>
                                <input type="range" min="1.0" max="7.0" step="0.1" id="petal_length" value="1.4" oninput="updateUI()" class="w-full accent-indigo-600">
                            </div>

                            <div>
                                <div class="flex justify-between text-xs font-semibold mb-1">
                                    <span>Chiều rộng cánh hoa (Petal Width)</span>
                                    <span id="pw_val" class="text-indigo-500 font-bold">0.2 cm</span>
                                </div>
                                <input type="range" min="0.1" max="2.5" step="0.1" id="petal_width" value="0.2" oninput="updateUI()" class="w-full accent-indigo-600">
                            </div>
                        </div>

                        <div id="warningBox" class="hidden p-2.5 bg-amber-50 dark:bg-amber-950/30 border border-amber-300 dark:border-amber-800 rounded-lg text-amber-700 dark:text-amber-400 text-xs flex items-center gap-2">
                            <span>⚠️</span> <span id="warningText">Tỷ lệ cánh hoa/đài hoa hơi bất thường.</span>
                        </div>

                        <button type="submit" class="w-full py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-bold rounded-xl shadow-lg transition duration-200 active:scale-98">
                            ⚡ Phân Tích & Dự Đoán
                        </button>
                    </form>
                </div>

                <div class="md:col-span-5 space-y-6">
                    
                    <div class="bg-white dark:bg-slate-800 p-4 rounded-2xl shadow-md border border-slate-200 dark:border-slate-700 text-center">
                        <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">Mô hình hình học hoa</span>
                        <div class="h-36 bg-slate-50 dark:bg-slate-900 rounded-xl flex items-center justify-center relative overflow-hidden border border-dashed border-slate-300 dark:border-slate-700">
                            <svg id="flowerSvg" class="transition-all duration-300" width="100" height="100" viewBox="-50 -50 100 100">
                                <ellipse id="svgSepal" cx="0" cy="0" rx="20" ry="40" fill="#818CF8" opacity="0.5"/>
                                <ellipse id="svgSepal2" cx="0" cy="0" rx="40" ry="20" fill="#818CF8" opacity="0.5"/>
                                <circle id="svgPetal" cx="0" cy="0" r="15" fill="#F472B6" opacity="0.8"/>
                            </svg>
                        </div>
                    </div>

                    <div id="resultCard" class="bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-md border border-slate-200 dark:border-slate-700 text-center transition-all duration-300">
                        <div id="resultIcon" class="text-6xl mb-2 animate-bounce">❓</div>
                        <h3 id="resultName" class="text-xl font-extrabold">Đang chờ dự đoán...</h3>
                        <p id="resultDesc" class="text-xs text-slate-400 mt-1 mb-4">Hãy chọn thông số và bấm Dự Đoán</p>

                        <div id="probBars" class="space-y-2 text-left hidden pt-2 border-t border-slate-100 dark:border-slate-700">
                            <span class="text-[10px] font-bold text-slate-400 uppercase">Độ tin cậy mô hình</span>
                            <div>
                                <div class="flex justify-between text-xs mb-0.5"><span>Setosa</span><span id="prob0">0%</span></div>
                                <div class="w-full bg-slate-100 dark:bg-slate-700 h-1.5 rounded-full overflow-hidden">
                                    <div id="bar0" class="bg-purple-500 h-full w-0 transition-all duration-500"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex justify-between text-xs mb-0.5"><span>Versicolor</span><span id="prob1">0%</span></div>
                                <div class="w-full bg-slate-100 dark:bg-slate-700 h-1.5 rounded-full overflow-hidden">
                                    <div id="bar1" class="bg-pink-500 h-full w-0 transition-all duration-500"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex justify-between text-xs mb-0.5"><span>Virginica</span><span id="prob2">0%</span></div>
                                <div class="w-full bg-slate-100 dark:bg-slate-700 h-1.5 rounded-full overflow-hidden">
                                    <div id="bar2" class="bg-blue-500 h-full w-0 transition-all duration-500"></div>
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
            </div>

            <div class="bg-white dark:bg-slate-800 p-5 rounded-2xl shadow-md border border-slate-200 dark:border-slate-700">
                <h3 class="text-sm font-bold mb-3 flex items-center gap-2">
                    <span>📜</span> Lịch sử dự đoán gần đây
                </h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs text-slate-500 dark:text-slate-400">
                        <thead class="bg-slate-50 dark:bg-slate-900 text-slate-700 dark:text-slate-300 font-semibold uppercase">
                            <tr>
                                <th class="p-2">Loài hoa</th>
                                <th class="p-2">Đài (D x R)</th>
                                <th class="p-2">Cánh (D x R)</th>
                                <th class="p-2">Mã Lớp</th>
                            </tr>
                        </thead>
                        <tbody id="historyTable" class="divide-y divide-slate-100 dark:divide-slate-700">
                            <tr><td colspan="4" class="p-3 text-center text-slate-400">Chưa có lịch sử dự đoán</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

        </div>

        <script>
            let historyData = [];

            function toggleDarkMode() {
                const html = document.documentElement;
                if (html.classList.contains('dark')) {
                    html.classList.remove('dark');
                    document.getElementById('themeIcon').textContent = '🌙';
                } else {
                    html.classList.add('dark');
                    document.getElementById('themeIcon').textContent = '☀️';
                }
            }

            function applyPreset(sl, sw, pl, pw) {
                document.getElementById('sepal_length').value = sl;
                document.getElementById('sepal_width').value = sw;
                document.getElementById('petal_length').value = pl;
                document.getElementById('petal_width').value = pw;
                updateUI();
            }

            function updateUI() {
                const sl = parseFloat(document.getElementById('sepal_length').value);
                const sw = parseFloat(document.getElementById('sepal_width').value);
                const pl = parseFloat(document.getElementById('petal_length').value);
                const pw = parseFloat(document.getElementById('petal_width').value);

                document.getElementById('sl_val').textContent = sl + ' cm';
                document.getElementById('sw_val').textContent = sw + ' cm';
                document.getElementById('pl_val').textContent = pl + ' cm';
                document.getElementById('pw_val').textContent = pw + ' cm';

                // Cập nhật kích thước hình học SVG
                document.getElementById('svgSepal').setAttribute('ry', sl * 6);
                document.getElementById('svgSepal').setAttribute('rx', sw * 6);
                document.getElementById('svgSepal2').setAttribute('rx', sl * 6);
                document.getElementById('svgSepal2').setAttribute('ry', sw * 6);
                document.getElementById('svgPetal').setAttribute('r', (pl + pw) * 4);

                // Kiểm tra cảnh báo bất thường
                const warnBox = document.getElementById('warningBox');
                if (pw > sw) {
                    warnBox.classList.remove('hidden');
                    document.getElementById('warningText').textContent = 'Chú ý: Chiều rộng cánh hoa lớn hơn chiều rộng đài hoa!';
                } else {
                    warnBox.classList.add('hidden');
                }
            }

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

                // Cập nhật giao diện kết quả
                document.getElementById('resultIcon').textContent = data.icon;
                document.getElementById('resultName').textContent = data.prediction;
                document.getElementById('resultName').style.color = data.color;
                document.getElementById('resultDesc').textContent = data.desc;

                // Cập nhật thanh xác suất
                document.getElementById('probBars').classList.remove('hidden');
                data.probabilities.forEach((p, idx) => {
                    const pct = (p * 100).toFixed(1) + '%';
                    document.getElementById(`prob${idx}`).textContent = pct;
                    document.getElementById(`bar${idx}`).style.width = pct;
                });

                // Sáng tạo 7: Bắn pháo hoa Confetti ăn mừng kết quả
                confetti({ particleCount: 50, spread: 60, origin: { y: 0.7 } });

                // Sáng tạo 8: Ghi lại lịch sử dự đoán
                historyData.unshift({
                    name: data.prediction,
                    icon: data.icon,
                    sepal: `${payload.sepal_length} x ${payload.sepal_width}`,
                    petal: `${payload.petal_length} x ${payload.petal_width}`,
                    id: data.class_id
                });
                if (historyData.length > 5) historyData.pop();
                renderHistory();
            });

            function renderHistory() {
                const tbody = document.getElementById('historyTable');
                tbody.innerHTML = historyData.map(item => `
                    <tr class="hover:bg-slate-50 dark:hover:bg-slate-700/50 transition">
                        <td class="p-2 font-bold">${item.icon} ${item.name}</td>
                        <td class="p-2">${item.sepal}</td>
                        <td class="p-2">${item.petal}</td>
                        <td class="p-2"><span class="px-2 py-0.5 bg-slate-200 dark:bg-slate-700 rounded text-[10px]">Lớp ${item.id}</span></td>
                    </tr>
                `).join('');
            }

            updateUI();
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    
    pred_id = int(model.predict(features)[0]) if model else 0
    
    # Tính toán xác suất (Probability / Decision Score)
    try:
        probs = model.predict_proba(features)[0].tolist()
    except Exception:
        probs = [0.0, 0.0, 0.0]
        probs[pred_id] = 1.0

    species_info = SPECIES_MAP.get(pred_id, {"name": "Không xác định", "icon": "❓", "color": "#6B7280", "desc": ""})
    
    return {
        "class_id": pred_id,
        "prediction": species_info["name"],
        "icon": species_info["icon"],
        "color": species_info["color"],
        "desc": species_info["desc"],
        "probabilities": probs
    }
