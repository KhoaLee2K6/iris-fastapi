from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import math

app = FastAPI(title="Iris Creative AI Classifier")

# Load model SVM
try:
    model = joblib.load("svm_model.pkl")
except Exception:
    model = None

SPECIES_MAP = {
    0: {
        "name": "Iris-setosa",
        "icon": "🪻",
        "color": "#8B5CF6",
        "desc": "Cánh hoa ngắn, đài hoa rộng",
        "care_tips": "🌱 **Thổ nhưỡng & Ánh sáng:** Ưa đất khô, tơi xốp và thoát nước tốt, không chịu được ngập úng. Phát triển tốt ở nơi nắng đầy đủ, cần nhiều ánh sáng.."
    },
    1: {
        "name": "Iris-versicolor",
        "icon": "🌺",
        "color": "#EC4899",
        "desc": "Kích thước trung bình, màu sắc sặc sỡ",
        "care_tips": "💧 **Thổ nhưỡng & Ánh sáng:** Ưa đất giàu dinh dưỡng, tơi xốp và thoát nước tốt, độ ẩm vừa phải. Thích ánh sáng nhẹ đến đầy đủ, tránh nắng quá gắt trong thời tiết nóng."
    },
    2: {
        "name": "Iris-virginica",
        "icon": "🪷",
        "color": "#3B82F6",
        "desc": "Cánh hoa dài & rộng, kích thước lớn nhất",
        "care_tips": "☀️ **Thổ nhưỡng & Ánh sáng:** Ưa đất ẩm, nhiều mùn và giàu dinh dưỡng, có khả năng chịu ngập nước. Phát triển tốt ở nơi nhiều ánh sáng và nắng trực tiếp."
    }
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
        <style>
            .glass-card {
                background: rgba(255, 255, 255, 0.45);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(255, 255, 255, 0.6);
            }
            .dark .glass-card {
                background: rgba(30, 41, 59, 0.55);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            .glass-btn {
                background: rgba(255, 255, 255, 0.3);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.4);
            }
            .dark .glass-btn {
                background: rgba(51, 65, 85, 0.4);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
        </style>
    </head>
    <body class="bg-gradient-to-br from-emerald-50 via-teal-50/50 to-indigo-100/60 dark:from-slate-950 dark:via-slate-900 dark:to-emerald-950/40 text-slate-800 dark:text-slate-100 min-h-screen transition-colors duration-300 p-4 md:p-8">
        <div class="max-w-4xl mx-auto space-y-6">
            
            <div class="flex justify-between items-center glass-card p-4 rounded-3xl shadow-lg">
                <div class="flex items-center gap-3">
                    <span class="text-3xl p-2 bg-emerald-100/60 dark:bg-emerald-900/40 rounded-2xl">🌿</span>
                    <div>
                        <h1 class="text-xl font-bold bg-gradient-to-r from-emerald-600 via-teal-600 to-indigo-600 dark:from-emerald-400 dark:to-indigo-300 bg-clip-text text-transparent">Iris Flower</h1>
                        <p class="text-xs text-slate-500 dark:text-slate-400">Phân loại & Phân tích Sinh thái Học</p>
                    </div>
                </div>
                <button onclick="toggleDarkMode()" class="p-2.5 rounded-2xl glass-btn hover:scale-105 transition active:scale-95 shadow-sm">
                    <span id="themeIcon">🌙</span>
                </button>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-12 gap-6">
                
                <div class="md:col-span-7 glass-card p-6 rounded-3xl shadow-xl space-y-5">
                    
                    <div>
                        <div class="flex items-center justify-between mb-2">
                            <label class="block text-xs font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider">⚡ Nút bấm mẫu nhanh (Preset Select)</label>
                            <span class="text-[10px] text-slate-400">Chọn chỉ số sinh học chuẩn</span>
                        </div>
                        <div class="grid grid-cols-3 gap-2">
                            <button onclick="applyPreset(5.1, 3.5, 1.4, 0.2)" class="py-2 px-2 glass-btn hover:bg-purple-100/50 dark:hover:bg-purple-900/40 text-purple-700 dark:text-purple-300 text-xs rounded-xl font-semibold transition shadow-sm hover:shadow active:scale-95">🪻 Setosa</button>
                            <button onclick="applyPreset(5.2, 3.6, 4.2, 1.4)" class="py-2 px-2 glass-btn hover:bg-pink-100/50 dark:hover:bg-pink-900/40 text-pink-700 dark:text-pink-300 text-xs rounded-xl font-semibold transition shadow-sm hover:shadow active:scale-95">🌺 Versicolor</button>
                            <button onclick="applyPreset(6.5, 3.0, 5.5, 2.0)" class="py-2 px-2 glass-btn hover:bg-blue-100/50 dark:hover:bg-blue-900/40 text-blue-700 dark:text-blue-300 text-xs rounded-xl font-semibold transition shadow-sm hover:shadow active:scale-95">🪷 Virginica</button>
                        </div>
                    </div>

                    <form id="irisForm" class="space-y-4">
                        <div class="space-y-3">
                            <div>
                                <div class="flex justify-between text-xs font-semibold mb-1">
                                    <span>Chiều dài đài hoa (Sepal Length)</span>
                                    <span id="sl_val" class="text-emerald-600 dark:text-emerald-400 font-bold">5.1 cm</span>
                                </div>
                                <div class="flex items-center gap-2">
                                    <button type="button" onclick="adjustValue('sepal_length', -0.1)" class="w-8 h-8 glass-btn hover:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 rounded-xl font-extrabold flex items-center justify-center transition active:scale-90 shadow-sm">◀</button>
                                    <input type="range" min="4.0" max="8.0" step="0.1" id="sepal_length" value="5.1" oninput="updateUI()" class="w-full accent-emerald-600">
                                    <button type="button" onclick="adjustValue('sepal_length', 0.1)" class="w-8 h-8 glass-btn hover:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 rounded-xl font-extrabold flex items-center justify-center transition active:scale-90 shadow-sm">▶</button>
                                </div>
                            </div>

                            <div>
                                <div class="flex justify-between text-xs font-semibold mb-1">
                                    <span>Chiều rộng đài hoa (Sepal Width)</span>
                                    <span id="sw_val" class="text-emerald-600 dark:text-emerald-400 font-bold">3.5 cm</span>
                                </div>
                                <div class="flex items-center gap-2">
                                    <button type="button" onclick="adjustValue('sepal_width', -0.1)" class="w-8 h-8 glass-btn hover:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 rounded-xl font-extrabold flex items-center justify-center transition active:scale-90 shadow-sm">◀</button>
                                    <input type="range" min="2.0" max="4.5" step="0.1" id="sepal_width" value="3.5" oninput="updateUI()" class="w-full accent-emerald-600">
                                    <button type="button" onclick="adjustValue('sepal_width', 0.1)" class="w-8 h-8 glass-btn hover:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 rounded-xl font-extrabold flex items-center justify-center transition active:scale-90 shadow-sm">▶</button>
                                </div>
                            </div>

                            <div>
                                <div class="flex justify-between text-xs font-semibold mb-1">
                                    <span>Chiều dài cánh hoa (Petal Length)</span>
                                    <span id="pl_val" class="text-emerald-600 dark:text-emerald-400 font-bold">1.4 cm</span>
                                </div>
                                <div class="flex items-center gap-2">
                                    <button type="button" onclick="adjustValue('petal_length', -0.1)" class="w-8 h-8 glass-btn hover:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 rounded-xl font-extrabold flex items-center justify-center transition active:scale-90 shadow-sm">◀</button>
                                    <input type="range" min="1.0" max="7.0" step="0.1" id="petal_length" value="1.4" oninput="updateUI()" class="w-full accent-emerald-600">
                                    <button type="button" onclick="adjustValue('petal_length', 0.1)" class="w-8 h-8 glass-btn hover:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 rounded-xl font-extrabold flex items-center justify-center transition active:scale-90 shadow-sm">▶</button>
                                </div>
                            </div>

                            <div>
                                <div class="flex justify-between text-xs font-semibold mb-1">
                                    <span>Chiều rộng cánh hoa (Petal Width)</span>
                                    <span id="pw_val" class="text-emerald-600 dark:text-emerald-400 font-bold">0.2 cm</span>
                                </div>
                                <div class="flex items-center gap-2">
                                    <button type="button" onclick="adjustValue('petal_width', -0.1)" class="w-8 h-8 glass-btn hover:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 rounded-xl font-extrabold flex items-center justify-center transition active:scale-90 shadow-sm">◀</button>
                                    <input type="range" min="0.1" max="2.5" step="0.1" id="petal_width" value="0.2" oninput="updateUI()" class="w-full accent-emerald-600">
                                    <button type="button" onclick="adjustValue('petal_width', 0.1)" class="w-8 h-8 glass-btn hover:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 rounded-xl font-extrabold flex items-center justify-center transition active:scale-90 shadow-sm">▶</button>
                                </div>
                            </div>
                        </div>

                        <div id="warningBox" class="hidden p-3 bg-amber-500/10 backdrop-blur-md border border-amber-500/30 rounded-xl text-amber-700 dark:text-amber-300 text-xs flex items-start gap-2 shadow-inner">
                            <span class="text-base">⚠️</span>
                            <div>
                                <span class="font-bold block">Cảnh báo ngoại lệ sinh học (Anomaly Detected):</span>
                                <span id="warningText">Tỷ lệ hình thái hoa bất thường so với thực tế tự nhiên.</span>
                            </div>
                        </div>

                        <button type="submit" class="w-full py-3.5 bg-gradient-to-r from-emerald-600 via-teal-600 to-indigo-600 hover:from-emerald-700 hover:to-indigo-700 text-white font-bold rounded-2xl shadow-lg transition duration-200 active:scale-98">
                            🔍 Phân Tích & Dự Đoán
                        </button>
                    </form>
                </div>

                <div class="md:col-span-5 space-y-6">
                    
                    <div class="glass-card p-4 rounded-3xl shadow-xl text-center">
                        <span class="text-xs font-bold text-slate-400 uppercase tracking-wider block mb-2">Mô hình hình học hoa</span>
                        <div class="h-36 bg-slate-900/5 dark:bg-slate-950/40 rounded-2xl flex items-center justify-center relative overflow-hidden border border-dashed border-emerald-500/30">
                            <svg id="flowerSvg" class="transition-all duration-300" width="100" height="100" viewBox="-50 -50 100 100">
                                <ellipse id="svgSepal" cx="0" cy="0" rx="20" ry="40" fill="#10B981" opacity="0.4"/>
                                <ellipse id="svgSepal2" cx="0" cy="0" rx="40" ry="20" fill="#10B981" opacity="0.4"/>
                                <circle id="svgPetal" cx="0" cy="0" r="15" fill="#EC4899" opacity="0.75"/>
                            </svg>
                        </div>
                    </div>

                    <div id="resultCard" class="glass-card p-6 rounded-3xl shadow-xl text-center transition-all duration-300">
                        <div id="resultIcon" class="text-6xl mb-2 animate-bounce">❓</div>
                        <h3 id="resultName" class="text-xl font-extrabold">Đang chờ dự đoán...</h3>
                        <p id="resultDesc" class="text-xs text-slate-400 mt-1 mb-3">Hãy chọn thông số và bấm Dự Đoán</p>

                        <div id="careBox" class="hidden p-3 mb-4 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl text-left text-xs text-emerald-800 dark:text-emerald-300 space-y-1">
                            <span class="font-bold flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
                                🪴 Mẹo chăm sóc sinh học (Botanic Care Tips):
                            </span>
                            <p id="careTipsText" class="leading-relaxed"></p>
                        </div>

                        <div id="probBars" class="space-y-2 text-left hidden pt-3 border-t border-slate-200/50 dark:border-slate-700/50">
                            <span class="text-[10px] font-bold text-slate-400 uppercase">Độ tin cậy mô hình</span>
                            <div>
                                <div class="flex justify-between text-xs mb-0.5"><span>Setosa</span><span id="prob0">0%</span></div>
                                <div class="w-full bg-slate-200/50 dark:bg-slate-700/50 h-1.5 rounded-full overflow-hidden">
                                    <div id="bar0" class="bg-purple-500 h-full w-0 transition-all duration-500"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex justify-between text-xs mb-0.5"><span>Versicolor</span><span id="prob1">0%</span></div>
                                <div class="w-full bg-slate-200/50 dark:bg-slate-700/50 h-1.5 rounded-full overflow-hidden">
                                    <div id="bar1" class="bg-pink-500 h-full w-0 transition-all duration-500"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex justify-between text-xs mb-0.5"><span>Virginica</span><span id="prob2">0%</span></div>
                                <div class="w-full bg-slate-200/50 dark:bg-slate-700/50 h-1.5 rounded-full overflow-hidden">
                                    <div id="bar2" class="bg-blue-500 h-full w-0 transition-all duration-500"></div>
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
            </div>

            <div class="glass-card p-5 rounded-3xl shadow-xl space-y-3">
                <div class="flex justify-between items-center">
                    <h3 class="text-sm font-bold flex items-center gap-2">
                        <span>📊</span> Khoảng giá trị đặc trưng của các loài Iris (Min - Max cm)
                    </h3>
                    <span class="text-[10px] text-slate-400">Dữ liệu chuẩn Dataset Iris</span>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs text-slate-600 dark:text-slate-300">
                        <thead class="bg-slate-100/50 dark:bg-slate-900/50 text-slate-700 dark:text-slate-300 font-semibold uppercase">
                            <tr>
                                <th class="p-2.5 rounded-l-xl">Loài hoa</th>
                                <th class="p-2.5">Đài dài (Sepal L)</th>
                                <th class="p-2.5">Đài rộng (Sepal W)</th>
                                <th class="p-2.5">Cánh dài (Petal L)</th>
                                <th class="p-2.5 rounded-r-xl">Cánh rộng (Petal W)</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-200/40 dark:divide-slate-700/40 font-medium">
                            <tr class="hover:bg-slate-100/40 dark:hover:bg-slate-700/30 transition">
                                <td class="p-2.5 font-bold text-purple-600 dark:text-purple-400">🪻 Iris-setosa</td>
                                <td class="p-2.5">4.3 – 5.8 cm</td>
                                <td class="p-2.5">2.3 – 4.4 cm</td>
                                <td class="p-2.5">1.0 – 1.9 cm</td>
                                <td class="p-2.5">0.1 – 0.6 cm</td>
                            </tr>
                            <tr class="hover:bg-slate-100/40 dark:hover:bg-slate-700/30 transition">
                                <td class="p-2.5 font-bold text-pink-600 dark:text-pink-400">🌺 Iris-versicolor</td>
                                <td class="p-2.5">4.9 – 7.0 cm</td>
                                <td class="p-2.5">2.0 – 3.4 cm</td>
                                <td class="p-2.5">3.0 – 5.1 cm</td>
                                <td class="p-2.5">1.0 – 1.8 cm</td>
                            </tr>
                            <tr class="hover:bg-slate-100/40 dark:hover:bg-slate-700/30 transition">
                                <td class="p-2.5 font-bold text-blue-600 dark:text-blue-400">🪷 Iris-virginica</td>
                                <td class="p-2.5">4.9 – 7.9 cm</td>
                                <td class="p-2.5">2.2 – 3.8 cm</td>
                                <td class="p-2.5">4.5 – 6.9 cm</td>
                                <td class="p-2.5">1.4 – 2.5 cm</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="glass-card p-5 rounded-3xl shadow-xl">
                <div class="flex justify-between items-center mb-3">
                    <h3 class="text-sm font-bold flex items-center gap-2">
                        <span>📜</span> Lịch sử dự đoán gần đây
                    </h3>
                    <button onclick="clearHistory()" class="px-3 py-1 glass-btn hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 text-xs rounded-xl font-semibold transition shadow-sm hover:shadow active:scale-95 flex items-center gap-1">
                        🗑️ Xóa lịch sử
                    </button>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs text-slate-600 dark:text-slate-300">
                        <thead class="bg-slate-100/50 dark:bg-slate-900/50 text-slate-700 dark:text-slate-300 font-semibold uppercase">
                            <tr>
                                <th class="p-2.5 rounded-l-xl">Loài hoa</th>
                                <th class="p-2.5">Đài (D x R)</th>
                                <th class="p-2.5">Cánh (D x R)</th>
                                <th class="p-2.5 rounded-r-xl">Mã Lớp</th>
                            </tr>
                        </thead>
                        <tbody id="historyTable" class="divide-y divide-slate-200/40 dark:divide-slate-700/40">
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

            function adjustValue(inputId, step) {
                const input = document.getElementById(inputId);
                let val = parseFloat(input.value) + step;
                const min = parseFloat(input.min);
                const max = parseFloat(input.max);
                
                val = Math.min(Math.max(val, min), max);
                input.value = val.toFixed(1);
                updateUI();
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

                document.getElementById('sl_val').textContent = sl.toFixed(1) + ' cm';
                document.getElementById('sw_val').textContent = sw.toFixed(1) + ' cm';
                document.getElementById('pl_val').textContent = pl.toFixed(1) + ' cm';
                document.getElementById('pw_val').textContent = pw.toFixed(1) + ' cm';

                // Cập nhật kích thước SVG
                document.getElementById('svgSepal').setAttribute('ry', sl * 5.5);
                document.getElementById('svgSepal').setAttribute('rx', sw * 5.5);
                document.getElementById('svgSepal2').setAttribute('rx', sl * 5.5);
                document.getElementById('svgSepal2').setAttribute('ry', sw * 5.5);
                document.getElementById('svgPetal').setAttribute('r', (pl + pw) * 3.8);

                // Anomaly Detector
                const warnBox = document.getElementById('warningBox');
                let anomalies = [];

                if (pw > sw) {
                    anomalies.push("Chiều rộng cánh hoa lớn hơn chiều rộng đài hoa.");
                }
                if (sw > sl) {
                    anomalies.push("Chiều rộng đài hoa vượt quá chiều dài đài hoa.");
                }
                if (pl > sl * 1.15) {
                    anomalies.push("Chiều dài cánh hoa vượt quá chiều dài đài hoa đáng kể.");
                }

                if (anomalies.length > 0) {
                    warnBox.classList.remove('hidden');
                    document.getElementById('warningText').textContent = anomalies.join(" ");
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

                // Cập nhật kết quả dự đoán
                document.getElementById('resultIcon').textContent = data.icon;
                document.getElementById('resultName').textContent = data.prediction;
                document.getElementById('resultName').style.color = data.color;
                document.getElementById('resultDesc').textContent = data.desc;

                // Hiển thị Mẹo chăm sóc sinh học
                if (data.care_tips) {
                    document.getElementById('careBox').classList.remove('hidden');
                    document.getElementById('careTipsText').innerHTML = data.care_tips.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                }

                // Cập nhật xác suất
                document.getElementById('probBars').classList.remove('hidden');
                data.probabilities.forEach((p, idx) => {
                    const pct = (p * 100).toFixed(1) + '%';
                    document.getElementById(`prob${idx}`).textContent = pct;
                    document.getElementById(`bar${idx}`).style.width = pct;
                });

                // Confetti hiệu ứng
                confetti({ particleCount: 50, spread: 60, origin: { y: 0.7 } });

                // Ghi nhận lịch sử
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

            function clearHistory() {
                historyData = [];
                renderHistory();
            }

            function renderHistory() {
                const tbody = document.getElementById('historyTable');
                if (historyData.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="4" class="p-3 text-center text-slate-400">Chưa có lịch sử dự đoán</td></tr>';
                    return;
                }
                tbody.innerHTML = historyData.map(item => `
                    <tr class="hover:bg-slate-100/40 dark:hover:bg-slate-700/30 transition">
                        <td class="p-2.5 font-bold">${item.icon} ${item.name}</td>
                        <td class="p-2.5">${item.sepal} cm</td>
                        <td class="p-2.5">${item.petal} cm</td>
                        <td class="p-2.5"><span class="px-2 py-0.5 bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300 rounded-md text-[10px] font-semibold">Lớp ${item.id}</span></td>
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
    sl, sw, pl, pw = data.sepal_length, data.sepal_width, data.petal_length, data.petal_width
    features = [[sl, sw, pl, pw]]
    
    probs = None
    
    # 1. Nếu mô hình tồn tại, trích xuất điểm số phân lớp và áp dụng Temperature Softmax
    if model is not None:
        try:
            if hasattr(model, "predict_proba"):
                raw_probs = model.predict_proba(features)[0].tolist()
                # Phân bổ mềm xác suất (Label Smoothing) để loại bỏ 100% tuyệt đối
                probs = [p * 0.85 + 0.05 for p in raw_probs]
                total = sum(probs)
                probs = [round(p / total, 4) for p in probs]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(features)[0]
                if hasattr(scores, "__len__"):
                    exp_scores = [math.exp(s / 1.5) for s in scores]
                    total = sum(exp_scores)
                    probs = [round(e / total, 4) for e in exp_scores]
        except Exception:
            probs = None

    # 2. Thuật toán fallback mềm hóa: Tính toán xác suất mềm dựa trên khoảng cách đặc trưng sinh học
    if probs is None:
        # Tâm trung bình sinh học chuẩn của 3 loại Iris
        centers = [
            [5.0, 3.4, 1.5, 0.2],  # Setosa
            [5.9, 2.7, 4.2, 1.3],  # Versicolor
            [6.5, 3.0, 5.5, 2.0]   # Virginica
        ]
        
        # Tính khoảng cách Euclidean
        distances = []
        for c in centers:
            dist = math.sqrt((sl - c[0])**2 + (sw - c[1])**2 + (pl - c[2])**2 + (pw - c[3])**2)
            distances.append(-dist * 1.2)  # Hệ số mượt cho Softmax
            
        # Hàm Softmax quy đổi khoảng cách thành phần trăm mềm
        exp_dist = [math.exp(d) for d in distances]
        total = sum(exp_dist)
        probs = [round(e / total, 4) for e in exp_dist]

    # Chọn nhãn có xác suất cao nhất
    pred_id = probs.index(max(probs))

    species_info = SPECIES_MAP.get(pred_id, {
        "name": "Không xác định", 
        "icon": "❓", 
        "color": "#6B7280", 
        "desc": "", 
        "care_tips": ""
    })
    
    return {
        "class_id": pred_id,
        "prediction": species_info["name"],
        "icon": species_info["icon"],
        "color": species_info["color"],
        "desc": species_info["desc"],
        "care_tips": species_info.get("care_tips", ""),
        "probabilities": probs
    }
