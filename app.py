from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

app = FastAPI(title="Iris AI Botanical Lab")

try:
    model = joblib.load("svm_model.pkl")
except Exception:
    model = None

SPECIES_MAP = {
    0: {
        "name": "Iris-setosa",
        "icon": "🪻",
        "color": "#8B5CF6",
        "desc": "Cánh hoa ngắn, đài hoa rộng đặc trưng",
        "tips": "Ưa vùng đất ẩm ướt, ánh sáng mặt trời bán phần.",
        "avg": [5.0, 3.4, 1.5, 0.2]
    },
    1: {
        "name": "Iris-versicolor",
        "icon": "🌺",
        "color": "#EC4899",
        "desc": "Màu hoa rực rỡ, kích thước cánh & đài cân đối",
        "tips": "Phát triển tốt bên ven ao hồ, đầm lầy, đất giàu mùn.",
        "avg": [5.9, 2.7, 4.2, 1.3]
    },
    2: {
        "name": "Iris-virginica",
        "icon": "🪷",
        "color": "#3B82F6",
        "desc": "Kích thước hoa lớn nhất, đài hoa dài nổi bật",
        "tips": "Đòi hỏi độ ẩm cao liên tục và không gian phát triển lớn.",
        "avg": [6.5, 3.0, 5.5, 2.0]
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
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris AI Studio - Botanical Analytics</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Plus Jakarta Sans', sans-serif; }
            .glass { background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(12px); }
        </style>
    </head>
    <body class="bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-100 min-h-screen text-slate-800 p-4 md:p-8">
        <div class="max-w-6xl mx-auto space-y-6">
            
            <header class="glass rounded-3xl p-5 shadow-sm border border-white/60 flex justify-between items-center">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-teal-500 rounded-2xl flex items-center justify-center text-white text-xl shadow-md">🌿</div>
                    <div>
                        <h1 class="text-xl font-extrabold bg-gradient-to-r from-teal-600 to-emerald-600 bg-clip-text text-transparent">Botanical AI Lab</h1>
                        <p class="text-xs text-slate-500">Phân tích & Nhận diện loài hoa Iris đa chiều</p>
                    </div>
                </div>
                <button onclick="exportCSV()" class="px-3 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-semibold shadow-md transition flex items-center gap-1.5">
                    <span>📥</span> Xuất Lịch Sử CSV
                </button>
            </header>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
                
                <div class="lg:col-span-5 glass p-6 rounded-3xl shadow-sm border border-white/60 space-y-5">
                    <div class="flex justify-between items-center">
                        <h2 class="text-sm font-bold text-slate-700 uppercase tracking-wider">Thông Số Đầu Vào</h2>
                        <span class="text-[10px] px-2 py-0.5 bg-teal-100 text-teal-700 rounded-full font-semibold">Tự động cập nhật</span>
                    </div>

                    <div class="grid grid-cols-3 gap-2">
                        <button onclick="setPreset(5.1, 3.5, 1.4, 0.2)" class="p-2 text-xs bg-purple-50 text-purple-700 rounded-xl hover:bg-purple-100 font-semibold transition border border-purple-200">🪻 Setosa</button>
                        <button onclick="setPreset(6.0, 2.7, 5.1, 1.6)" class="p-2 text-xs bg-pink-50 text-pink-700 rounded-xl hover:bg-pink-100 font-semibold transition border border-pink-200">🌺 Versicolor</button>
                        <button onclick="setPreset(6.5, 3.0, 5.5, 2.0)" class="p-2 text-xs bg-blue-50 text-blue-700 rounded-xl hover:bg-blue-100 font-semibold transition border border-blue-200">🪷 Virginica</button>
                    </div>

                    <div class="space-y-4">
                        <div>
                            <div class="flex justify-between text-xs font-semibold mb-1">
                                <span>Chiều dài đài (Sepal Length)</span>
                                <span id="v_sl" class="text-teal-600 font-extrabold">5.1 cm</span>
                            </div>
                            <input type="range" min="4.0" max="8.0" step="0.1" id="sl" value="5.1" oninput="onInputChange()" class="w-full accent-teal-600">
                        </div>
                        <div>
                            <div class="flex justify-between text-xs font-semibold mb-1">
                                <span>Chiều rộng đài (Sepal Width)</span>
                                <span id="v_sw" class="text-teal-600 font-extrabold">3.5 cm</span>
                            </div>
                            <input type="range" min="2.0" max="4.5" step="0.1" id="sw" value="3.5" oninput="onInputChange()" class="w-full accent-teal-600">
                        </div>
                        <div>
                            <div class="flex justify-between text-xs font-semibold mb-1">
                                <span>Chiều dài cánh (Petal Length)</span>
                                <span id="v_pl" class="text-teal-600 font-extrabold">1.4 cm</span>
                            </div>
                            <input type="range" min="1.0" max="7.0" step="0.1" id="pl" value="1.4" oninput="onInputChange()" class="w-full accent-teal-600">
                        </div>
                        <div>
                            <div class="flex justify-between text-xs font-semibold mb-1">
                                <span>Chiều rộng cánh (Petal Width)</span>
                                <span id="v_pw" class="text-teal-600 font-extrabold">0.2 cm</span>
                            </div>
                            <input type="range" min="0.1" max="2.5" step="0.1" id="pw" value="0.2" oninput="onInputChange()" class="w-full accent-teal-600">
                        </div>
                    </div>

                    <div id="anomalyAlert" class="hidden p-3 bg-amber-50 border border-amber-200 text-amber-800 rounded-xl text-xs flex items-center gap-2">
                        <span>⚠️</span> <span>Thông số ngoại lệ: Hình dạng đài/cánh bất thường so với tự nhiên.</span>
                    </div>
                </div>

                <div class="lg:col-span-7 space-y-6">
                    
                    <div id="resultBanner" class="glass p-6 rounded-3xl shadow-sm border border-white/60 transition-all duration-300">
                        <div class="flex items-center justify-between">
                            <div class="flex items-center gap-4">
                                <span id="resIcon" class="text-6xl animate-pulse">🪻</span>
                                <div>
                                    <h2 id="resName" class="text-2xl font-extrabold">Iris-setosa</h2>
                                    <p id="resDesc" class="text-xs text-slate-500 mt-0.5">Cánh hoa ngắn, đài hoa rộng đặc trưng</p>
                                </div>
                            </div>
                            <button onclick="copySummary()" class="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold rounded-xl transition">
                                📋 Sao Chép
                            </button>
                        </div>
                        <div class="mt-4 pt-3 border-t border-slate-200/60 text-xs text-slate-600 flex items-center gap-2">
                            <span>💡 <b>Mẹo chăm sóc:</b></span>
                            <span id="resTips">Ưa vùng đất ẩm ướt, ánh sáng mặt trời bán phần.</span>
                        </div>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="glass p-4 rounded-3xl border border-white/60">
                            <span class="text-xs font-bold text-slate-600 block mb-2 text-center">Đồ Thị Radar So Với Trung Bình</span>
                            <div class="h-48 relative">
                                <canvas id="radarChart"></canvas>
                            </div>
                        </div>

                        <div class="glass p-4 rounded-3xl border border-white/60">
                            <span class="text-xs font-bold text-slate-600 block mb-2 text-center">Không Gian Đặc Trưng (Cánh Hoa)</span>
                            <div class="h-48 relative">
                                <canvas id="scatterChart"></canvas>
                            </div>
                        </div>
                    </div>

                </div>
            </div>

            <div class="glass p-5 rounded-3xl shadow-sm border border-white/60">
                <h3 class="text-sm font-bold mb-3">📜 Nhật Ký Phân Tích Gần Đây</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs text-slate-600">
                        <thead class="bg-slate-100/70 font-semibold uppercase">
                            <tr>
                                <th class="p-2.5 rounded-l-xl">Loài Hoa</th>
                                <th class="p-2.5">Đài Hoa (D x R)</th>
                                <th class="p-2.5">Cánh Hoa (D x R)</th>
                                <th class="p-2.5 rounded-r-xl">Độ Tin Cậy</th>
                            </tr>
                        </thead>
                        <tbody id="historyBody" class="divide-y divide-slate-100">
                            <tr><td colspan="4" class="p-3 text-center text-slate-400">Chưa có dữ liệu lịch sử</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

        </div>

        <script>
            let radarChart, scatterChart;
            let historyLog = [];
            let currentRes = {};
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

            // Play Subtle UI Sound
            function playBeep() {
                try {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.frequency.value = 440;
                    gain.gain.value = 0.01;
                    osc.connect(gain);
                    gain.connect(audioCtx.destination);
                    osc.start();
                    osc.stop(audioCtx.currentTime + 0.05);
                } catch(e) {}
            }

            function initCharts() {
                // Radar Chart Setup
                const ctxRadar = document.getElementById('radarChart').getContext('2d');
                radarChart = new Chart(ctxRadar, {
                    type: 'radar',
                    data: {
                        labels: ['Dài Đài', 'Rộng Đài', 'Dài Cánh', 'Rộng Cánh'],
                        datasets: [
                            { label: 'Đầu Vào', data: [5.1, 3.5, 1.4, 0.2], borderColor: '#0D9488', backgroundColor: 'rgba(13, 148, 136, 0.2)' },
                            { label: 'Trung Bình Loài', data: [5.0, 3.4, 1.5, 0.2], borderColor: '#8B5CF6', backgroundColor: 'rgba(139, 92, 246, 0.1)' }
                        ]
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
                });

                // Scatter Chart Setup
                const ctxScatter = document.getElementById('scatterChart').getContext('2d');
                scatterChart = new Chart(ctxScatter, {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            label: 'Mẫu Hiện Tại',
                            data: [{ x: 1.4, y: 0.2 }],
                            backgroundColor: '#0D9488',
                            pointRadius: 8
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { title: { display: true, text: 'Dài Cánh (cm)' }, min: 0, max: 8 },
                            y: { title: { display: true, text: 'Rộng Cánh (cm)' }, min: 0, max: 3 }
                        },
                        plugins: { legend: { display: false } }
                    }
                });
            }

            function setPreset(sl, sw, pl, pw) {
                document.getElementById('sl').value = sl;
                document.getElementById('sw').value = sw;
                document.getElementById('pl').value = pl;
                document.getElementById('pw').value = pw;
                onInputChange();
            }

            let debounceTimer;
            function onInputChange() {
                playBeep();
                const sl = parseFloat(document.getElementById('sl').value);
                const sw = parseFloat(document.getElementById('sw').value);
                const pl = parseFloat(document.getElementById('pl').value);
                const pw = parseFloat(document.getElementById('pw').value);

                document.getElementById('v_sl').textContent = sl + ' cm';
                document.getElementById('v_sw').textContent = sw + ' cm';
                document.getElementById('v_pl').textContent = pl + ' cm';
                document.getElementById('v_pw').textContent = pw + ' cm';

                // Check anomaly
                const anomaly = document.getElementById('anomalyAlert');
                if (pw > sw || pl > sl * 1.2) {
                    anomaly.classList.remove('hidden');
                } else {
                    anomaly.classList.add('hidden');
                }

                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(() => requestPrediction(sl, sw, pl, pw), 200);
            }

            async function requestPrediction(sl, sw, pl, pw) {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sepal_length: sl, sepal_width: sw, petal_length: pl, petal_width: pw })
                });
                const data = await res.json();
                currentRes = data;

                // Update UI Banner
                document.getElementById('resIcon').textContent = data.icon;
                document.getElementById('resName').textContent = data.prediction;
                document.getElementById('resName').style.color = data.color;
                document.getElementById('resDesc').textContent = data.desc;
                document.getElementById('resTips').textContent = data.tips;

                // Update Radar Chart
                radarChart.data.datasets[0].data = [sl, sw, pl, pw];
                radarChart.data.datasets[1].data = data.avg;
                radarChart.data.datasets[1].borderColor = data.color;
                radarChart.update();

                // Update Scatter Chart
                scatterChart.data.datasets[0].data = [{ x: pl, y: pw }];
                scatterChart.data.datasets[0].backgroundColor = data.color;
                scatterChart.update();

                // Add to history
                const probPercent = (Math.max(...data.probabilities) * 100).toFixed(1) + '%';
                historyLog.unshift({
                    name: `${data.icon} ${data.prediction}`,
                    sepal: `${sl} x ${sw}`,
                    petal: `${pl} x ${pw}`,
                    confidence: probPercent
                });
                if (historyLog.length > 5) historyLog.pop();
                renderHistory();
            }

            function renderHistory() {
                document.getElementById('historyBody').innerHTML = historyLog.map(item => `
                    <tr class="hover:bg-slate-50/50 transition">
                        <td class="p-2.5 font-bold">${item.name}</td>
                        <td class="p-2.5">${item.sepal}</td>
                        <td class="p-2.5">${item.petal}</td>
                        <td class="p-2.5"><span class="px-2 py-0.5 bg-emerald-100 text-emerald-800 rounded font-semibold text-[10px]">${item.confidence}</span></td>
                    </tr>
                `).join('');
            }

            function copySummary() {
                const text = `Loài hoa: ${currentRes.prediction}\nKích thước cánh: ${document.getElementById('v_pl').textContent} x ${document.getElementById('v_pw').textContent}\nKích thước đài: ${document.getElementById('v_sl').textContent} x ${document.getElementById('v_sw').textContent}`;
                navigator.clipboard.writeText(text);
                alert("Đã sao chép tóm tắt kết quả!");
            }

            function exportCSV() {
                if (historyLog.length === 0) return alert("Chưa có lịch sử để xuất file!");
                let csv = "Loai Hoa,Dai Hoa,Canh Hoa,Do Tin Cay\n";
                historyLog.forEach(r => {
                    csv += `"${r.name}","${r.sepal}","${r.petal}","${r.confidence}"\n`;
                });
                const blob = new Blob([csv], { type: 'text/csv' });
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'iris_history.csv';
                a.click();
            }

            window.onload = () => {
                initCharts();
                onInputChange();
            };
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    pred_id = int(model.predict(features)[0]) if model else 0
    
    try:
        probs = model.predict_proba(features)[0].tolist()
    except Exception:
        probs = [0.0, 0.0, 0.0]
        probs[pred_id] = 1.0

    species_info = SPECIES_MAP.get(pred_id, SPECIES_MAP[0])
    
    return {
        "class_id": pred_id,
        "prediction": species_info["name"],
        "icon": species_info["icon"],
        "color": species_info["color"],
        "desc": species_info["desc"],
        "tips": species_info["tips"],
        "avg": species_info["avg"],
        "probabilities": probs
    }
