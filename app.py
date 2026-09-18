from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris Nature Classifier")

# Load model SVM (nếu có)
try:
    model = joblib.load("svm_model.pkl")
except Exception:
    model = None

SPECIES_MAP = {
    0: {
        "name": "Iris-setosa",
        "icon": "🪻",
        "color": "#8b5cf6",
        "desc": "Cánh hoa ngắn thanh thoát, đài hoa rộng. Vẻ đẹp nhỏ gọn, mộc mạc.",
        "care_tips": "🌱 Thổ nhưỡng & Ánh sáng: Ưa đất tơi xốp, thoáng khí và thoát nước tốt. Phát triển rạng rỡ dưới ánh nắng ban mai nhẹ nhàng."
    },
    1: {
        "name": "Iris-versicolor",
        "icon": "🌷",
        "color": "#fb923c",  # Tông màu tulip hồng cam
        "desc": "Kích thước vừa vặn, sắc hoa hài hòa giữa thiên nhiên, vươn mình mềm mại.",
        "care_tips": "💧 Thổ nhưỡng & Ánh sáng: Ưa đất ẩm mịn, giàu mùn hữu cơ tự nhiên. Thích hợp không gian dịu mát có ánh sáng tán xạ."
    },
    2: {
        "name": "Iris-virginica",
        "icon": "🪷",
        "color": "#3b82f6",
        "desc": "Cánh hoa dài kiêu hãnh, vươn cao tràn đầy sức sống của núi rừng.",
        "care_tips": "☀️ Thổ nhưỡng & Ánh sáng: Ưa đất phù sa ẩm đầm lầy, chịu nước tốt. Phát triển mạnh mẽ dưới ánh nắng mặt trời trọn vẹn."
    }
}

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def calculate_soft_probabilities(features: list) -> list:
    centroids = np.array([
        [5.01, 3.43, 1.46, 0.25],  # Setosa
        [5.94, 2.77, 4.26, 1.33],  # Versicolor
        [6.59, 2.97, 5.55, 2.03]   # Virginica
    ])
    stds = np.array([0.5, 0.4, 0.5, 0.3])
    diff = (np.array(features) - centroids) / stds
    distances = np.sum(diff ** 2, axis=1)
    
    logits = -0.5 * distances
    exp_logits = np.exp((logits - np.max(logits)) / 1.5)
    probs = exp_logits / np.sum(exp_logits)
    
    probs = np.clip(probs, 0.005, 0.980)
    probs = probs / np.sum(probs)
    return probs.tolist()

@app.post("/predict")
def predict(data: IrisInput):
    features = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    
    if model is not None:
        try:
            if hasattr(model, "predict_proba"):
                raw_probs = model.predict_proba([features])[0]
            else:
                dec = model.decision_function([features])[0]
                exp_dec = np.exp(dec - np.max(dec))
                raw_probs = exp_dec / np.sum(exp_dec)
            
            probs = np.clip(raw_probs, 0.01, 0.98)
            probs = probs / np.sum(probs)
            pred_class = int(np.argmax(probs))
        except Exception:
            probs = calculate_soft_probabilities(features)
            pred_class = int(np.argmax(probs))
    else:
        probs = calculate_soft_probabilities(features)
        pred_class = int(np.argmax(probs))
        
    info = SPECIES_MAP[pred_class]
    return {
        "prediction": info["name"],
        "icon": info["icon"],
        "color": info["color"],
        "desc": info["desc"],
        "care_tips": info["care_tips"],
        "probabilities": [round(float(p), 4) for p in probs]
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="vi" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vườn Hoa Iris // Phân Tích Thiên Nhiên</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;1,400&display=swap" rel="stylesheet">
        <script>
            tailwind.config = {
                darkMode: 'class',
                theme: {
                    extend: {
                        fontFamily: {
                            sans: ['Plus Jakarta Sans', 'sans-serif'],
                            serif: ['Playfair Display', 'serif'],
                        },
                        colors: {
                            sage: {
                                50: '#f4f7f4',
                                100: '#e3ebe3',
                                200: '#c7d7c7',
                                300: '#a3be9e',
                                500: '#52796f',
                                700: '#354f52',
                                800: '#2f3e46',
                                900: '#1d2a22',
                                950: '#111914',
                            }
                        }
                    }
                }
            }
        </script>
        <style>
            body { background-color: #f4f7f4; }
            .dark body { background-color: #0f1712; }
            .nature-card {
                background: rgba(255, 255, 255, 0.75);
                backdrop-filter: blur(16px);
                border: 1px solid rgba(82, 121, 111, 0.12);
                box-shadow: 0 10px 30px -10px rgba(47, 62, 70, 0.05);
            }
            .dark .nature-card {
                background: rgba(23, 33, 26, 0.75);
                border: 1px solid rgba(163, 190, 158, 0.12);
                box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
            }
            .nature-card:hover { border-color: rgba(82, 121, 111, 0.3); }
            input[type=range] {
                height: 5px;
                border-radius: 9999px;
            }
        </style>
    </head>
    <body class="text-stone-800 dark:text-stone-100 font-sans min-h-screen p-4 md:p-8 selection:bg-emerald-200 selection:text-emerald-900 transition-colors duration-500">
        
        <!-- Ambient Atmospheric Glows -->
        <div class="fixed inset-0 pointer-events-none z-0 overflow-hidden">
            <div class="absolute -top-32 left-1/4 w-96 h-96 bg-emerald-200/30 dark:bg-emerald-900/15 rounded-full blur-3xl"></div>
            <div class="absolute top-1/3 -right-20 w-96 h-96 bg-teal-100/40 dark:bg-teal-900/10 rounded-full blur-3xl"></div>
            <div class="absolute -bottom-32 left-10 w-96 h-96 bg-stone-200/50 dark:bg-stone-900/30 rounded-full blur-3xl"></div>
        </div>

        <div class="max-w-6xl mx-auto space-y-6 relative z-10">
            
            <!-- HEADER -->
            <header class="nature-card p-5 rounded-3xl flex flex-wrap justify-between items-center gap-4">
                <div class="flex items-center gap-3.5">
                    <div class="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl text-emerald-600 dark:text-emerald-400">
                        <!-- Biểu tượng Lá Cây Xanh Tự Nhiên -->
                        <svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
                        </svg>
                    </div>
                    <div>
                        <div class="flex items-center gap-2.5">
                            <h1 class="text-xl font-bold font-serif tracking-wide text-stone-900 dark:text-stone-50">IRIS FLOWERS</h1>
                            <span class="px-2.5 py-0.5 text-[10px] font-medium bg-emerald-100 dark:bg-emerald-950/80 text-emerald-700 dark:text-emerald-300 border border-emerald-300/40 dark:border-emerald-800/50 rounded-full">Sương Ban Mai</span>
                        </div>
                        <p class="text-xs text-stone-500 dark:text-stone-400">Không gian nhận diện & lắng nghe nhịp điệu hoa cỏ</p>
                    </div>
                </div>

                <div class="flex items-center gap-3">
                    <div class="flex items-center gap-1.5 bg-stone-200/60 dark:bg-stone-900/60 p-1.5 rounded-2xl border border-stone-300/50 dark:border-stone-800">
                        <span class="text-[11px] text-stone-500 dark:text-stone-400 px-2 hidden sm:inline font-medium">Mẫu hoa:</span>
                        <button onclick="applyPreset(5.1, 3.5, 1.4, 0.2)" class="px-3 py-1 hover:bg-white dark:hover:bg-stone-800 text-purple-700 dark:text-purple-300 text-xs rounded-xl font-medium transition shadow-sm">Setosa</button>
                        <button onclick="applyPreset(5.2, 3.6, 4.2, 1.4)" class="px-3 py-1 hover:bg-white dark:hover:bg-stone-800 text-orange-600 dark:text-orange-300 text-xs rounded-xl font-medium transition shadow-sm">Versicolor</button>
                        <button onclick="applyPreset(6.5, 3.0, 5.5, 2.0)" class="px-3 py-1 hover:bg-white dark:hover:bg-stone-800 text-blue-700 dark:text-blue-300 text-xs rounded-xl font-medium transition shadow-sm">Virginica</button>
                    </div>

                    <button onclick="toggleTheme()" id="themeBtn" class="px-3 py-2 bg-stone-200/80 dark:bg-stone-800/80 hover:bg-stone-300 dark:hover:bg-stone-700 text-stone-700 dark:text-stone-200 rounded-2xl border border-stone-300/60 dark:border-stone-700 text-xs transition flex items-center gap-1.5 shadow-sm">
                        <span id="themeIcon">☀️</span>
                        <span id="themeText" class="hidden md:inline">Thanh Dịu</span>
                    </button>
                </div>
            </header>

            <!-- MAIN CONTENT GRID -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
                
                <!-- INPUT FORM CARD -->
                <div class="lg:col-span-7 nature-card p-6 rounded-3xl space-y-6">
                    <div class="flex justify-between items-center border-b border-stone-200 dark:border-stone-800/80 pb-4">
                        <h2 class="text-sm font-semibold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> Thông Số Kích Thước Tự Nhiên
                        </h2>
                        <span class="text-xs text-stone-400 font-mono">Đơn vị: cm</span>
                    </div>

                    <form id="irisForm" class="space-y-5">
                        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            <!-- Sepal Length -->
                            <div class="p-4 bg-stone-50/80 dark:bg-stone-900/40 rounded-2xl border border-stone-200/70 dark:border-stone-800 hover:border-emerald-300 dark:hover:border-emerald-800 transition space-y-2">
                                <div class="flex justify-between items-center">
                                    <span class="text-xs text-stone-600 dark:text-stone-400 font-medium">Chiều dài đài hoa</span>
                                    <span id="sl_val" class="font-mono text-xs font-bold text-emerald-700 dark:text-emerald-400 bg-emerald-100/60 dark:bg-emerald-950/60 px-2 py-0.5 rounded-md">5.1 cm</span>
                                </div>
                                <div class="flex items-center gap-2 pt-1">
                                    <button type="button" onclick="adjustValue('sepal_length', -0.1)" class="w-7 h-7 bg-stone-200 dark:bg-stone-800 hover:bg-stone-300 dark:hover:bg-stone-700 rounded-lg text-xs font-bold text-stone-700 dark:text-stone-300 transition">-</button>
                                    <input type="range" min="4.0" max="8.0" step="0.1" id="sepal_length" value="5.1" oninput="updateUI()" class="w-full accent-emerald-600 dark:accent-emerald-400 bg-stone-200 dark:bg-stone-800">
                                    <button type="button" onclick="adjustValue('sepal_length', 0.1)" class="w-7 h-7 bg-stone-200 dark:bg-stone-800 hover:bg-stone-300 dark:hover:bg-stone-700 rounded-lg text-xs font-bold text-stone-700 dark:text-stone-300 transition">+</button>
                                </div>
                            </div>

                            <!-- Sepal Width -->
                            <div class="p-4 bg-stone-50/80 dark:bg-stone-900/40 rounded-2xl border border-stone-200/70 dark:border-stone-800 hover:border-emerald-300 dark:hover:border-emerald-800 transition space-y-2">
                                <div class="flex justify-between items-center">
                                    <span class="text-xs text-stone-600 dark:text-stone-400 font-medium">Chiều rộng đài hoa</span>
                                    <span id="sw_val" class="font-mono text-xs font-bold text-emerald-700 dark:text-emerald-400 bg-emerald-100/60 dark:bg-emerald-950/60 px-2 py-0.5 rounded-md">3.5 cm</span>
                                </div>
                                <div class="flex items-center gap-2 pt-1">
                                    <button type="button" onclick="adjustValue('sepal_width', -0.1)" class="w-7 h-7 bg-stone-200 dark:bg-stone-800 hover:bg-stone-300 dark:hover:bg-stone-700 rounded-lg text-xs font-bold text-stone-700 dark:text-stone-300 transition">-</button>
                                    <input type="range" min="2.0" max="4.5" step="0.1" id="sepal_width" value="3.5" oninput="updateUI()" class="w-full accent-emerald-600 dark:accent-emerald-400 bg-stone-200 dark:bg-stone-800">
                                    <button type="button" onclick="adjustValue('sepal_width', 0.1)" class="w-7 h-7 bg-stone-200 dark:bg-stone-800 hover:bg-stone-300 dark:hover:bg-stone-700 rounded-lg text-xs font-bold text-stone-700 dark:text-stone-300 transition">+</button>
                                </div>
                            </div>

                            <!-- Petal Length -->
                            <div class="p-4 bg-stone-50/80 dark:bg-stone-900/40 rounded-2xl border border-stone-200/70 dark:border-stone-800 hover:border-emerald-300 dark:hover:border-emerald-800 transition space-y-2">
                                <div class="flex justify-between items-center">
                                    <span class="text-xs text-stone-600 dark:text-stone-400 font-medium">Chiều dài cánh hoa</span>
                                    <span id="pl_val" class="font-mono text-xs font-bold text-emerald-700 dark:text-emerald-400 bg-emerald-100/60 dark:bg-emerald-950/60 px-2 py-0.5 rounded-md">1.4 cm</span>
                                </div>
                                <div class="flex items-center gap-2 pt-1">
                                    <button type="button" onclick="adjustValue('petal_length', -0.1)" class="w-7 h-7 bg-stone-200 dark:bg-stone-800 hover:bg-stone-300 dark:hover:bg-stone-700 rounded-lg text-xs font-bold text-stone-700 dark:text-stone-300 transition">-</button>
                                    <input type="range" min="1.0" max="7.0" step="0.1" id="petal_length" value="1.4" oninput="updateUI()" class="w-full accent-emerald-600 dark:accent-emerald-400 bg-stone-200 dark:bg-stone-800">
                                    <button type="button" onclick="adjustValue('petal_length', 0.1)" class="w-7 h-7 bg-stone-200 dark:bg-stone-800 hover:bg-stone-300 dark:hover:bg-stone-700 rounded-lg text-xs font-bold text-stone-700 dark:text-stone-300 transition">+</button>
                                </div>
                            </div>

                            <!-- Petal Width -->
                            <div class="p-4 bg-stone-50/80 dark:bg-stone-900/40 rounded-2xl border border-stone-200/70 dark:border-stone-800 hover:border-emerald-300 dark:hover:border-emerald-800 transition space-y-2">
                                <div class="flex justify-between items-center">
                                    <span class="text-xs text-stone-600 dark:text-stone-400 font-medium">Chiều rộng cánh hoa</span>
                                    <span id="pw_val" class="font-mono text-xs font-bold text-emerald-700 dark:text-emerald-400 bg-emerald-100/60 dark:bg-emerald-950/60 px-2 py-0.5 rounded-md">0.2 cm</span>
                                </div>
                                <div class="flex items-center gap-2 pt-1">
                                    <button type="button" onclick="adjustValue('petal_width', -0.1)" class="w-7 h-7 bg-stone-200 dark:bg-stone-800 hover:bg-stone-300 dark:hover:bg-stone-700 rounded-lg text-xs font-bold text-stone-700 dark:text-stone-300 transition">-</button>
                                    <input type="range" min="0.1" max="2.5" step="0.1" id="petal_width" value="0.2" oninput="updateUI()" class="w-full accent-emerald-600 dark:accent-emerald-400 bg-stone-200 dark:bg-stone-800">
                                    <button type="button" onclick="adjustValue('petal_width', 0.1)" class="w-7 h-7 bg-stone-200 dark:bg-stone-800 hover:bg-stone-300 dark:hover:bg-stone-700 rounded-lg text-xs font-bold text-stone-700 dark:text-stone-300 transition">+</button>
                                </div>
                            </div>
                        </div>

                        <!-- Warning Alert Box -->
                        <div id="warningBox" class="hidden p-3.5 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900/50 rounded-2xl text-amber-800 dark:text-amber-300 text-xs flex items-start gap-2.5">
                            <span class="text-base">🌾</span>
                            <div>
                                <span class="font-semibold block uppercase text-[11px] text-amber-900 dark:text-amber-400">Lưu ý sinh thái</span>
                                <span id="warningText" class="text-stone-600 dark:text-stone-300 leading-relaxed"></span>
                            </div>
                        </div>

                        <!-- Submit Button -->
                        <button type="submit" class="w-full py-4 bg-stone-800 dark:bg-stone-100 hover:bg-stone-900 dark:hover:bg-white text-stone-100 dark:text-stone-900 font-bold rounded-2xl shadow-md transition duration-300 active:scale-[0.99] flex items-center justify-center gap-2 tracking-wide">
                            <svg class="w-5 h-5 text-emerald-400 dark:text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
                            PHÂN TÍCH & DỰ ĐOÁN
                        </button>
                    </form>
                </div>

                <!-- PREVIEW & RESULT CARD -->
                <div class="lg:col-span-5 space-y-6">
                    
                    <!-- SVG Visualizer -->
                    <div class="nature-card p-5 rounded-3xl space-y-3 text-center relative overflow-hidden">
                        <div class="flex justify-between items-center text-xs text-stone-400">
                            <span>MÔ PHỎNG HÌNH DÁNG HẠT & CÁNH</span>
                            <span class="text-emerald-600 dark:text-emerald-400 font-medium">Mô Phỏng Trực Tiếp</span>
                        </div>
                        <div class="h-32 bg-stone-100/70 dark:bg-stone-950/50 rounded-2xl flex items-center justify-center relative border border-stone-200/60 dark:border-stone-800">
                            <svg id="flowerSvg" class="transition-all duration-500 filter drop-shadow-md" width="100" height="100" viewBox="-50 -50 100 100">
                                <ellipse id="svgSepal" cx="0" cy="0" rx="20" ry="40" fill="#52796f" opacity="0.4" stroke="#2f3e46" stroke-width="1"/>
                                <ellipse id="svgSepal2" cx="0" cy="0" rx="40" ry="20" fill="#52796f" opacity="0.4" stroke="#2f3e46" stroke-width="1"/>
                                <circle id="svgPetal" cx="0" cy="0" r="15" fill="#e5989b" opacity="0.7" stroke="#b5838d" stroke-width="1.5"/>
                            </svg>
                        </div>
                    </div>

                    <!-- Prediction Result Card -->
                    <div id="resultCard" class="nature-card p-6 rounded-3xl text-center space-y-4 transition-all duration-300">
                        <div class="space-y-1">
                            <div id="resultIcon" class="text-5xl mb-2 animate-bounce inline-block">🍃</div>
                            <h3 id="resultName" class="text-2xl font-bold font-serif text-stone-800 dark:text-stone-100">Sẵn Sàng Khám Phá</h3>
                            <p id="resultDesc" class="text-xs text-stone-500 dark:text-stone-400">Điều chỉnh thông số và bấm để lắng nghe kết quả</p>
                        </div>

                        <div id="careBox" class="hidden p-4 bg-emerald-50/60 dark:bg-stone-900/80 border border-emerald-200 dark:border-emerald-900/40 rounded-2xl text-left text-xs text-stone-700 dark:text-stone-300 space-y-2">
                            <span class="font-semibold text-emerald-800 dark:text-emerald-400 text-[11px] uppercase tracking-wider block border-b border-emerald-200 dark:border-stone-800 pb-1">🌱 Đặc tính & Mẹo Chăm Sóc</span>
                            <p id="careTipsText" class="leading-relaxed"></p>
                        </div>

                        <!-- Confidence Bars -->
                        <div id="probBars" class="space-y-2.5 text-left hidden pt-3 border-t border-stone-200 dark:border-stone-800">
                            <span class="text-[10px] font-semibold text-stone-400 uppercase tracking-widest block">Mức độ tương thích</span>
                            
                            <div class="space-y-1">
                                <div class="flex justify-between text-xs"><span>Iris-setosa</span><span id="prob0" class="text-purple-600 dark:text-purple-400 font-bold">0%</span></div>
                                <div class="w-full bg-stone-200 dark:bg-stone-800 h-2 rounded-full overflow-hidden">
                                    <div id="bar0" class="bg-purple-500 h-full w-0 transition-all duration-500 rounded-full"></div>
                                </div>
                            </div>
                            <div class="space-y-1">
                                <div class="flex justify-between text-xs"><span>Iris-versicolor</span><span id="prob1" class="text-orange-500 dark:text-orange-400 font-bold">0%</span></div>
                                <div class="w-full bg-stone-200 dark:bg-stone-800 h-2 rounded-full overflow-hidden">
                                    <div id="bar1" class="bg-orange-400 h-full w-0 transition-all duration-500 rounded-full"></div>
                                </div>
                            </div>
                            <div class="space-y-1">
                                <div class="flex justify-between text-xs"><span>Iris-virginica</span><span id="prob2" class="text-blue-600 dark:text-blue-400 font-bold">0%</span></div>
                                <div class="w-full bg-stone-200 dark:bg-stone-800 h-2 rounded-full overflow-hidden">
                                    <div id="bar2" class="bg-blue-500 h-full w-0 transition-all duration-500 rounded-full"></div>
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
            </div>

            <!-- REFERENCE & LOGS -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <!-- Reference Table -->
                <div class="lg:col-span-6 nature-card p-5 rounded-3xl space-y-4">
                    <div class="flex justify-between items-center border-b border-stone-200 dark:border-stone-800 pb-3">
                        <h3 class="text-xs font-semibold text-stone-700 dark:text-stone-300 uppercase tracking-wider flex items-center gap-2">
                            <span>📖</span> Bảng Chỉ Số Tham Chiếu Tự Nhiên
                        </h3>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-xs font-mono">
                            <thead class="text-stone-400 border-b border-stone-200 dark:border-stone-800 uppercase text-[10px]">
                                <tr>
                                    <th class="pb-2">Loài Hoa</th>
                                    <th class="pb-2">Đài (Dài)</th>
                                    <th class="pb-2">Đài (Rộng)</th>
                                    <th class="pb-2">Cánh (Dài)</th>
                                    <th class="pb-2">Cánh (Rộng)</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-stone-200/60 dark:divide-stone-800/60 text-stone-700 dark:text-stone-300">
                                <tr>
                                    <td class="py-2.5 font-bold text-purple-600 dark:text-purple-400">Setosa</td>
                                    <td>4.3 - 5.8</td>
                                    <td>2.3 - 4.4</td>
                                    <td>1.0 - 1.9</td>
                                    <td>0.1 - 0.6</td>
                                </tr>
                                <tr>
                                    <td class="py-2.5 font-bold text-orange-500 dark:text-orange-400">Versicolor</td>
                                    <td>4.9 - 7.0</td>
                                    <td>2.0 - 3.4</td>
                                    <td>3.0 - 5.1</td>
                                    <td>1.0 - 1.8</td>
                                </tr>
                                <tr>
                                    <td class="py-2.5 font-bold text-blue-600 dark:text-blue-400">Virginica</td>
                                    <td>4.9 - 7.9</td>
                                    <td>2.2 - 3.8</td>
                                    <td>4.5 - 6.9</td>
                                    <td>1.4 - 2.5</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- History Logs -->
                <div class="lg:col-span-6 nature-card p-5 rounded-3xl space-y-4">
                    <div class="flex justify-between items-center border-b border-stone-200 dark:border-stone-800 pb-3">
                        <h3 class="text-xs font-semibold text-stone-700 dark:text-stone-300 uppercase tracking-wider flex items-center gap-2">
                            <span>📜</span> Nhật Ký Quan Sát Thiên Nhiên
                        </h3>
                        <button onclick="clearHistory()" class="px-2.5 py-1 bg-stone-200/60 dark:bg-stone-800 hover:bg-rose-100 dark:hover:bg-rose-950/50 text-rose-600 dark:text-rose-400 text-[11px] rounded-lg font-mono transition">
                            Xóa Lịch Sử
                        </button>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-xs font-mono">
                            <thead class="text-stone-400 border-b border-stone-200 dark:border-stone-800 uppercase text-[10px]">
                                <tr>
                                    <th class="pb-2">Kết quả</th>
                                    <th class="pb-2">Đài (DxR)</th>
                                    <th class="pb-2">Cánh (DxR)</th>
                                </tr>
                            </thead>
                            <tbody id="historyTable" class="divide-y divide-stone-200/60 dark:divide-stone-800/60 text-stone-700 dark:text-stone-300">
                                <tr><td colspan="3" class="py-4 text-center text-stone-400 font-sans">Chưa có nhật ký quan sát</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

        </div>

        <script>
            let historyData = [];

            function toggleTheme() {
                const html = document.documentElement;
                const icon = document.getElementById('themeIcon');
                const text = document.getElementById('themeText');
                
                if (html.classList.contains('dark')) {
                    html.classList.remove('dark');
                    icon.textContent = '🌙';
                    text.textContent = 'Sáng';
                } else {
                    html.classList.add('dark');
                    icon.textContent = '☀️';
                    text.textContent = 'Tối';
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

                document.getElementById('svgSepal').setAttribute('ry', sl * 5.5);
                document.getElementById('svgSepal').setAttribute('rx', sw * 5.5);
                document.getElementById('svgSepal2').setAttribute('rx', sl * 5.5);
                document.getElementById('svgSepal2').setAttribute('ry', sw * 5.5);
                document.getElementById('svgPetal').setAttribute('r', (pl + pw) * 3.8);

                const warnBox = document.getElementById('warningBox');
                let anomalies = [];

                if (pw > sw) anomalies.push("Chiều rộng cánh hoa lớn hơn chiều rộng đài hoa.");
                if (sw > sl) anomalies.push("Chiều rộng đài hoa vượt quá chiều dài đài.");
                if (pl > sl * 1.15) anomalies.push("Cánh hoa dài bất thường so với đài.");

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

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();

                    document.getElementById('resultIcon').textContent = data.icon;
                    document.getElementById('resultName').textContent = data.prediction;
                    document.getElementById('resultName').style.color = data.color;
                    document.getElementById('resultDesc').textContent = data.desc;
                    
                    document.getElementById('careTipsText').innerHTML = data.care_tips;
                    document.getElementById('careBox').classList.remove('hidden');

                    const probs = data.probabilities;
                    document.getElementById('prob0').textContent = Math.round(probs[0] * 100) + '%';
                    document.getElementById('prob1').textContent = Math.round(probs[1] * 100) + '%';
                    document.getElementById('prob2').textContent = Math.round(probs[2] * 100) + '%';

                    document.getElementById('bar0').style.width = (probs[0] * 100) + '%';
                    document.getElementById('bar1').style.width = (probs[1] * 100) + '%';
                    document.getElementById('bar2').style.width = (probs[2] * 100) + '%';
                    document.getElementById('probBars').classList.remove('hidden');

                    if (window.confetti) {
                        confetti({ particleCount: 50, spread: 60, origin: { y: 0.7 } });
                    }

                    addHistory(data.prediction, payload);
                } catch (err) {
                    console.error("Error predicting:", err);
                }
            });

            function addHistory(pred, p) {
                historyData.unshift({ pred, ...p });
                if (historyData.length > 5) historyData.pop();
                renderHistory();
            }

            function renderHistory() {
                const tbody = document.getElementById('historyTable');
                if (historyData.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="3" class="py-4 text-center text-stone-400 font-sans">Chưa có nhật ký quan sát</td></tr>';
                    return;
                }
                tbody.innerHTML = historyData.map(item => `
                    <tr>
                        <td class="py-2 font-bold">${item.pred}</td>
                        <td>${item.sepal_length} x ${item.sepal_width}</td>
                        <td>${item.petal_length} x ${item.petal_width}</td>
                    </tr>
                `).join('');
            }

            function clearHistory() {
                historyData = [];
                renderHistory();
            }

            updateUI();
        </script>
    </body>
    </html>
    """
