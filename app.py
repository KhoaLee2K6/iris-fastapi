from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris Sky & Nature Classifier")

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
        "color": "#fb923c",
        "desc": "Kích thước vừa vặn, sắc hoa hài hòa giữa thiên nhiên, vươn mình mềm mại.",
        "care_tips": "💧 Thổ nhưỡng & Ánh sáng: Ưa đất ẩm mịn, giàu mùn hữu cơ tự nhiên. Thích hợp không gian dịu mát có ánh sáng tán xạ."
    },
    2: {
        "name": "Iris-virginica",
        "icon": "🪷",
        "color": "#0284c7",
        "desc": "Cánh hoa dài kiêu hãnh, vươn cao tràn đầy sức sống dưới bầu trời.",
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
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris Flowers - Ánh Bình Minh & Bầu Trời Tinh Tú</title>
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
                        }
                    }
                }
            }
        </script>
        <style>
            body {
                background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 35%, #fed7aa 70%, #fef3c7 100%);
                background-attachment: fixed;
            }
            .dark body {
                background: linear-gradient(135deg, #030712 0%, #0b132b 40%, #1c2541 100%);
                background-attachment: fixed;
            }

            .nature-card {
                background: rgba(255, 255, 255, 0.75);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.9);
                box-shadow: 0 10px 35px -10px rgba(251, 146, 60, 0.2);
            }
            .dark .nature-card {
                background: rgba(15, 23, 42, 0.75);
                border: 1px solid rgba(255, 255, 255, 0.12);
                box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7);
            }

            input[type=range] {
                height: 6px;
                border-radius: 9999px;
            }

            .sunlight-container { display: block; }
            .dark .sunlight-container { display: none; }

            /* Hiệu ứng chiếu sáng cho biểu tượng Ban ngày */
            .sun-icon-glow {
                display: inline-block;
                animation: sunlightIconGlow 2.5s infinite ease-in-out alternate;
            }
            @keyframes sunlightIconGlow {
                0% {
                    filter: drop-shadow(0 0 3px rgba(245, 158, 11, 0.6)) drop-shadow(0 0 6px rgba(251, 191, 36, 0.4));
                    transform: scale(1);
                }
                100% {
                    filter: drop-shadow(0 0 10px rgba(245, 158, 11, 0.9)) drop-shadow(0 0 18px rgba(251, 146, 60, 0.8));
                    transform: scale(1.18);
                }
            }

            .sun-core {
                position: absolute;
                top: -120px;
                left: -120px;
                width: 550px;
                height: 550px;
                background: radial-gradient(circle, rgba(254, 240, 138, 0.95) 0%, rgba(251, 146, 60, 0.55) 45%, rgba(255, 255, 255, 0) 70%);
                border-radius: 50%;
                filter: blur(35px);
                animation: sunPulse 6s infinite ease-in-out alternate;
            }
            @keyframes sunPulse {
                0% { transform: scale(0.9) rotate(0deg); opacity: 0.85; }
                100% { transform: scale(1.15) rotate(15deg); opacity: 1; }
            }

            .sun-ray {
                position: absolute;
                top: -20%;
                left: -10%;
                width: 150vw;
                height: 150vh;
                background: repeating-conic-gradient(
                    from 0deg at 10% 10%,
                    rgba(253, 224, 71, 0.2) 0deg 15deg,
                    transparent 15deg 30deg
                );
                filter: blur(10px);
                pointer-events: none;
                animation: rayRotate 70s linear infinite;
            }
            @keyframes rayRotate {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }

            .sun-particle {
                position: absolute;
                background: radial-gradient(circle, #ffffff 0%, #fde047 100%);
                border-radius: 50%;
                box-shadow: 0 0 10px rgba(251, 146, 60, 0.8);
                animation: floatUp var(--duration, 8s) infinite ease-in-out var(--delay, 0s);
            }
            @keyframes floatUp {
                0% { transform: translateY(0) translateX(0) scale(0.8); opacity: 0; }
                20% { opacity: 0.9; }
                80% { opacity: 0.9; }
                100% { transform: translateY(-110vh) translateX(var(--drift, 40px)) scale(1.3); opacity: 0; }
            }

            .stars-container { display: none; }
            .dark .stars-container { display: block; }
            
            .star {
                position: absolute;
                background-color: #ffffff;
                border-radius: 50%;
                animation: twinkle var(--duration, 3s) infinite ease-in-out var(--delay, 0s);
            }
            @keyframes twinkle {
                0%, 100% { opacity: 0.2; transform: scale(0.8); }
                50% { opacity: 1; transform: scale(1.5); box-shadow: 0 0 10px rgba(255, 255, 255, 1); }
            }

            .star-sparkle {
                display: inline-block;
                animation: sparkleGlow 2s infinite ease-in-out alternate;
            }
            @keyframes sparkleGlow {
                0% { transform: scale(1) rotate(0deg); filter: drop-shadow(0 0 2px #38bdf8); }
                50% { transform: scale(1.25) rotate(15deg); filter: drop-shadow(0 0 8px #818cf8) drop-shadow(0 0 12px #e0e7ff); }
                100% { transform: scale(1) rotate(0deg); filter: drop-shadow(0 0 2px #38bdf8); }
            }

            .shooting-star {
                position: absolute;
                height: 2px;
                background: linear-gradient(-45deg, rgba(255, 255, 255, 1), rgba(147, 197, 253, 0.8), rgba(255, 255, 255, 0));
                border-radius: 999px;
                filter: drop-shadow(0 0 8px rgba(255, 255, 255, 0.9));
                animation: tail var(--speed, 3s) ease-in-out infinite var(--delay, 0s), shooting var(--speed, 3s) ease-in-out infinite var(--delay, 0s);
                opacity: 0;
            }
            @keyframes tail {
                0% { width: 0; }
                30% { width: 140px; }
                100% { width: 0; }
            }
            @keyframes shooting {
                0% { transform: translateX(0) translateY(0) rotate(-35deg); opacity: 1; }
                70% { opacity: 1; }
                100% { transform: translateX(-900px) translateY(600px) rotate(-35deg); opacity: 0; }
            }
        </style>
    </head>
    <body class="text-slate-800 dark:text-slate-100 font-sans min-h-screen p-4 md:p-8 transition-colors duration-500 overflow-x-hidden">
        
        <!-- HIỆU ỨNG GIAO DIỆN BAN MAI -->
        <div id="sunlight" class="sunlight-container fixed inset-0 pointer-events-none z-0 overflow-hidden">
            <div class="sun-core"></div>
            <div class="sun-ray"></div>
            <div class="absolute top-1/4 right-10 w-96 h-96 bg-amber-200/40 rounded-full blur-3xl"></div>
            <div class="absolute bottom-10 left-1/3 w-80 h-80 bg-rose-200/30 rounded-full blur-3xl"></div>
            <div id="sunParticles"></div>
        </div>

        <!-- HIỆU ỨNG GIAO DIỆN BAN ĐÊM -->
        <div id="stars" class="stars-container fixed inset-0 pointer-events-none z-0 overflow-hidden">
            <div id="starsList"></div>
            <div id="shootingStarsList"></div>
        </div>

        <div class="max-w-6xl mx-auto space-y-6 relative z-10">
            
            <!-- HEADER -->
            <header class="nature-card p-5 rounded-3xl flex flex-wrap justify-between items-center gap-4">
                <div class="flex items-center gap-3.5">
                    <div class="p-3 bg-amber-500/20 dark:bg-indigo-500/20 border border-amber-400/40 dark:border-indigo-400/30 rounded-2xl text-amber-700 dark:text-indigo-300">
                        <svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
                        </svg>
                    </div>
                    <div>
                        <div class="flex items-center gap-2.5">
                            <h1 class="text-xl font-bold font-serif tracking-wide text-amber-950 dark:text-slate-50">IRIS FLOWERS</h1>
                            <span id="skyBadge" class="px-3 py-0.5 text-[11px] font-semibold bg-amber-100 dark:bg-indigo-950/80 text-amber-900 dark:text-indigo-200 border border-amber-300/60 dark:border-indigo-700/60 rounded-full flex items-center gap-1 shadow-sm transition-all duration-300">
                                <span class="sun-icon-glow">🌅</span> Ánh bình minh
                            </span>
                        </div>
                        <p id="subHeadline" class="text-xs text-amber-900/70 dark:text-slate-400">Như tia nắng đầu ngày, mở ra thế giới rực rỡ của muôn sắc hoa </p>
                    </div>
                </div>

                <div class="flex items-center gap-3">
                    <div class="flex items-center gap-1.5 bg-white/60 dark:bg-slate-900/60 p-1.5 rounded-2xl border border-amber-200/60 dark:border-slate-800">
                        <span class="text-[11px] text-amber-900/70 dark:text-slate-400 px-2 hidden sm:inline font-medium">Mẫu hoa:</span>
                        <button type="button" onclick="applyPreset(5.1, 3.5, 1.4, 0.2)" class="px-3 py-1 hover:bg-white dark:hover:bg-slate-800 text-purple-700 dark:text-purple-300 text-xs rounded-xl font-medium transition shadow-sm">Setosa</button>
                        <button type="button" onclick="applyPreset(5.9, 2.8, 4.3, 1.3)" class="px-3 py-1 hover:bg-white dark:hover:bg-slate-800 text-orange-600 dark:text-orange-300 text-xs rounded-xl font-medium transition shadow-sm">Versicolor</button>
                        <button type="button" onclick="applyPreset(6.5, 3.0, 5.5, 2.0)" class="px-3 py-1 hover:bg-white dark:hover:bg-slate-800 text-sky-700 dark:text-sky-300 text-xs rounded-xl font-medium transition shadow-sm">Virginica</button>
                    </div>

                    <!-- Nút Chuyển Chế Độ Ngày/Đêm -->
                    <button type="button" onclick="toggleTheme()" id="themeBtn" class="px-3.5 py-2 bg-white/80 dark:bg-slate-800/80 hover:bg-white dark:hover:bg-slate-700 text-amber-900 dark:text-slate-200 rounded-2xl border border-amber-200/80 dark:border-slate-700 text-xs transition flex items-center gap-2 shadow-sm font-semibold">
                        <span id="themeIcon" class="sun-icon-glow">🌅</span>
                        <span id="themeText">Ban ngày</span>
                    </button>
                </div>
            </header>

            <!-- MAIN CONTENT GRID -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
                
                <!-- INPUT FORM CARD -->
                <div class="lg:col-span-7 nature-card p-6 rounded-3xl space-y-6">
                    <div class="flex justify-between items-center border-b border-amber-200/60 dark:border-slate-800 pb-4">
                        <h2 class="text-sm font-semibold text-amber-800 dark:text-indigo-400 uppercase tracking-wider flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span> Thông Số Kích Thước Tự Nhiên
                        </h2>
                        <span class="text-xs text-amber-900/60 dark:text-slate-400 font-mono">Đơn vị: cm</span>
                    </div>

                    <form id="irisForm" onsubmit="submitForm(event)" class="space-y-4">
                        <!-- Sepal Length -->
                        <div class="p-3.5 bg-white/80 dark:bg-slate-900/40 rounded-2xl border border-amber-100 dark:border-slate-800 hover:border-amber-300 dark:hover:border-indigo-800 transition">
                            <div class="grid grid-cols-12 gap-2 items-center">
                                <span class="col-span-4 text-xs text-slate-700 dark:text-slate-300 font-medium">Đài hoa (Dài)</span>
                                <div class="col-span-8 flex items-center gap-2">
                                    <button type="button" onclick="adjustValue('sepal_length', -0.1)" class="w-7 h-7 flex-none bg-amber-100 dark:bg-slate-800 hover:bg-amber-200 dark:hover:bg-slate-700 rounded-lg text-xs font-bold text-amber-900 dark:text-slate-300 transition">-</button>
                                    <input type="range" min="4.0" max="8.0" step="0.1" id="sepal_length" value="5.1" oninput="updateUI()" class="w-full accent-amber-500 dark:accent-indigo-400 bg-amber-100 dark:bg-slate-800">
                                    <button type="button" onclick="adjustValue('sepal_length', 0.1)" class="w-7 h-7 flex-none bg-amber-100 dark:bg-slate-800 hover:bg-amber-200 dark:hover:bg-slate-700 rounded-lg text-xs font-bold text-amber-900 dark:text-slate-300 transition">+</button>
                                    <span id="sl_val" class="w-16 flex-none text-right font-mono text-xs font-bold text-amber-700 dark:text-indigo-400 bg-amber-50 dark:bg-slate-900/80 px-2 py-1 rounded-md border border-amber-200/50 dark:border-indigo-900/50">5.1 cm</span>
                                </div>
                            </div>
                        </div>

                        <!-- Sepal Width -->
                        <div class="p-3.5 bg-white/80 dark:bg-slate-900/40 rounded-2xl border border-amber-100 dark:border-slate-800 hover:border-amber-300 dark:hover:border-indigo-800 transition">
                            <div class="grid grid-cols-12 gap-2 items-center">
                                <span class="col-span-4 text-xs text-slate-700 dark:text-slate-300 font-medium">Đài hoa (Rộng)</span>
                                <div class="col-span-8 flex items-center gap-2">
                                    <button type="button" onclick="adjustValue('sepal_width', -0.1)" class="w-7 h-7 flex-none bg-amber-100 dark:bg-slate-800 hover:bg-amber-200 dark:hover:bg-slate-700 rounded-lg text-xs font-bold text-amber-900 dark:text-slate-300 transition">-</button>
                                    <input type="range" min="2.0" max="4.5" step="0.1" id="sepal_width" value="3.5" oninput="updateUI()" class="w-full accent-amber-500 dark:accent-indigo-400 bg-amber-100 dark:bg-slate-800">
                                    <button type="button" onclick="adjustValue('sepal_width', 0.1)" class="w-7 h-7 flex-none bg-amber-100 dark:bg-slate-800 hover:bg-amber-200 dark:hover:bg-slate-700 rounded-lg text-xs font-bold text-amber-900 dark:text-slate-300 transition">+</button>
                                    <span id="sw_val" class="w-16 flex-none text-right font-mono text-xs font-bold text-amber-700 dark:text-indigo-400 bg-amber-50 dark:bg-slate-900/80 px-2 py-1 rounded-md border border-amber-200/50 dark:border-indigo-900/50">3.5 cm</span>
                                </div>
                            </div>
                        </div>

                        <!-- Petal Length -->
                        <div class="p-3.5 bg-white/80 dark:bg-slate-900/40 rounded-2xl border border-amber-100 dark:border-slate-800 hover:border-amber-300 dark:hover:border-indigo-800 transition">
                            <div class="grid grid-cols-12 gap-2 items-center">
                                <span class="col-span-4 text-xs text-slate-700 dark:text-slate-300 font-medium">Cánh hoa (Dài)</span>
                                <div class="col-span-8 flex items-center gap-2">
                                    <button type="button" onclick="adjustValue('petal_length', -0.1)" class="w-7 h-7 flex-none bg-amber-100 dark:bg-slate-800 hover:bg-amber-200 dark:hover:bg-slate-700 rounded-lg text-xs font-bold text-amber-900 dark:text-slate-300 transition">-</button>
                                    <input type="range" min="1.0" max="7.0" step="0.1" id="petal_length" value="1.4" oninput="updateUI()" class="w-full accent-amber-500 dark:accent-indigo-400 bg-amber-100 dark:bg-slate-800">
                                    <button type="button" onclick="adjustValue('petal_length', 0.1)" class="w-7 h-7 flex-none bg-amber-100 dark:bg-slate-800 hover:bg-amber-200 dark:hover:bg-slate-700 rounded-lg text-xs font-bold text-amber-900 dark:text-slate-300 transition">+</button>
                                    <span id="pl_val" class="w-16 flex-none text-right font-mono text-xs font-bold text-amber-700 dark:text-indigo-400 bg-amber-50 dark:bg-slate-900/80 px-2 py-1 rounded-md border border-amber-200/50 dark:border-indigo-900/50">1.4 cm</span>
                                </div>
                            </div>
                        </div>

                        <!-- Petal Width -->
                        <div class="p-3.5 bg-white/80 dark:bg-slate-900/40 rounded-2xl border border-amber-100 dark:border-slate-800 hover:border-amber-300 dark:hover:border-indigo-800 transition">
                            <div class="grid grid-cols-12 gap-2 items-center">
                                <span class="col-span-4 text-xs text-slate-700 dark:text-slate-300 font-medium">Cánh hoa (Rộng)</span>
                                <div class="col-span-8 flex items-center gap-2">
                                    <button type="button" onclick="adjustValue('petal_width', -0.1)" class="w-7 h-7 flex-none bg-amber-100 dark:bg-slate-800 hover:bg-amber-200 dark:hover:bg-slate-700 rounded-lg text-xs font-bold text-amber-900 dark:text-slate-300 transition">-</button>
                                    <input type="range" min="0.1" max="2.5" step="0.1" id="petal_width" value="0.2" oninput="updateUI()" class="w-full accent-amber-500 dark:accent-indigo-400 bg-amber-100 dark:bg-slate-800">
                                    <button type="button" onclick="adjustValue('petal_width', 0.1)" class="w-7 h-7 flex-none bg-amber-100 dark:bg-slate-800 hover:bg-amber-200 dark:hover:bg-slate-700 rounded-lg text-xs font-bold text-amber-900 dark:text-slate-300 transition">+</button>
                                    <span id="pw_val" class="w-16 flex-none text-right font-mono text-xs font-bold text-amber-700 dark:text-indigo-400 bg-amber-50 dark:bg-slate-900/80 px-2 py-1 rounded-md border border-amber-200/50 dark:border-indigo-900/50">0.2 cm</span>
                                </div>
                            </div>
                        </div>

                        <!-- Submit Button -->
                        <button type="submit" class="w-full py-4 bg-amber-500 hover:bg-amber-600 dark:bg-indigo-600 dark:hover:bg-indigo-500 text-white font-bold rounded-2xl shadow-lg shadow-amber-500/25 dark:shadow-indigo-600/30 transition duration-300 active:scale-[0.99] flex items-center justify-center gap-2 tracking-wide mt-2">
                            <svg class="w-5 h-5 text-amber-100 dark:text-indigo-200" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
                            PHÂN TÍCH & DỰ ĐOÁN
                        </button>
                    </form>

                    <!-- REFERENCE TABLE -->
                    <div class="pt-4 border-t border-amber-200/60 dark:border-slate-800 space-y-3">
                        <div class="flex justify-between items-center">
                            <h3 class="text-xs font-semibold text-amber-900 dark:text-slate-300 uppercase tracking-wider flex items-center gap-2">
                                <span>📖</span> Bảng Chỉ Số Tham Chiếu Tự Nhiên
                            </h3>
                        </div>
                        <div class="overflow-x-auto">
                            <table class="w-full text-left text-xs font-mono">
                                <thead class="text-amber-900/50 dark:text-slate-400 border-b border-amber-200/60 dark:border-slate-800 uppercase text-[10px]">
                                    <tr>
                                        <th class="pb-2">Loài Hoa</th>
                                        <th class="pb-2">Đài (Dài)</th>
                                        <th class="pb-2">Đài (Rộng)</th>
                                        <th class="pb-2">Cánh (Dài)</th>
                                        <th class="pb-2">Cánh (Rộng)</th>
                                    </tr>
                                </thead>
                                <tbody class="divide-y divide-amber-100 dark:divide-slate-800/60 text-slate-700 dark:text-slate-300">
                                    <tr>
                                        <td class="py-2 font-bold text-purple-600 dark:text-purple-400">Setosa</td>
                                        <td>4.3 - 5.8 cm</td>
                                        <td>2.3 - 4.4 cm</td>
                                        <td>1.0 - 1.9 cm</td>
                                        <td>0.1 - 0.6 cm</td>
                                    </tr>
                                    <tr>
                                        <td class="py-2 font-bold text-orange-500 dark:text-orange-400">Versicolor</td>
                                        <td>4.9 - 7.0 cm</td>
                                        <td>2.0 - 3.4 cm</td>
                                        <td>3.0 - 5.1 cm</td>
                                        <td>1.0 - 1.8 cm</td>
                                    </tr>
                                    <tr>
                                        <td class="py-2 font-bold text-sky-600 dark:text-sky-400">Virginica</td>
                                        <td>4.9 - 7.9 cm</td>
                                        <td>2.2 - 3.8 cm</td>
                                        <td>4.5 - 6.9 cm</td>
                                        <td>1.4 - 2.5 cm</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- PREVIEW & RESULT CARD -->
                <div class="lg:col-span-5 space-y-6">
                    <!-- SVG Visualizer -->
                    <div class="nature-card p-5 rounded-3xl space-y-3 text-center relative overflow-hidden">
                        <div class="flex justify-between items-center text-xs text-amber-900/60 dark:text-slate-400">
                            <span>MÔ PHỎNG HÌNH DÁNG HẠT & CÁNH</span>
                            <span class="text-amber-700 dark:text-indigo-400 font-medium">Trực Quan</span>
                        </div>
                        <div class="h-32 bg-white/60 dark:bg-slate-950/50 rounded-2xl flex items-center justify-center relative border border-amber-100 dark:border-slate-800">
                            <svg id="flowerSvg" class="transition-all duration-500 filter drop-shadow-md" width="100" height="100" viewBox="-50 -50 100 100">
                                <ellipse id="svgSepal" cx="0" cy="0" rx="20" ry="40" fill="#f59e0b" opacity="0.35" stroke="#d97706" stroke-width="1"/>
                                <ellipse id="svgSepal2" cx="0" cy="0" rx="40" ry="20" fill="#f59e0b" opacity="0.35" stroke="#d97706" stroke-width="1"/>
                                <circle id="svgPetal" cx="0" cy="0" r="15" fill="#fbbf24" opacity="0.75" stroke="#f59e0b" stroke-width="1.5"/>
                            </svg>
                        </div>
                    </div>

                    <!-- Prediction Result Card -->
                    <div id="resultCard" class="nature-card p-6 rounded-3xl text-center space-y-4 transition-all duration-300">
                        <div class="space-y-1">
                            <div id="resultIcon" class="text-5xl mb-2 animate-bounce inline-block">🌅</div>
                            <h3 id="resultName" class="text-2xl font-bold font-serif text-amber-950 dark:text-slate-100">Sẵn Sàng Khám Phá</h3>
                            <p id="resultDesc" class="text-xs text-amber-900/70 dark:text-slate-400">Điều chỉnh thông số và bấm để xem kết quả</p>
                        </div>

                        <div id="careBox" class="hidden p-4 bg-white/80 dark:bg-slate-900/80 border border-amber-200 dark:border-indigo-900/40 rounded-2xl text-left text-xs text-slate-700 dark:text-slate-300 space-y-2">
                            <span class="font-semibold text-amber-800 dark:text-indigo-400 text-[11px] uppercase tracking-wider block border-b border-amber-100 dark:border-slate-800 pb-1">🌱 Đặc tính & Mẹo Chăm Sóc</span>
                            <p id="careTipsText" class="leading-relaxed"></p>
                        </div>

                        <!-- Confidence Bars -->
                        <div id="probBars" class="space-y-2.5 text-left hidden pt-3 border-t border-amber-200/60 dark:border-slate-800">
                            <span class="text-[10px] font-semibold text-amber-900/50 dark:text-slate-400 uppercase tracking-widest block">Mức độ tương thích</span>
                            
                            <div class="space-y-1">
                                <div class="flex justify-between text-xs"><span>Iris-setosa</span><span id="prob0" class="text-purple-600 dark:text-purple-400 font-bold">0%</span></div>
                                <div class="w-full bg-amber-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                                    <div id="bar0" class="bg-purple-500 h-full w-0 transition-all duration-500 rounded-full"></div>
                                </div>
                            </div>
                            <div class="space-y-1">
                                <div class="flex justify-between text-xs"><span>Iris-versicolor</span><span id="prob1" class="text-orange-500 dark:text-orange-400 font-bold">0%</span></div>
                                <div class="w-full bg-amber-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                                    <div id="bar1" class="bg-orange-400 h-full w-0 transition-all duration-500 rounded-full"></div>
                                </div>
                            </div>
                            <div class="space-y-1">
                                <div class="flex justify-between text-xs"><span>Iris-virginica</span><span id="prob2" class="text-sky-600 dark:text-sky-400 font-bold">0%</span></div>
                                <div class="w-full bg-amber-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                                    <div id="bar2" class="bg-sky-500 h-full w-0 transition-all duration-500 rounded-full"></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- HISTORY CARD -->
                    <div class="nature-card p-5 rounded-3xl space-y-3">
                        <div class="flex justify-between items-center border-b border-amber-200/60 dark:border-slate-800 pb-2">
                            <h3 class="text-xs font-semibold text-amber-900 dark:text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                                <span>📜</span> Lịch Sử Phân Tích
                            </h3>
                            <button type="button" onclick="clearHistory()" class="text-[11px] text-rose-500 hover:text-rose-600 dark:text-rose-400 font-medium px-2 py-0.5 rounded-lg hover:bg-rose-50 dark:hover:bg-rose-950/50 transition flex items-center gap-1">
                                🗑️ Xóa lịch sử
                            </button>
                        </div>
                        <div id="historyContainer" class="max-h-48 overflow-y-auto">
                            <table class="w-full text-left text-xs font-mono">
                                <thead class="text-amber-900/50 dark:text-slate-400 border-b border-amber-200/60 dark:border-slate-800 uppercase text-[10px] sticky top-0 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md">
                                    <tr>
                                        <th class="pb-2">Tên loài</th>
                                        <th class="pb-2">Đài (DxR)</th>
                                        <th class="pb-2">Cánh (DxR)</th>
                                        <th class="pb-2">Thời gian</th>
                                    </tr>
                                </thead>
                                <tbody id="historyBody" class="divide-y divide-amber-100 dark:divide-slate-800/60 text-slate-700 dark:text-slate-300">
                                    <tr>
                                        <td colspan="4" class="py-4 text-center text-slate-400 italic">Chưa có bản ghi phân tích nào.</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

            </div>
        </div>

        <script>
            // Khởi tạo các hạt nắng & ngôi sao tự động
            function initBackgrounds() {
                const sunParticles = document.getElementById('sunParticles');
                for (let i = 0; i < 25; i++) {
                    const p = document.createElement('div');
                    p.className = 'sun-particle';
                    const size = Math.random() * 6 + 3;
                    p.style.width = `${size}px`;
                    p.style.height = `${size}px`;
                    p.style.left = `${Math.random() * 100}%`;
                    p.style.bottom = `-10px`;
                    p.style.setProperty('--duration', `${Math.random() * 6 + 6}s`);
                    p.style.setProperty('--delay', `${Math.random() * 5}s`);
                    p.style.setProperty('--drift', `${Math.random() * 80 - 40}px`);
                    sunParticles.appendChild(p);
                }

                const starsList = document.getElementById('starsList');
                for (let i = 0; i < 60; i++) {
                    const s = document.createElement('div');
                    s.className = 'star';
                    const size = Math.random() * 3 + 1;
                    s.style.width = `${size}px`;
                    s.style.height = `${size}px`;
                    s.style.top = `${Math.random() * 100}%`;
                    s.style.left = `${Math.random() * 100}%`;
                    s.style.setProperty('--duration', `${Math.random() * 3 + 2}s`);
                    s.style.setProperty('--delay', `${Math.random() * 3}s`);
                    starsList.appendChild(s);
                }

                const shootingStarsList = document.getElementById('shootingStarsList');
                for (let i = 0; i < 4; i++) {
                    const ss = document.createElement('div');
                    ss.className = 'shooting-star';
                    ss.style.top = `${Math.random() * 50}%`;
                    ss.style.right = `${Math.random() * 30}%`;
                    ss.style.setProperty('--speed', `${Math.random() * 2 + 2.5}s`);
                    ss.style.setProperty('--delay', `${Math.random() * 6 + i * 2}s`);
                    shootingStarsList.appendChild(ss);
                }
            }

            let historyData = [];

            function toggleTheme() {
                const html = document.documentElement;
                const skyBadge = document.getElementById('skyBadge');
                const subHeadline = document.getElementById('subHeadline');
                const themeIcon = document.getElementById('themeIcon');
                const themeText = document.getElementById('themeText');

                if (html.classList.contains('dark')) {
                    html.classList.remove('dark');
                    skyBadge.innerHTML = '<span class="sun-icon-glow">🌅</span> Ánh bình minh';
                    subHeadline.innerText = 'Như tia nắng đầu ngày, mở ra thế giới rực rỡ của muôn sắc hoa';
                    themeIcon.innerHTML = '🌅';
                    themeIcon.className = 'sun-icon-glow';
                    themeText.innerText = 'Ban ngày';
                } else {
                    html.classList.add('dark');
                    skyBadge.innerHTML = '<span class="star-sparkle">✨</span> Bầu trời tinh tú';
                    subHeadline.innerText = 'Ánh sáng lung linh từ ngàn vì sao thắp sáng bầu trời đêm';
                    themeIcon.innerHTML = '✨';
                    themeIcon.className = 'star-sparkle';
                    themeText.innerText = 'Ban đêm';
                }
            }

            function adjustValue(id, delta) {
                const input = document.getElementById(id);
                let val = parseFloat(input.value) + delta;
                val = Math.max(parseFloat(input.min), Math.min(parseFloat(input.max), val));
                input.value = val.toFixed(1);
                updateUI();
            }

            function applyPreset(sl, sw, pl, pw) {
                document.getElementById('sepal_length').value = sl;
                document.getElementById('sepal_width').value = sw;
                document.getElementById('petal_length').value = pl;
                document.getElementById('petal_width').value = pw;
                updateUI();
                submitForm();
            }

            function updateUI() {
                const sl = parseFloat(document.getElementById('sepal_length').value);
                const sw = parseFloat(document.getElementById('sepal_width').value);
                const pl = parseFloat(document.getElementById('petal_length').value);
                const pw = parseFloat(document.getElementById('petal_width').value);

                document.getElementById('sl_val').innerText = sl.toFixed(1) + ' cm';
                document.getElementById('sw_val').innerText = sw.toFixed(1) + ' cm';
                document.getElementById('pl_val').innerText = pl.toFixed(1) + ' cm';
                document.getElementById('pw_val').innerText = pw.toFixed(1) + ' cm';

                // Cập nhật SVG hoa
                const svgSepal = document.getElementById('svgSepal');
                const svgSepal2 = document.getElementById('svgSepal2');
                const svgPetal = document.getElementById('svgPetal');

                svgSepal.setAttribute('ry', Math.min(48, sl * 6));
                svgSepal.setAttribute('rx', Math.min(30, sw * 5));
                svgSepal2.setAttribute('rx', Math.min(48, sl * 6));
                svgSepal2.setAttribute('ry', Math.min(30, sw * 5));
                svgPetal.setAttribute('r', Math.min(25, (pl + pw) * 3));
            }

            async function submitForm(e) {
                if (e) e.preventDefault();

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

                    // Hiển thị kết quả
                    document.getElementById('resultIcon').innerText = data.icon;
                    document.getElementById('resultName').innerText = data.prediction;
                    document.getElementById('resultName').style.color = data.color;
                    document.getElementById('resultDesc').innerText = data.desc;
                    document.getElementById('careTipsText').innerText = data.care_tips;
                    document.getElementById('careBox').classList.remove('hidden');

                    // Bar tỉ lệ
                    document.getElementById('probBars').classList.remove('hidden');
                    const p0 = (data.probabilities[0] * 100).toFixed(1);
                    const p1 = (data.probabilities[1] * 100).toFixed(1);
                    const p2 = (data.probabilities[2] * 100).toFixed(1);

                    document.getElementById('prob0').innerText = p0 + '%';
                    document.getElementById('prob1').innerText = p1 + '%';
                    document.getElementById('prob2').innerText = p2 + '%';

                    document.getElementById('bar0').style.width = p0 + '%';
                    document.getElementById('bar1').style.width = p1 + '%';
                    document.getElementById('bar2').style.width = p2 + '%';

                    // Bắn pháo hoa Confetti
                    confetti({
                        particleCount: 50,
                        spread: 60,
                        origin: { y: 0.8 }
                    });

                    // Thêm vào Lịch sử
                    const now = new Date();
                    const timeString = now.toLocaleTimeString('vi-VN');

                    historyData.unshift({
                        name: data.prediction,
                        icon: data.icon,
                        sepal: `${payload.sepal_length}x${payload.sepal_width}`,
                        petal: `${payload.petal_length}x${payload.petal_width}`,
                        time: timeString
                    });

                    renderHistory();

                } catch (err) {
                    console.error("Lỗi dự đoán:", err);
                }
            }

            function renderHistory() {
                const tbody = document.getElementById('historyBody');
                if (historyData.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="4" class="py-4 text-center text-slate-400 italic">Chưa có bản ghi phân tích nào.</td></tr>';
                    return;
                }

                tbody.innerHTML = historyData.map(item => `
                    <tr class="hover:bg-amber-50/50 dark:hover:bg-slate-800/40 transition-colors">
                        <td class="py-2 font-medium flex items-center gap-1"><span>${item.icon}</span> <span>${item.name}</span></td>
                        <td>${item.sepal}</td>
                        <td>${item.petal}</td>
                        <td class="text-amber-800/70 dark:text-slate-400 text-[11px]">${item.time}</td>
                    </tr>
                `).join('');
            }

            function clearHistory() {
                historyData = [];
                renderHistory();
            }

            window.onload = function() {
                initBackgrounds();
                updateUI();
            };
        </script>
    </body>
    </html>
    """
