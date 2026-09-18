from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris BioLab AI Classifier")

# Load model SVM
try:
    model = joblib.load("svm_model.pkl")
except Exception:
    model = None

SPECIES_MAP = {
    0: {
        "name": "Iris-setosa",
        "icon": "🪻",
        "color": "#A855F7",
        "desc": "Cánh hoa ngắn, đài hoa rộng. Cấu trúc đơn giản, nhỏ gọn.",
        "care_tips": "🌱 **Thổ nhưỡng & Ánh sáng:** Ưa đất khô, tơi xốp và thoát nước tốt, không chịu được ngập úng. Phát triển tốt ở nơi nắng đầy đủ."
    },
    1: {
        "name": "Iris-versicolor",
        "icon": "🌺",
        "color": "#EC4899",
        "desc": "Kích thước trung bình, màu sắc sặc sỡ, phân nhánh tốt.",
        "care_tips": "💧 **Thổ nhưỡng & Ánh sáng:** Ưa đất giàu dinh dưỡng, tơi xốp, độ ẩm vừa phải. Thích ánh sáng nhẹ đến đầy đủ."
    },
    2: {
        "name": "Iris-virginica",
        "icon": "🪷",
        "color": "#3B82F6",
        "desc": "Cánh hoa dài & rộng, kích thước phát triển lớn nhất.",
        "care_tips": "☀️ **Thổ nhưỡng & Ánh sáng:** Ưa đất ẩm, nhiều mùn, chịu được ngập nước. Phát triển tốt ở nơi nhiều ánh sáng trực tiếp."
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
    <html lang="vi" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BIOLAB // Iris AI Diagnostics</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
        <script>
            tailwind.config = {
                darkMode: 'class',
                theme: {
                    extend: {
                        fontFamily: {
                            sans: ['Plus Jakarta Sans', 'sans-serif'],
                            mono: ['JetBrains Mono', 'monospace'],
                        }
                    }
                }
            }
        </script>
        <style>
            .bento-card {
                background: rgba(255, 255, 255, 0.85);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(0, 0, 0, 0.08);
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
            }
            .dark .bento-card {
                background: rgba(15, 23, 42, 0.75);
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: none;
            }
            .bento-card:hover {
                border-color: rgba(16, 185, 129, 0.35);
            }
            .glow-emerald {
                box-shadow: 0 0 30px -5px rgba(16, 185, 129, 0.15);
            }
            input[type=range] {
                height: 6px;
                border-radius: 9999px;
            }
        </style>
    </head>
    <body class="bg-slate-100 dark:bg-slate-950 text-slate-800 dark:text-slate-100 font-sans min-h-screen p-4 md:p-8 selection:bg-emerald-500 selection:text-slate-950 transition-colors duration-300">
        
        <!-- Background Decorative Elements -->
        <div class="fixed inset-0 pointer-events-none z-0 overflow-hidden">
            <div class="absolute -top-40 -left-40 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl"></div>
            <div class="absolute top-1/2 -right-40 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl"></div>
        </div>

        <div class="max-w-6xl mx-auto space-y-6 relative z-10">
            
            <!-- HEADER BAR -->
            <header class="bento-card p-5 rounded-2xl flex flex-wrap justify-between items-center gap-4">
                <div class="flex items-center gap-3.5">
                    <div class="p-2.5 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-emerald-500 dark:text-emerald-400">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
                    </div>
                    <div>
                        <div class="flex items-center gap-2">
                            <h1 class="text-lg font-extrabold tracking-tight text-slate-900 dark:text-white font-mono">BIOLAB // IRIS-AI</h1>
                            <span class="px-2 py-0.5 text-[10px] font-mono uppercase bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30 rounded-full">v2.4 Online</span>
                        </div>
                        <p class="text-xs text-slate-500 dark:text-slate-400">Hệ thống phân tích & nhận diện phân loại mẫu sinh học</p>
                    </div>
                </div>

                <!-- CONTROLS & PRESETS -->
                <div class="flex items-center gap-3">
                    <!-- PRESET BUTTONS -->
                    <div class="flex items-center gap-1.5 bg-slate-200/80 dark:bg-slate-900/80 p-1.5 rounded-xl border border-slate-300 dark:border-slate-800">
                        <span class="text-[11px] font-mono text-slate-500 dark:text-slate-400 px-2 uppercase hidden sm:inline">Mẫu:</span>
                        <button onclick="applyPreset(5.1, 3.5, 1.4, 0.2)" class="px-2.5 py-1 hover:bg-slate-300 dark:hover:bg-slate-800 text-purple-600 dark:text-purple-400 text-xs rounded-lg font-mono transition">Setosa</button>
                        <button onclick="applyPreset(5.2, 3.6, 4.2, 1.4)" class="px-2.5 py-1 hover:bg-slate-300 dark:hover:bg-slate-800 text-pink-600 dark:text-pink-400 text-xs rounded-lg font-mono transition">Versicolor</button>
                        <button onclick="applyPreset(6.5, 3.0, 5.5, 2.0)" class="px-2.5 py-1 hover:bg-slate-300 dark:hover:bg-slate-800 text-blue-600 dark:text-blue-400 text-xs rounded-lg font-mono transition">Virginica</button>
                    </div>

                    <!-- THEME TOGGLE BUTTON -->
                    <button onclick="toggleTheme()" id="themeBtn" class="px-3 py-2 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-xl border border-slate-300 dark:border-slate-700 text-xs font-mono transition flex items-center gap-1.5 shadow-sm">
                        <span id="themeIcon">☀️</span>
                        <span id="themeText" class="hidden md:inline">Sáng</span>
                    </button>
                </div>
            </header>

            <!-- MAIN BENTO GRID -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
                
                <!-- INPUT PARAMETERS CARD (LEFT 7 COLS) -->
                <div class="lg:col-span-7 bento-card p-6 rounded-3xl space-y-6">
                    <div class="flex justify-between items-center border-b border-slate-200 dark:border-slate-800 pb-4">
                        <h2 class="text-sm font-bold font-mono text-emerald-600 dark:text-emerald-400 uppercase tracking-wider flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Thông Số Trích Xuất Morphology
                        </h2>
                        <span class="text-xs text-slate-500 dark:text-slate-400 font-mono">Đơn vị: Centimeter (cm)</span>
                    </div>

                    <form id="irisForm" class="space-y-5">
                        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            
                            <!-- Sepal Length -->
                            <div class="p-4 bg-slate-100/80 dark:bg-slate-900/60 rounded-2xl border border-slate-200 dark:border-slate-800/80 hover:border-slate-400 dark:hover:border-slate-700 transition space-y-2">
                                <div class="flex justify-between items-center">
                                    <span class="text-xs text-slate-600 dark:text-slate-400 font-medium">Chiều dài đài (Sepal L)</span>
                                    <span id="sl_val" class="font-mono text-sm font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-md">5.1 cm</span>
                                </div>
                                <div class="flex items-center gap-2 pt-1">
                                    <button type="button" onclick="adjustValue('sepal_length', -0.1)" class="w-7 h-7 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg text-xs font-mono font-bold">-</button>
                                    <input type="range" min="4.0" max="8.0" step="0.1" id="sepal_length" value="5.1" oninput="updateUI()" class="w-full accent-emerald-500 bg-slate-300 dark:bg-slate-800">
                                    <button type="button" onclick="adjustValue('sepal_length', 0.1)" class="w-7 h-7 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg text-xs font-mono font-bold">+</button>
                                </div>
                            </div>

                            <!-- Sepal Width -->
                            <div class="p-4 bg-slate-100/80 dark:bg-slate-900/60 rounded-2xl border border-slate-200 dark:border-slate-800/80 hover:border-slate-400 dark:hover:border-slate-700 transition space-y-2">
                                <div class="flex justify-between items-center">
                                    <span class="text-xs text-slate-600 dark:text-slate-400 font-medium">Chiều rộng đài (Sepal W)</span>
                                    <span id="sw_val" class="font-mono text-sm font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-md">3.5 cm</span>
                                </div>
                                <div class="flex items-center gap-2 pt-1">
                                    <button type="button" onclick="adjustValue('sepal_width', -0.1)" class="w-7 h-7 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg text-xs font-mono font-bold">-</button>
                                    <input type="range" min="2.0" max="4.5" step="0.1" id="sepal_width" value="3.5" oninput="updateUI()" class="w-full accent-emerald-500 bg-slate-300 dark:bg-slate-800">
                                    <button type="button" onclick="adjustValue('sepal_width', 0.1)" class="w-7 h-7 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg text-xs font-mono font-bold">+</button>
                                </div>
                            </div>

                            <!-- Petal Length -->
                            <div class="p-4 bg-slate-100/80 dark:bg-slate-900/60 rounded-2xl border border-slate-200 dark:border-slate-800/80 hover:border-slate-400 dark:hover:border-slate-700 transition space-y-2">
                                <div class="flex justify-between items-center">
                                    <span class="text-xs text-slate-600 dark:text-slate-400 font-medium">Chiều dài cánh (Petal L)</span>
                                    <span id="pl_val" class="font-mono text-sm font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-md">1.4 cm</span>
                                </div>
                                <div class="flex items-center gap-2 pt-1">
                                    <button type="button" onclick="adjustValue('petal_length', -0.1)" class="w-7 h-7 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg text-xs font-mono font-bold">-</button>
                                    <input type="range" min="1.0" max="7.0" step="0.1" id="petal_length" value="1.4" oninput="updateUI()" class="w-full accent-emerald-500 bg-slate-300 dark:bg-slate-800">
                                    <button type="button" onclick="adjustValue('petal_length', 0.1)" class="w-7 h-7 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg text-xs font-mono font-bold">+</button>
                                </div>
                            </div>

                            <!-- Petal Width -->
                            <div class="p-4 bg-slate-100/80 dark:bg-slate-900/60 rounded-2xl border border-slate-200 dark:border-slate-800/80 hover:border-slate-400 dark:hover:border-slate-700 transition space-y-2">
                                <div class="flex justify-between items-center">
                                    <span class="text-xs text-slate-600 dark:text-slate-400 font-medium">Chiều rộng cánh (Petal W)</span>
                                    <span id="pw_val" class="font-mono text-sm font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-md">0.2 cm</span>
                                </div>
                                <div class="flex items-center gap-2 pt-1">
                                    <button type="button" onclick="adjustValue('petal_width', -0.1)" class="w-7 h-7 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg text-xs font-mono font-bold">-</button>
                                    <input type="range" min="0.1" max="2.5" step="0.1" id="petal_width" value="0.2" oninput="updateUI()" class="w-full accent-emerald-500 bg-slate-300 dark:bg-slate-800">
                                    <button type="button" onclick="adjustValue('petal_width', 0.1)" class="w-7 h-7 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg text-xs font-mono font-bold">+</button>
                                </div>
                            </div>

                        </div>

                        <!-- ANOMALY WARNING BOX -->
                        <div id="warningBox" class="hidden p-3.5 bg-amber-500/10 border border-amber-500/30 rounded-xl text-amber-700 dark:text-amber-300 text-xs flex items-start gap-2.5">
                            <span class="text-base">⚡</span>
                            <div>
                                <span class="font-bold block font-mono uppercase text-[11px] text-amber-800 dark:text-amber-400">Anomaly Alert: Bất thường tỷ lệ sinh học</span>
                                <span id="warningText" class="text-slate-700 dark:text-slate-300 leading-relaxed"></span>
                            </div>
                        </div>

                        <!-- SUBMIT BUTTON -->
                        <button type="submit" class="w-full py-4 bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 text-slate-950 font-extrabold font-mono rounded-2xl shadow-lg shadow-emerald-500/20 transition duration-200 active:scale-[0.99] flex items-center justify-center gap-2">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                            KÍCH HOẠT PHÂN TÍCH SUY LUẬN AI
                        </button>
                    </form>
                </div>

                <!-- RIGHT BENTO COLUMN: SVG + RESULTS (RIGHT 5 COLS) -->
                <div class="lg:col-span-5 space-y-6">
                    
                    <!-- SVG BIOMETRIC CANVAS -->
                    <div class="bento-card p-5 rounded-3xl space-y-3 text-center relative overflow-hidden">
                        <div class="flex justify-between items-center text-xs font-mono text-slate-500 dark:text-slate-400">
                            <span>GEO-VECTOR MONITOR</span>
                            <span class="text-emerald-600 dark:text-emerald-400">LIVE RENDER</span>
                        </div>
                        <div class="h-32 bg-slate-200/50 dark:bg-slate-950/60 rounded-2xl flex items-center justify-center relative border border-slate-300 dark:border-slate-800/80">
                            <svg id="flowerSvg" class="transition-all duration-300 filter drop-shadow-[0_0_12px_rgba(16,185,129,0.3)]" width="100" height="100" viewBox="-50 -50 100 100">
                                <ellipse id="svgSepal" cx="0" cy="0" rx="20" ry="40" fill="#10B981" opacity="0.35" stroke="#10B981" stroke-width="1"/>
                                <ellipse id="svgSepal2" cx="0" cy="0" rx="40" ry="20" fill="#10B981" opacity="0.35" stroke="#10B981" stroke-width="1"/>
                                <circle id="svgPetal" cx="0" cy="0" r="15" fill="#EC4899" opacity="0.6" stroke="#EC4899" stroke-width="1.5"/>
                            </svg>
                        </div>
                    </div>

                    <!-- AI INFERENCE RESULT CARD -->
                    <div id="resultCard" class="bento-card p-6 rounded-3xl text-center space-y-4 glow-emerald transition-all duration-300">
                        <div class="space-y-1">
                            <div id="resultIcon" class="text-5xl mb-2 animate-bounce inline-block">❓</div>
                            <h3 id="resultName" class="text-2xl font-extrabold tracking-tight">Đang Chờ Dữ Liệu...</h3>
                            <p id="resultDesc" class="text-xs text-slate-500 dark:text-slate-400">Nhập thông số và bấm Kích Hoạt Phân Tích</p>
                        </div>

                        <!-- CARE TIPS -->
                        <div id="careBox" class="hidden p-3.5 bg-slate-200/60 dark:bg-slate-900/90 border border-emerald-500/30 rounded-2xl text-left text-xs text-slate-700 dark:text-slate-300 space-y-1.5">
                            <span class="font-mono font-bold text-emerald-600 dark:text-emerald-400 text-[11px] uppercase tracking-wider block border-b border-slate-300 dark:border-slate-800 pb-1">🪴 Đặc tính & Mẹo Sinh Thái</span>
                            <p id="careTipsText" class="leading-relaxed"></p>
                        </div>

                        <!-- PROBABILITY METRICS -->
                        <div id="probBars" class="space-y-2.5 text-left hidden pt-3 border-t border-slate-200 dark:border-slate-800">
                            <span class="text-[10px] font-mono font-bold text-slate-500 dark:text-slate-400 uppercase tracking-widest block">Xác xuất tin cậy (Confidence Metrics)</span>
                            
                            <div class="space-y-1">
                                <div class="flex justify-between text-xs font-mono"><span>Setosa</span><span id="prob0" class="text-purple-600 dark:text-purple-400 font-bold">0%</span></div>
                                <div class="w-full bg-slate-200 dark:bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-300 dark:border-slate-800">
                                    <div id="bar0" class="bg-purple-500 h-full w-0 transition-all duration-500 rounded-full"></div>
                                </div>
                            </div>
                            <div class="space-y-1">
                                <div class="flex justify-between text-xs font-mono"><span>Versicolor</span><span id="prob1" class="text-pink-600 dark:text-pink-400 font-bold">0%</span></div>
                                <div class="w-full bg-slate-200 dark:bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-300 dark:border-slate-800">
                                    <div id="bar1" class="bg-pink-500 h-full w-0 transition-all duration-500 rounded-full"></div>
                                </div>
                            </div>
                            <div class="space-y-1">
                                <div class="flex justify-between text-xs font-mono"><span>Virginica</span><span id="prob2" class="text-blue-600 dark:text-blue-400 font-bold">0%</span></div>
                                <div class="w-full bg-slate-200 dark:bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-300 dark:border-slate-800">
                                    <div id="bar2" class="bg-blue-500 h-full w-0 transition-all duration-500 rounded-full"></div>
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
            </div>

            <!-- LOWER BENTO GRID: BENCHMARKS & LOGS -->
            <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
                
                <!-- DATASET BENCHMARK TABLE (6 COLS) -->
                <div class="lg:col-span-6 bento-card p-5 rounded-3xl space-y-4">
                    <div class="flex justify-between items-center border-b border-slate-200 dark:border-slate-800 pb-3">
                        <h3 class="text-xs font-bold font-mono text-slate-700 dark:text-slate-300 uppercase tracking-wider flex items-center gap-2">
                            <span>📊</span> Khoảng Chỉ Số Tham Chiếu Chuẩn (Iris Dataset)
                        </h3>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-xs font-mono">
                            <thead class="text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800 uppercase text-[10px]">
                                <tr>
                                    <th class="pb-2">Loài</th>
                                    <th class="pb-2">Đài (D)</th>
                                    <th class="pb-2">Đài (R)</th>
                                    <th class="pb-2">Cánh (D)</th>
                                    <th class="pb-2">Cánh (R)</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-200 dark:divide-slate-800/60 text-slate-700 dark:text-slate-300">
                                <tr>
                                    <td class="py-2.5 font-bold text-purple-600 dark:text-purple-400">Setosa</td>
                                    <td>4.3-5.8</td>
                                    <td>2.3-4.4</td>
                                    <td>1.0-1.9</td>
                                    <td>0.1-0.6</td>
                                </tr>
                                <tr>
                                    <td class="py-2.5 font-bold text-pink-600 dark:text-pink-400">Versicolor</td>
                                    <td>4.9-7.0</td>
                                    <td>2.0-3.4</td>
                                    <td>3.0-5.1</td>
                                    <td>1.0-1.8</td>
                                </tr>
                                <tr>
                                    <td class="py-2.5 font-bold text-blue-600 dark:text-blue-400">Virginica</td>
                                    <td>4.9-7.9</td>
                                    <td>2.2-3.8</td>
                                    <td>4.5-6.9</td>
                                    <td>1.4-2.5</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- SCAN HISTORY LOG (6 COLS) -->
                <div class="lg:col-span-6 bento-card p-5 rounded-3xl space-y-4">
                    <div class="flex justify-between items-center border-b border-slate-200 dark:border-slate-800 pb-3">
                        <h3 class="text-xs font-bold font-mono text-slate-700 dark:text-slate-300 uppercase tracking-wider flex items-center gap-2">
                            <span>📜</span> Nhật Ký Phân Tích Gần Đây (Scan Logs)
                        </h3>
                        <button onclick="clearHistory()" class="px-2.5 py-1 bg-slate-200 dark:bg-slate-900 hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 border border-slate-300 dark:border-slate-800 hover:border-rose-500/30 text-[11px] rounded-lg font-mono transition">
                            Xóa Lịch Sử
                        </button>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-xs font-mono">
                            <thead class="text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800 uppercase text-[10px]">
                                <tr>
                                    <th class="pb-2">Kết quả</th>
                                    <th class="pb-2">Đài (DxR)</th>
                                    <th class="pb-2">Cánh (DxR)</th>
                                    <th class="pb-2">Class ID</th>
                                </tr>
                            </thead>
                            <tbody id="historyTable" class="divide-y divide-slate-200 dark:divide-slate-800/60 text-slate-700 dark:text-slate-300">
                                <tr><td colspan="4" class="py-4 text-center text-slate-400 dark:text-slate-500 font-sans">Chưa có dữ liệu phân tích</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

            </div>

        </div>

        <script>
            let historyData = [];

            // DARK / LIGHT THEME TOGGLE
            function toggleTheme() {
                const html = document.documentElement;
                const icon = document.getElementById('themeIcon');
                const text = document.getElementById('themeText');
                
                if (html.classList.contains('dark')) {
                    html.classList.remove('dark');
                    icon.textContent = '🌙';
                    text.textContent = 'Tối';
                } else {
                    html.classList.add('dark');
                    icon.textContent = '☀️';
                    text.textContent = 'Sáng';
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

                // Cập nhật SVG Visualizer
                document.getElementById('svgSepal').setAttribute('ry', sl * 5.5);
                document.getElementById('svgSepal').setAttribute('rx', sw * 5.5);
                document.getElementById('svgSepal2').setAttribute('rx', sl * 5.5);
                document.getElementById('svgSepal2').setAttribute('ry', sw * 5.5);
                document.getElementById('svgPetal').setAttribute('r', (pl + pw) * 3.8);

                // Anomaly Detector
                const warnBox = document.getElementById('warningBox');
                let anomalies = [];

                if (pw > sw) anomalies.push("Rộng cánh lớn hơn Rộng đài.");
                if (sw > sl) anomalies.push("Rộng đài vượt quá Dài đài.");
                if (pl > sl * 1.15) anomalies.push("Dài cánh lớn bất thường so meo Dài đài.");

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

                // Hiển thị Mẹo chăm sóc
                if (data.care_tips) {
                    document.getElementById('careBox').classList.remove('hidden');
                    document.getElementById('careTipsText').innerHTML = data.care_tips.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                }

                // Cập nhật thanh xác suất
                document.getElementById('probBars').classList.remove('hidden');
                data.probabilities.forEach((p, idx) => {
                    const pct = (p * 100).toFixed(1) + '%';
                    document.getElementById(`prob${idx}`).textContent = pct;
                    document.getElementById(`bar${idx}`).style.width = pct;
                });

                // Confetti
                confetti({ particleCount: 40, spread: 50, origin: { y: 0.7 } });

                // Ghi lịch sử
                historyData.unshift({
                    name: data.prediction,
                    icon: data.icon,
                    sepal: `${payload.sepal_length}x${payload.sepal_width}`,
                    petal: `${payload.petal_length}x${payload.petal_width}`,
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
                    tbody.innerHTML = '<tr><td colspan="4" class="py-4 text-center text-slate-400 dark:text-slate-500 font-sans">Chưa có dữ liệu phân tích</td></tr>';
                    return;
                }
                tbody.innerHTML = historyData.map(item => `
                    <tr class="hover:bg-slate-200/50 dark:hover:bg-slate-900/50 transition">
                        <td class="py-2 font-bold">${item.icon} ${item.name}</td>
                        <td>${item.sepal}</td>
                        <td>${item.petal}</td>
                        <td><span class="px-2 py-0.5 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 rounded text-[10px]">#0${item.id}</span></td>
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
    
    # 1. Tính toán xác suất thô (Raw Probabilities)
    if model is not None:
        try:
            raw_probs = model.predict_proba(features)[0].tolist()
        except Exception:
            try:
                pred_class = int(model.predict(features)[0])
                raw_probs = [0.1, 0.1, 0.1]
                raw_probs[pred_class] = 0.8
            except Exception:
                raw_probs = [0.33, 0.33, 0.34]
    else:
        # Giả lập xác suất dựa theo đặc trưng nếu chưa load được model .pkl
        pl = data.petal_length
        if pl < 2.5:
            raw_probs = [0.70, 0.20, 0.10]
        elif pl < 4.8:
            raw_probs = [0.15, 0.60, 0.25]
        else:
            raw_probs = [0.10, 0.20, 0.70]

    # 2. Áp dụng Label Smoothing (mềm hóa xác suất) để kết quả KHÔNG BAO GIỜ bị 100%
    smoothing_factor = 0.25  # Tỷ lệ làm mềm
    num_classes = len(raw_probs)
    
    smoothed_probs = [
        (1 - smoothing_factor) * p + (smoothing_factor / num_classes)
        for p in raw_probs
    ]
    
    # Chuẩn hóa lại tổng = 1.0 và làm tròn 3 chữ số thập phân
    total = sum(smoothed_probs)
    probs = [round(p / total, 3) for p in smoothed_probs]

    # Lấy class_id có xác suất cao nhất sau khi làm mềm
    pred_id = int(np.argmax(probs))

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
