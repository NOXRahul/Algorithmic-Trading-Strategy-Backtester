# 🚀 Algorithmic Trading Strategy Backtester

A full-stack quantitative trading backtesting framework built with:

- 🐍 FastAPI (Python Backend)
- ⚛️ React + Vite (Frontend)
- 📊 Recharts (Data Visualization)

---

## 📌 Overview

This project simulates algorithmic trading strategies on historical market data and evaluates performance using professional risk metrics.

It provides:

- Strategy simulation
- Trade execution engine
- Portfolio tracking
- Risk analytics
- Interactive dashboard visualization

The goal is to replicate how quantitative trading systems are researched and tested in real-world hedge funds and prop desks.

---

## ⚙️ Strategy Implemented

Current strategy includes:

- Moving Average Crossover (MA 20 / MA 50)
- RSI Confirmation Filter
- ATR-based Position Sizing
- Commission modeling
- Slippage simulation
- Risk-per-trade management

---

## 📊 Performance Metrics Calculated

The system computes:

- Final Equity
- Total Return
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Win Rate
- Profit Factor
- Exposure
- Annual Volatility
- Trade Distribution

---

## 🧠 System Architecture

React Frontend  
↓  
FastAPI Backend  
↓  
Backtest Engine  
↓  
Strategy Logic  
↓  
Broker Simulation  
↓  
Portfolio & Risk Layer  
↓  
Performance Metrics  
↓  
JSON API Response  
↓  
Interactive Dashboard  

---

## 📂 Project Structure


algorithmic-trading-strategy-backtester/
│
├── 📁 frontend/                      # React (Vite) Dashboard
│   │
│   ├── 📁 src/
│   │   ├── components/
│   │   │   └── Dashboard.jsx         # Main dashboard UI
│   │   │
│   │   ├── App.jsx                   # Root React component
│   │   ├── main.jsx                  # React entry point
│   │   └── index.css                 # Global styles
│   │
│   ├── package.json
│   └── vite.config.js
│
├── 📁 backend/                       # FastAPI Backend
│   │
│   ├── main.py                       # FastAPI entry point
│   │
│   ├── 📁 core/
│   │   ├── base.py                   # Strategy base class
│   │   ├── engine.py                 # Core backtesting engine
│   │   ├── broker.py                 # Trade execution simulator
│   │   ├── portfolio.py              # Portfolio & capital logic
│   │   ├── performance.py            # Performance metrics
│   │   ├── manager.py                # Strategy manager
│   │   └── loader.py                 # Historical data loader
│   │
│   ├── run_demo.py                   # Demo runner script
│   └── requirements.txt              # Python dependencies
│
├── .gitignore
└── README.md
## 🛠️ Tech Stack

### 🔹 Backend
- **Python 3.x**
- **FastAPI** – High-performance API framework
- **Pandas** – Data manipulation & analysis
- **NumPy** – Numerical computations
- **Uvicorn** – ASGI server

### 🔹 Frontend
- **React (Vite)** – Modern UI framework
- **JavaScript (ES6+)**
- **Recharts** – Financial data visualization
- **CSS3** – Styling & layout

### 🔹 Dev & Deployment
- **Git & GitHub** – Version control
- ## 👨‍💻 Author

**Rahul Kafle**  
Aspiring Quant Developer | Algorithmic Trading Systems Builder  

Passionate about building systematic trading engines, risk analytics tools, and performance-driven financial systems.

- GitHub: https://github.com/NOXRahul
- Email: rrkafle2@gmail.com
- **Vercel** – Frontend deployment
- **Render** – Backend deployment

