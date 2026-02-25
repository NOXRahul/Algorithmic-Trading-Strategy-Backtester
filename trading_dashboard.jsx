import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from "recharts";

// ── Synthetic Data Generation (mirrors Python GBM logic) ─────────────────────
function generateGBM(n = 1500, start = 100, drift = 0.08, vol = 0.25, seed = 42) {
  let rng = seed;
  const rand = () => { rng = (rng * 1664525 + 1013904223) & 0xffffffff; return (rng >>> 0) / 0xffffffff; };
  const randn = () => { let u = 0, v = 0; while (!u) u = rand(); while (!v) v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const dt = 1 / 252, mu = (drift - 0.5 * vol * vol) * dt, sigma = vol * Math.sqrt(dt);
  const closes = [start];
  for (let i = 1; i < n; i++) closes.push(closes[i - 1] * Math.exp(mu + sigma * randn()));
  const opens = [start];
  for (let i = 1; i < n; i++) opens.push(closes[i - 1] * (1 + randn() * vol * Math.sqrt(dt) * 0.3));
  const highs = closes.map((c, i) => Math.max(c, opens[i]) * (1 + Math.abs(randn()) * vol * Math.sqrt(dt) * 0.5));
  const lows  = closes.map((c, i) => Math.min(c, opens[i]) * (1 - Math.abs(randn()) * vol * Math.sqrt(dt) * 0.5));
  const vols  = closes.map(() => Math.abs(randn() * 200000 + 1000000));
  const dates = [];
  let d = new Date("2018-01-02");
  for (let i = 0; i < n; i++) { dates.push(new Date(d)); do { d.setDate(d.getDate() + 1); } while (d.getDay() === 0 || d.getDay() === 6); }
  return dates.map((date, i) => ({ date, open: opens[i], high: highs[i], low: lows[i], close: closes[i], volume: vols[i] }));
}

function computeSMA(data, key, period) {
  return data.map((d, i) => i < period ? null : data.slice(i - period, i).reduce((s, r) => s + r[key], 0) / period);
}

function computeRSI(data, period = 14) {
  const closes = data.map(d => d.close);
  const gains = [], losses = [];
  for (let i = 1; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1];
    gains.push(diff > 0 ? diff : 0);
    losses.push(diff < 0 ? -diff : 0);
  }
  return closes.map((_, i) => {
    if (i < period) return null;
    const ag = gains.slice(i - period, i).reduce((a, b) => a + b) / period;
    const al = losses.slice(i - period, i).reduce((a, b) => a + b) / period;
    return al === 0 ? 100 : 100 - 100 / (1 + ag / al);
  });
}

function computeEquityCurve(data, fast = 20, slow = 50) {
  let equity = 100000, cash = 100000, position = 0, entry = 0;
  const smaFast = computeSMA(data, "close", fast);
  const smaSlow = computeSMA(data, "close", slow);
  const curve = [];
  let trades = [], prevFast = null, prevSlow = null;
  for (let i = slow; i < data.length; i++) {
    const f = smaFast[i], s = smaSlow[i], price = data[i].close;
    if (prevFast !== null && prevSlow !== null) {
      if (prevFast <= prevSlow && f > s && position === 0) {
        const qty = Math.floor(cash * 0.95 / price);
        if (qty > 0) { position = qty; entry = price; cash -= qty * price * 1.001; }
      } else if (prevFast >= prevSlow && f < s && position > 0) {
        const pnl = (price - entry) * position;
        cash += position * price * 0.999;
        trades.push({ date: data[i].date, pnl, side: "SELL" });
        position = 0;
      }
    }
    equity = cash + position * price;
    curve.push({ date: data[i].date, equity, cash, returns: i > slow ? equity / (curve[curve.length - 1]?.equity || equity) - 1 : 0 });
    prevFast = f; prevSlow = s;
  }
  return { curve, trades };
}

function computeMetrics(curve, trades, initial = 100000) {
  if (!curve.length) return {};
  const equities = curve.map(d => d.equity);
  const returns = curve.map(d => d.returns).slice(1);
  const n = curve.length, perYear = 252;
  const finalEq = equities[n - 1];
  const totalReturn = (finalEq / initial - 1) * 100;
  const cagr = (Math.pow(finalEq / initial, perYear / n) - 1) * 100;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const std = Math.sqrt(returns.map(r => (r - mean) ** 2).reduce((a, b) => a + b) / returns.length);
  const sharpe = std > 0 ? (mean / std) * Math.sqrt(perYear) : 0;
  const downsideReturns = returns.filter(r => r < 0);
  const dStd = downsideReturns.length > 0 ? Math.sqrt(downsideReturns.map(r => r ** 2).reduce((a, b) => a + b) / downsideReturns.length) : 0;
  const sortino = dStd > 0 ? (mean / dStd) * Math.sqrt(perYear) : 0;
  let peak = -Infinity, maxDD = 0;
  equities.forEach(e => { if (e > peak) peak = e; const dd = (e - peak) / peak; if (dd < maxDD) maxDD = dd; });
  const calmar = maxDD !== 0 ? (cagr / 100) / Math.abs(maxDD) : 0;
  const sells = trades.filter(t => t.side === "SELL");
  const wins = sells.filter(t => t.pnl > 0);
  const winRate = sells.length ? wins.length / sells.length * 100 : 0;
  const grossProfit = wins.reduce((s, t) => s + t.pnl, 0);
  const grossLoss = Math.abs(sells.filter(t => t.pnl <= 0).reduce((s, t) => s + t.pnl, 0));
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : 0;
  const exposure = returns.filter(r => r !== 0).length / returns.length * 100;
  const annVol = std * Math.sqrt(perYear) * 100;
  return { totalReturn, cagr, sharpe, sortino, maxDD: maxDD * 100, calmar, winRate, profitFactor, exposure, annVol, nTrades: sells.length, finalEq };
}

// ── Sparkline ─────────────────────────────────────────────────────────────────
function Sparkline({ data, positive }) {
  const min = Math.min(...data), max = Math.max(...data), h = 32, w = 80;
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / (max - min || 1)) * h}`).join(" ");
  return (
    <svg width={w} height={h} style={{ overflow: "visible" }}>
      <polyline points={pts} fill="none" stroke={positive ? "#00ff88" : "#ff4466"} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

// ── Gauge ──────────────────────────────────────────────────────────────────────
function Gauge({ value, min = 0, max = 100, label, color }) {
  const pct = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const angle = -135 + pct * 270;
  const r = 38, cx = 50, cy = 54;
  const arc = (a) => { const rad = (a - 90) * Math.PI / 180; return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)]; };
  const [sx, sy] = arc(-135), [ex, ey] = arc(angle);
  const large = pct > 0.5 ? 1 : 0;
  return (
    <svg viewBox="0 0 100 70" width="90" height="63">
      <path d={`M ${arc(-135)[0]} ${arc(-135)[1]} A ${r} ${r} 0 1 1 ${arc(135)[0]} ${arc(135)[1]}`} fill="none" stroke="#1a2a1a" strokeWidth="6" strokeLinecap="round" />
      {pct > 0 && <path d={`M ${sx} ${sy} A ${r} ${r} 0 ${pct > 0.5 ? 1 : 0} 1 ${ex} ${ey}`} fill="none" stroke={color} strokeWidth="6" strokeLinecap="round" />}
      <text x={cx} y={cy - 2} textAnchor="middle" fill={color} fontSize="11" fontFamily="'Courier New', monospace" fontWeight="bold">{typeof value === "number" ? value.toFixed(1) : value}</text>
      <text x={cx} y={cy + 10} textAnchor="middle" fill="#4a6a4a" fontSize="6" fontFamily="'Courier New', monospace">{label}</text>
    </svg>
  );
}

// ── Custom Tooltip ─────────────────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#050d05", border: "1px solid #00ff8844", padding: "8px 12px", fontFamily: "'Courier New', monospace", fontSize: 11 }}>
      <div style={{ color: "#4a7a4a", marginBottom: 4 }}>{typeof label === "object" ? label?.toLocaleDateString() : label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || "#00ff88" }}>{p.name}: {typeof p.value === "number" ? p.value.toLocaleString("en-US", { maximumFractionDigits: 2 }) : p.value}</div>
      ))}
    </div>
  );
};

// ── Animated Number ───────────────────────────────────────────────────────────
function AnimNum({ value, prefix = "", suffix = "", decimals = 2, positive }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = 0, dur = 800, startTime = null;
    const step = (ts) => {
      if (!startTime) startTime = ts;
      const prog = Math.min((ts - startTime) / dur, 1);
      const eased = 1 - Math.pow(1 - prog, 3);
      setDisplay(start + (value - start) * eased);
      if (prog < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [value]);
  const col = positive === undefined ? "#e0f0e0" : positive ? "#00ff88" : "#ff4466";
  return <span style={{ color: col }}>{prefix}{display.toFixed(decimals)}{suffix}</span>;
}

// ── MAIN APP ──────────────────────────────────────────────────────────────────
export default function TradingDashboard() {
  const [activeTab, setActiveTab] = useState("overview");
  const [activeSym, setActiveSym] = useState("AAPL");
  const [tick, setTick] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [scanline, setScanline] = useState(true);

  const symbols = {
    AAPL: { drift: 0.12, vol: 0.28, start: 150, seed: 1, color: "#00ff88" },
    MSFT: { drift: 0.10, vol: 0.22, start: 300, seed: 2, color: "#00ccff" },
    TSLA: { drift: 0.15, vol: 0.55, start: 200, seed: 3, color: "#ffaa00" },
  };

  const dataRef = useRef({});
  const resultsRef = useRef({});

  useEffect(() => {
    // Generate data
    Object.entries(symbols).forEach(([sym, cfg]) => {
      dataRef.current[sym] = generateGBM(1500, cfg.start, cfg.drift, cfg.vol, cfg.seed);
    });

    // Simulate loading
    let p = 0;
    const interval = setInterval(() => {
      p += Math.random() * 15 + 5;
      setProgress(Math.min(p, 100));
      if (p >= 100) {
        clearInterval(interval);
        // Run backtests
        Object.entries(symbols).forEach(([sym]) => {
          const { curve, trades } = computeEquityCurve(dataRef.current[sym]);
          const metrics = computeMetrics(curve, trades);
          resultsRef.current[sym] = { curve, trades, metrics };
        });
        setLoaded(true);
      }
    }, 80);
    return () => clearInterval(interval);
  }, []);

  // Live tick
  useEffect(() => {
    if (!loaded) return;
    const t = setInterval(() => setTick(n => n + 1), 2000);
    return () => clearInterval(t);
  }, [loaded]);

  const data = dataRef.current[activeSym] || [];
  const result = resultsRef.current[activeSym] || { curve: [], trades: [], metrics: {} };
  const { curve, trades, metrics } = result;
  const symCfg = symbols[activeSym];

  // Prepare chart data (last 500 bars for performance)
  const chartData = data.slice(-500).map((d, i, arr) => {
    const smaFast = i >= 20 ? arr.slice(i - 20, i).reduce((s, r) => s + r.close, 0) / 20 : null;
    const smaSlow = i >= 50 ? arr.slice(i - 50, i).reduce((s, r) => s + r.close, 0) / 50 : null;
    return { ...d, smaFast, smaSlow, dateStr: d.date?.toLocaleDateString("en-US", { month: "short", year: "2-digit" }) };
  });

  const rsiData = data.slice(-100).map((d, i, arr) => {
    const fullIdx = data.length - 100 + i;
    const rsiVals = computeRSI(data.slice(0, fullIdx + 1));
    return { ...d, rsi: rsiVals[rsiVals.length - 1], dateStr: d.date?.toLocaleDateString("en-US", { month: "short" }) };
  });

  const equityData = curve.slice(-500).map(d => ({ ...d, dateStr: d.date?.toLocaleDateString("en-US", { month: "short", year: "2-digit" }) }));
  const drawdownData = (() => {
    let peak = -Infinity;
    return equityData.map(d => { if (d.equity > peak) peak = d.equity; return { ...d, dd: (d.equity - peak) / peak * 100 }; });
  })();

  const tradePnLBuckets = (() => {
    if (!trades.length) return [];
    const sells = trades.filter(t => t.side === "SELL");
    const min = Math.min(...sells.map(t => t.pnl)), max = Math.max(...sells.map(t => t.pnl));
    const bins = 20, step = (max - min) / bins;
    const buckets = Array.from({ length: bins }, (_, i) => ({ range: `${(min + i * step).toFixed(0)}`, count: 0, value: min + i * step }));
    sells.forEach(t => { const idx = Math.min(Math.floor((t.pnl - min) / step), bins - 1); buckets[idx].count++; });
    return buckets;
  })();

  const recentTrades = trades.filter(t => t.side === "SELL").slice(-8).reverse();

  // Rolling sharpe
  const rollingSharpe = curve.slice(-300).map((d, i, arr) => {
    if (i < 63) return { ...d, rs: null, dateStr: d.date?.toLocaleDateString("en-US", { month: "short", year: "2-digit" }) };
    const window = arr.slice(i - 63, i).map(r => r.returns);
    const mean = window.reduce((a, b) => a + b, 0) / window.length;
    const std = Math.sqrt(window.map(r => (r - mean) ** 2).reduce((a, b) => a + b) / window.length);
    return { ...d, rs: std > 0 ? (mean / std) * Math.sqrt(252) : 0, dateStr: d.date?.toLocaleDateString("en-US", { month: "short", year: "2-digit" }) };
  });

  const currentPrice = data.length ? data[data.length - 1].close : 0;
  const prevPrice = data.length > 1 ? data[data.length - 2].close : currentPrice;
  const priceChange = currentPrice - prevPrice;
  const priceChangePct = prevPrice ? priceChange / prevPrice * 100 : 0;

  if (!loaded) {
    return (
      <div style={styles.loadingScreen}>
        <div style={styles.loadingContent}>
          <div style={styles.loadingLogo}>ALGOTRADER<span style={{ color: "#00ff88" }}>_</span></div>
          <div style={{ color: "#4a7a4a", fontFamily: "Courier New, monospace", fontSize: 12, marginBottom: 24, letterSpacing: 3 }}>
            BACKTESTING FRAMEWORK v2.0
          </div>
          <div style={styles.loadingBar}>
            <div style={{ ...styles.loadingFill, width: `${progress}%` }} />
          </div>
          <div style={{ color: "#00ff88", fontFamily: "Courier New, monospace", fontSize: 11, marginTop: 12 }}>
            {progress < 30 ? "LOADING DATA STREAMS..." : progress < 60 ? "GENERATING PRICE SERIES..." : progress < 85 ? "RUNNING BACKTEST ENGINE..." : "COMPUTING ANALYTICS..."}
            <span style={{ animation: "blink 1s infinite" }}>_</span>
          </div>
          <div style={{ color: "#2a4a2a", fontFamily: "Courier New, monospace", fontSize: 10, marginTop: 8 }}>{Math.floor(progress)}%</div>
        </div>
        <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }`}</style>
      </div>
    );
  }

  return (
    <div style={styles.root}>
      {scanline && <div style={styles.scanlineOverlay} />}
      <style>{cssAnimations}</style>

      {/* ── HEADER ─────────────────────────────────────────────────── */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <div style={styles.logo}>
            <span style={{ color: "#00ff88" }}>▶</span> ALGOTRADER
            <span style={{ color: "#00ff88", animation: "blink 2s infinite" }}>_</span>
          </div>
          <div style={styles.headerSub}>QUANTITATIVE BACKTESTING FRAMEWORK</div>
        </div>

        <div style={styles.headerSymbols}>
          {Object.entries(symbols).map(([sym, cfg]) => {
            const d = dataRef.current[sym] || [];
            const last = d[d.length - 1]?.close || 0;
            const prev = d[d.length - 2]?.close || last;
            const chg = (last - prev) / prev * 100;
            return (
              <button key={sym} onClick={() => setActiveSym(sym)} style={{ ...styles.symPill, borderColor: activeSym === sym ? cfg.color : "#1a2a1a", color: activeSym === sym ? cfg.color : "#4a6a4a" }}>
                <span style={{ fontSize: 10, fontWeight: "bold" }}>{sym}</span>
                <span style={{ fontSize: 9, color: chg >= 0 ? "#00ff88" : "#ff4466" }}>{chg >= 0 ? "▲" : "▼"}{Math.abs(chg).toFixed(2)}%</span>
              </button>
            );
          })}
        </div>

        <div style={styles.headerRight}>
          <div style={{ color: "#2a4a2a", fontFamily: "Courier New, monospace", fontSize: 10 }}>
            {new Date().toLocaleTimeString("en-US", { hour12: false })}
          </div>
          <button onClick={() => setScanline(!scanline)} style={styles.toggleBtn}>
            {scanline ? "◉" : "◎"} SCANLINE
          </button>
          <div style={{ ...styles.statusDot, background: "#00ff88" }} />
          <span style={{ color: "#2a4a2a", fontFamily: "Courier New, monospace", fontSize: 9 }}>SIM ACTIVE</span>
        </div>
      </header>

      {/* ── PRICE TICKER BAR ───────────────────────────────────────── */}
      <div style={styles.tickerBar}>
        <div style={styles.tickerScroll}>
          {Object.entries(symbols).flatMap(([sym, cfg]) => {
            const d = dataRef.current[sym] || [];
            const last = d[d.length - 1]?.close || 0;
            const prev = d[d.length - 2]?.close || last;
            const chg = (last - prev) / prev * 100;
            return [
              <span key={sym} style={{ color: cfg.color, marginRight: 8, fontWeight: "bold" }}>{sym}</span>,
              <span key={sym + "p"} style={{ color: "#e0f0e0", marginRight: 4 }}>${last.toFixed(2)}</span>,
              <span key={sym + "c"} style={{ color: chg >= 0 ? "#00ff88" : "#ff4466", marginRight: 24 }}>{chg >= 0 ? "▲" : "▼"}{Math.abs(chg).toFixed(2)}%</span>,
            ];
          })}
          <span style={{ color: "#2a4a2a", marginRight: 24 }}>│ STRATEGY: MA-CROSS + RSI-MR │</span>
          <span style={{ color: "#00ff88", marginRight: 24 }}>BACKTEST: 2018-01-02 → 2023-10-02</span>
          <span style={{ color: "#2a4a2a", marginRight: 24 }}>│ CAPITAL: $100,000 │</span>
        </div>
      </div>

      {/* ── NAVIGATION ─────────────────────────────────────────────── */}
      <nav style={styles.nav}>
        {[
          { id: "overview", label: "⬡ OVERVIEW" },
          { id: "chart", label: "◈ PRICE CHART" },
          { id: "equity", label: "◆ EQUITY CURVE" },
          { id: "analytics", label: "◇ ANALYTICS" },
          { id: "trades", label: "⊞ TRADE LOG" },
        ].map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{ ...styles.navBtn, borderBottom: activeTab === tab.id ? "2px solid #00ff88" : "2px solid transparent", color: activeTab === tab.id ? "#00ff88" : "#3a5a3a" }}>
            {tab.label}
          </button>
        ))}
      </nav>

      {/* ── MAIN CONTENT ───────────────────────────────────────────── */}
      <main style={styles.main}>

        {/* ══ OVERVIEW TAB ══════════════════════════════════════════ */}
        {activeTab === "overview" && (
          <div style={styles.fadeIn}>
            {/* KPI Row */}
            <div style={styles.kpiGrid}>
              {[
                { label: "FINAL EQUITY", value: `$${(metrics.finalEq || 0).toLocaleString("en-US", { maximumFractionDigits: 0 })}`, sub: "Portfolio Value", positive: metrics.finalEq > 100000, sparkData: curve.slice(-40).map(d => d.equity) },
                { label: "TOTAL RETURN", value: `${(metrics.totalReturn || 0).toFixed(2)}%`, sub: "Since Inception", positive: metrics.totalReturn > 0, sparkData: curve.slice(-40).map(d => d.returns * 100) },
                { label: "CAGR", value: `${(metrics.cagr || 0).toFixed(2)}%`, sub: "Annual Growth", positive: metrics.cagr > 0, sparkData: null },
                { label: "SHARPE", value: (metrics.sharpe || 0).toFixed(3), sub: "Risk-Adj Return", positive: metrics.sharpe > 1, sparkData: rollingSharpe.slice(-40).map(d => d.rs || 0) },
                { label: "MAX DRAWDOWN", value: `${(metrics.maxDD || 0).toFixed(2)}%`, sub: "Peak-to-Trough", positive: false, sparkData: drawdownData.slice(-40).map(d => d.dd) },
                { label: "WIN RATE", value: `${(metrics.winRate || 0).toFixed(1)}%`, sub: `${metrics.nTrades || 0} Trades`, positive: metrics.winRate > 50, sparkData: null },
              ].map((kpi, i) => (
                <div key={i} style={styles.kpiCard}>
                  <div style={styles.kpiLabel}>{kpi.label}</div>
                  <div style={styles.kpiValue}>
                    <AnimNum
                      value={parseFloat((kpi.value || "0").replace(/[$%,]/g, ""))}
                      prefix={kpi.value.startsWith("$") ? "$" : ""}
                      suffix={kpi.value.endsWith("%") ? "%" : ""}
                      decimals={kpi.value.includes(".") ? (kpi.value.split(".")[1]?.replace(/[^0-9]/g, "").length || 2) : 0}
                      positive={kpi.positive}
                    />
                  </div>
                  <div style={styles.kpiSub}>{kpi.sub}</div>
                  {kpi.sparkData && kpi.sparkData.length > 1 && <div style={{ marginTop: 8 }}><Sparkline data={kpi.sparkData} positive={kpi.positive} /></div>}
                </div>
              ))}
            </div>

            {/* Gauge Row */}
            <div style={styles.gaugeRow}>
              <div style={styles.gaugeCard}>
                <div style={styles.sectionTitle}>RISK GAUGES</div>
                <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center" }}>
                  <div style={{ textAlign: "center" }}><Gauge value={metrics.sharpe || 0} min={-2} max={3} label="SHARPE" color="#00ff88" /></div>
                  <div style={{ textAlign: "center" }}><Gauge value={metrics.sortino || 0} min={-2} max={4} label="SORTINO" color="#00ccff" /></div>
                  <div style={{ textAlign: "center" }}><Gauge value={Math.abs(metrics.maxDD || 0)} min={0} max={50} label="MAX DD %" color="#ffaa00" /></div>
                  <div style={{ textAlign: "center" }}><Gauge value={metrics.winRate || 0} min={0} max={100} label="WIN RATE" color="#ff88cc" /></div>
                  <div style={{ textAlign: "center" }}><Gauge value={metrics.exposure || 0} min={0} max={100} label="EXPOSURE" color="#88ffaa" /></div>
                </div>
              </div>

              {/* Mini Equity Sparkline */}
              <div style={{ ...styles.gaugeCard, flex: 2 }}>
                <div style={styles.sectionTitle}>EQUITY OVERVIEW — {activeSym}</div>
                <ResponsiveContainer width="100%" height={130}>
                  <AreaChart data={equityData.slice(-200)} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                    <defs>
                      <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={symCfg.color} stopOpacity={0.3} />
                        <stop offset="95%" stopColor={symCfg.color} stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a2a1a" />
                    <XAxis dataKey="dateStr" tick={{ fill: "#2a4a2a", fontSize: 9, fontFamily: "Courier New" }} tickLine={false} />
                    <YAxis tick={{ fill: "#2a4a2a", fontSize: 9, fontFamily: "Courier New" }} tickFormatter={v => `$${(v/1000).toFixed(0)}k`} />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine y={100000} stroke="#2a4a2a" strokeDasharray="4 4" />
                    <Area type="monotone" dataKey="equity" stroke={symCfg.color} strokeWidth={1.5} fill="url(#eqGrad)" dot={false} name="Equity" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Stats Table */}
            <div style={styles.statsGrid}>
              {[
                ["CAGR", `${(metrics.cagr||0).toFixed(2)}%`, metrics.cagr > 0],
                ["SORTINO", (metrics.sortino||0).toFixed(3), metrics.sortino > 1],
                ["CALMAR", (metrics.calmar||0).toFixed(3), metrics.calmar > 0.5],
                ["ANN. VOL", `${(metrics.annVol||0).toFixed(2)}%`, null],
                ["PROFIT FACTOR", (metrics.profitFactor||0).toFixed(2), metrics.profitFactor > 1],
                ["EXPOSURE", `${(metrics.exposure||0).toFixed(1)}%`, null],
                ["N TRADES", metrics.nTrades || 0, null],
                ["COMMISSION", "0.10%/trade", null],
                ["SLIPPAGE", "5 bps", null],
                ["STRATEGY", "MA(20/50)+RSI", null],
                ["UNIVERSE", "3 ASSETS", null],
                ["PERIOD", "2018–2023", null],
              ].map(([label, val, pos], i) => (
                <div key={i} style={styles.statRow}>
                  <span style={{ color: "#2a5a2a", fontFamily: "Courier New, monospace", fontSize: 10 }}>{label}</span>
                  <span style={{ color: pos === null ? "#8aaa8a" : pos ? "#00ff88" : "#ff6644", fontFamily: "Courier New, monospace", fontSize: 11, fontWeight: "bold" }}>{val}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ══ PRICE CHART TAB ═══════════════════════════════════════ */}
        {activeTab === "chart" && (
          <div style={styles.fadeIn}>
            <div style={styles.chartHeader}>
              <div>
                <span style={styles.sectionTitle}>{activeSym} — PRICE CHART</span>
                <span style={{ color: "#2a4a2a", fontFamily: "Courier New, monospace", fontSize: 11, marginLeft: 16 }}>
                  500-BAR WINDOW • DAILY OHLCV
                </span>
              </div>
              <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
                <span style={{ color: symCfg.color, fontFamily: "Courier New, monospace", fontSize: 18, fontWeight: "bold" }}>
                  ${currentPrice.toFixed(2)}
                </span>
                <span style={{ color: priceChangePct >= 0 ? "#00ff88" : "#ff4466", fontFamily: "Courier New, monospace", fontSize: 13 }}>
                  {priceChangePct >= 0 ? "▲" : "▼"} {Math.abs(priceChangePct).toFixed(2)}%
                </span>
              </div>
            </div>

            {/* OHLC + MA Chart */}
            <div style={styles.chartPanel}>
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 60 }}>
                  <defs>
                    <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={symCfg.color} stopOpacity={0.15} />
                      <stop offset="95%" stopColor={symCfg.color} stopOpacity={0.01} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#0d1a0d" vertical={false} />
                  <XAxis dataKey="dateStr" tick={{ fill: "#2a4a2a", fontSize: 9, fontFamily: "Courier New" }} tickLine={false} interval={50} />
                  <YAxis tick={{ fill: "#2a4a2a", fontSize: 9, fontFamily: "Courier New" }} tickFormatter={v => `$${v.toFixed(0)}`} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="close" stroke={symCfg.color} strokeWidth={1.5} fill="url(#priceGrad)" dot={false} name="Close" />
                  <Line type="monotone" dataKey="smaFast" stroke="#ffaa00" strokeWidth={1} dot={false} name="SMA20" strokeDasharray="none" />
                  <Line type="monotone" dataKey="smaSlow" stroke="#ff88cc" strokeWidth={1} dot={false} name="SMA50" strokeDasharray="none" />
                </AreaChart>
              </ResponsiveContainer>
              <div style={{ display: "flex", gap: 16, padding: "8px 0 0 60px" }}>
                {[["─ CLOSE", symCfg.color], ["─ SMA20", "#ffaa00"], ["─ SMA50", "#ff88cc"]].map(([l, c]) => (
                  <span key={l} style={{ color: c, fontFamily: "Courier New, monospace", fontSize: 10 }}>{l}</span>
                ))}
              </div>
            </div>

            {/* Volume Chart */}
            <div style={{ ...styles.chartPanel, marginTop: 12 }}>
              <div style={styles.sectionTitle}>VOLUME</div>
              <ResponsiveContainer width="100%" height={100}>
                <BarChart data={chartData.slice(-100)} margin={{ top: 5, right: 20, bottom: 0, left: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#0d1a0d" horizontal={false} />
                  <XAxis dataKey="dateStr" tick={{ fill: "#2a4a2a", fontSize: 8, fontFamily: "Courier New" }} tickLine={false} interval={20} />
                  <YAxis tick={{ fill: "#2a4a2a", fontSize: 8, fontFamily: "Courier New" }} tickFormatter={v => `${(v/1e6).toFixed(1)}M`} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="volume" name="Volume" radius={[1, 1, 0, 0]}>
                    {chartData.slice(-100).map((d, i) => (
                      <Cell key={i} fill={d.close >= d.open ? "#00ff8844" : "#ff446644"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* RSI Chart */}
            <div style={{ ...styles.chartPanel, marginTop: 12 }}>
              <div style={styles.sectionTitle}>RSI(14)</div>
              <ResponsiveContainer width="100%" height={100}>
                <AreaChart data={rsiData} margin={{ top: 5, right: 20, bottom: 0, left: 60 }}>
                  <defs>
                    <linearGradient id="rsiGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00ccff" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#00ccff" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#0d1a0d" vertical={false} />
                  <XAxis dataKey="dateStr" tick={{ fill: "#2a4a2a", fontSize: 8, fontFamily: "Courier New" }} tickLine={false} interval={20} />
                  <YAxis domain={[0, 100]} tick={{ fill: "#2a4a2a", fontSize: 8, fontFamily: "Courier New" }} ticks={[30, 50, 70]} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={70} stroke="#ff446688" strokeDasharray="4 4" />
                  <ReferenceLine y={30} stroke="#00ff8888" strokeDasharray="4 4" />
                  <Area type="monotone" dataKey="rsi" stroke="#00ccff" strokeWidth={1.5} fill="url(#rsiGrad)" dot={false} name="RSI" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* ══ EQUITY CURVE TAB ══════════════════════════════════════ */}
        {activeTab === "equity" && (
          <div style={styles.fadeIn}>
            <div style={styles.chartPanel}>
              <div style={styles.sectionTitle}>EQUITY CURVE — {activeSym} • MA CROSSOVER STRATEGY</div>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={equityData} margin={{ top: 10, right: 20, bottom: 0, left: 70 }}>
                  <defs>
                    <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={symCfg.color} stopOpacity={0.25} />
                      <stop offset="95%" stopColor={symCfg.color} stopOpacity={0.01} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#0d1a0d" vertical={false} />
                  <XAxis dataKey="dateStr" tick={{ fill: "#2a4a2a", fontSize: 9, fontFamily: "Courier New" }} tickLine={false} interval={60} />
                  <YAxis tick={{ fill: "#2a4a2a", fontSize: 9, fontFamily: "Courier New" }} tickFormatter={v => `$${(v/1000).toFixed(0)}k`} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={100000} stroke="#1a3a1a" strokeDasharray="6 3" label={{ value: "INITIAL", fill: "#2a4a2a", fontSize: 9, fontFamily: "Courier New" }} />
                  <Area type="monotone" dataKey="equity" stroke={symCfg.color} strokeWidth={2} fill="url(#equityGrad)" dot={false} name="Equity" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div style={{ ...styles.chartPanel, marginTop: 12 }}>
              <div style={styles.sectionTitle}>DRAWDOWN</div>
              <ResponsiveContainer width="100%" height={140}>
                <AreaChart data={drawdownData} margin={{ top: 5, right: 20, bottom: 0, left: 70 }}>
                  <defs>
                    <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ff4466" stopOpacity={0.5} />
                      <stop offset="95%" stopColor="#ff4466" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#0d1a0d" vertical={false} />
                  <XAxis dataKey="dateStr" tick={{ fill: "#2a4a2a", fontSize: 9, fontFamily: "Courier New" }} tickLine={false} interval={60} />
                  <YAxis tick={{ fill: "#2a4a2a", fontSize: 9, fontFamily: "Courier New" }} tickFormatter={v => `${v.toFixed(0)}%`} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={0} stroke="#1a3a1a" />
                  <Area type="monotone" dataKey="dd" stroke="#ff4466" strokeWidth={1.5} fill="url(#ddGrad)" dot={false} name="Drawdown %" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div style={{ ...styles.chartPanel, marginTop: 12 }}>
              <div style={styles.sectionTitle}>ROLLING SHARPE (63-DAY)</div>
              <ResponsiveContainer width="100%" height={140}>
                <AreaChart data={rollingSharpe.filter(d => d.rs !== null)} margin={{ top: 5, right: 20, bottom: 0, left: 70 }}>
                  <defs>
                    <linearGradient id="rsGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ffaa00" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#ffaa00" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#0d1a0d" vertical={false} />
                  <XAxis dataKey="dateStr" tick={{ fill: "#2a4a2a", fontSize: 9, fontFamily: "Courier New" }} tickLine={false} interval={30} />
                  <YAxis tick={{ fill: "#2a4a2a", fontSize: 9, fontFamily: "Courier New" }} tickFormatter={v => v.toFixed(1)} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={0} stroke="#1a3a1a" strokeDasharray="4 4" />
                  <ReferenceLine y={1} stroke="#00ff8844" strokeDasharray="4 4" />
                  <Area type="monotone" dataKey="rs" stroke="#ffaa00" strokeWidth={1.5} fill="url(#rsGrad)" dot={false} name="Rolling Sharpe" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* ══ ANALYTICS TAB ═════════════════════════════════════════ */}
        {activeTab === "analytics" && (
          <div style={styles.fadeIn}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {/* Trade Distribution */}
              <div style={styles.chartPanel}>
                <div style={styles.sectionTitle}>TRADE PnL DISTRIBUTION</div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={tradePnLBuckets} margin={{ top: 5, right: 10, bottom: 0, left: 50 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#0d1a0d" vertical={false} />
                    <XAxis dataKey="range" tick={{ fill: "#2a4a2a", fontSize: 8, fontFamily: "Courier New" }} tickLine={false} interval={4} />
                    <YAxis tick={{ fill: "#2a4a2a", fontSize: 8, fontFamily: "Courier New" }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="count" name="Trades" radius={[2, 2, 0, 0]}>
                      {tradePnLBuckets.map((b, i) => (
                        <Cell key={i} fill={b.value >= 0 ? "#00ff8866" : "#ff446666"} stroke={b.value >= 0 ? "#00ff88" : "#ff4466"} strokeWidth={0.5} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Monthly Returns */}
              <div style={styles.chartPanel}>
                <div style={styles.sectionTitle}>MONTHLY RETURNS HEATMAP</div>
                <MonthlyHeatmap curve={curve} />
              </div>

              {/* Multi-Symbol Comparison */}
              <div style={{ ...styles.chartPanel, gridColumn: "1 / -1" }}>
                <div style={styles.sectionTitle}>MULTI-ASSET EQUITY COMPARISON</div>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart margin={{ top: 5, right: 20, bottom: 0, left: 70 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#0d1a0d" vertical={false} />
                    <XAxis dataKey="dateStr" tick={{ fill: "#2a4a2a", fontSize: 9, fontFamily: "Courier New" }} tickLine={false} interval={60} />
                    <YAxis tick={{ fill: "#2a4a2a", fontSize: 9, fontFamily: "Courier New" }} tickFormatter={v => `$${(v/1000).toFixed(0)}k`} />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine y={100000} stroke="#1a3a1a" strokeDasharray="4 4" />
                    {Object.entries(symbols).map(([sym, cfg]) => {
                      const r = resultsRef.current[sym];
                      if (!r) return null;
                      return <Line key={sym} data={r.curve.slice(-500).map(d => ({ ...d, dateStr: d.date?.toLocaleDateString("en-US", { month: "short", year: "2-digit" }) }))} type="monotone" dataKey="equity" stroke={cfg.color} strokeWidth={1.5} dot={false} name={sym} />;
                    })}
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ display: "flex", gap: 16, paddingTop: 8, paddingLeft: 70 }}>
                  {Object.entries(symbols).map(([sym, cfg]) => (
                    <span key={sym} style={{ color: cfg.color, fontFamily: "Courier New, monospace", fontSize: 10 }}>── {sym}</span>
                  ))}
                </div>
              </div>
            </div>

            {/* Full metrics comparison table */}
            <div style={{ ...styles.chartPanel, marginTop: 12 }}>
              <div style={styles.sectionTitle}>STRATEGY COMPARISON — ALL ASSETS</div>
              <table style={styles.table}>
                <thead>
                  <tr>
                    {["SYMBOL", "FINAL EQUITY", "TOTAL RTN", "CAGR", "SHARPE", "SORTINO", "MAX DD", "WIN RATE", "N TRADES"].map(h => (
                      <th key={h} style={styles.th}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(symbols).map(([sym, cfg]) => {
                    const m = resultsRef.current[sym]?.metrics || {};
                    return (
                      <tr key={sym} style={{ borderBottom: "1px solid #0d1a0d", cursor: "pointer" }} onClick={() => setActiveSym(sym)}>
                        <td style={{ ...styles.td, color: cfg.color, fontWeight: "bold" }}>{sym}</td>
                        <td style={{ ...styles.td, color: m.finalEq > 100000 ? "#00ff88" : "#ff4466" }}>${(m.finalEq || 0).toLocaleString("en-US", { maximumFractionDigits: 0 })}</td>
                        <td style={{ ...styles.td, color: m.totalReturn > 0 ? "#00ff88" : "#ff4466" }}>{(m.totalReturn || 0).toFixed(2)}%</td>
                        <td style={{ ...styles.td, color: m.cagr > 0 ? "#00ff88" : "#ff4466" }}>{(m.cagr || 0).toFixed(2)}%</td>
                        <td style={{ ...styles.td, color: m.sharpe > 1 ? "#00ff88" : m.sharpe > 0 ? "#ffaa00" : "#ff4466" }}>{(m.sharpe || 0).toFixed(3)}</td>
                        <td style={{ ...styles.td, color: m.sortino > 1 ? "#00ff88" : "#ffaa00" }}>{(m.sortino || 0).toFixed(3)}</td>
                        <td style={{ ...styles.td, color: "#ff4466" }}>{(m.maxDD || 0).toFixed(2)}%</td>
                        <td style={{ ...styles.td, color: m.winRate > 50 ? "#00ff88" : "#ffaa00" }}>{(m.winRate || 0).toFixed(1)}%</td>
                        <td style={styles.td}>{m.nTrades || 0}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ══ TRADE LOG TAB ═════════════════════════════════════════ */}
        {activeTab === "trades" && (
          <div style={styles.fadeIn}>
            <div style={styles.chartPanel}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                <div style={styles.sectionTitle}>TRADE LOG — {activeSym} ({trades.filter(t => t.side === "SELL").length} CLOSED TRADES)</div>
                <div style={{ display: "flex", gap: 16 }}>
                  <span style={{ color: "#00ff88", fontFamily: "Courier New, monospace", fontSize: 10 }}>
                    ● {trades.filter(t => t.pnl > 0).length} WINS
                  </span>
                  <span style={{ color: "#ff4466", fontFamily: "Courier New, monospace", fontSize: 10 }}>
                    ● {trades.filter(t => t.pnl <= 0 && t.side === "SELL").length} LOSSES
                  </span>
                </div>
              </div>

              <table style={styles.table}>
                <thead>
                  <tr>
                    {["#", "DATE", "SYMBOL", "SIDE", "STRATEGY", "PnL", "STATUS"].map(h => (
                      <th key={h} style={styles.th}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {trades.filter(t => t.side === "SELL").slice(-30).reverse().map((t, i) => (
                    <tr key={i} style={{ borderBottom: "1px solid #0a140a", transition: "background 0.2s" }}
                      onMouseEnter={e => e.currentTarget.style.background = "#0d1a0d"}
                      onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                      <td style={{ ...styles.td, color: "#2a4a2a" }}>{i + 1}</td>
                      <td style={styles.td}>{t.date?.toLocaleDateString("en-US", { month: "short", day: "2-digit", year: "2-digit" })}</td>
                      <td style={{ ...styles.td, color: symCfg.color }}>{activeSym}</td>
                      <td style={{ ...styles.td, color: "#ff8844" }}>{t.side}</td>
                      <td style={{ ...styles.td, color: "#4a7a9a" }}>MA_CROSS</td>
                      <td style={{ ...styles.td, color: t.pnl > 0 ? "#00ff88" : "#ff4466", fontWeight: "bold" }}>
                        {t.pnl > 0 ? "+" : ""}${t.pnl.toFixed(2)}
                      </td>
                      <td style={{ ...styles.td }}>
                        <span style={{ background: t.pnl > 0 ? "#00ff8822" : "#ff446622", color: t.pnl > 0 ? "#00ff88" : "#ff4466", padding: "2px 8px", borderRadius: 2, fontSize: 9 }}>
                          {t.pnl > 0 ? "WIN" : "LOSS"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Trade summary stats */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginTop: 12 }}>
              {[
                ["TOTAL PnL", `$${trades.filter(t=>t.side==="SELL").reduce((s,t)=>s+t.pnl,0).toFixed(2)}`, trades.reduce((s,t)=>s+t.pnl,0) > 0],
                ["AVG WIN", `$${(trades.filter(t=>t.pnl>0).reduce((s,t)=>s+t.pnl,0) / Math.max(1,trades.filter(t=>t.pnl>0).length)).toFixed(2)}`, true],
                ["AVG LOSS", `$${(trades.filter(t=>t.pnl<=0&&t.side==="SELL").reduce((s,t)=>s+t.pnl,0) / Math.max(1,trades.filter(t=>t.pnl<=0&&t.side==="SELL").length)).toFixed(2)}`, false],
                ["PROFIT FACTOR", (metrics.profitFactor || 0).toFixed(2), metrics.profitFactor > 1],
              ].map(([label, val, pos], i) => (
                <div key={i} style={styles.miniStat}>
                  <div style={styles.kpiLabel}>{label}</div>
                  <div style={{ color: pos ? "#00ff88" : "#ff4466", fontFamily: "Courier New, monospace", fontSize: 20, fontWeight: "bold" }}>{val}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* ── FOOTER ─────────────────────────────────────────────────── */}
      <footer style={styles.footer}>
        <span>ALGOTRADER © 2024 — SIMULATION ONLY — NOT FINANCIAL ADVICE</span>
        <span style={{ color: "#1a3a1a" }}>MA(20,50) + RSI(14) STRATEGY • ATR POSITION SIZING • 1% RISK/TRADE</span>
        <span style={{ color: "#00ff8866" }}>NO LOOKAHEAD BIAS • NO SURVIVORSHIP BIAS</span>
      </footer>
    </div>
  );
}

// ── Monthly Heatmap Component ─────────────────────────────────────────────────
function MonthlyHeatmap({ curve }) {
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const monthly = {};
  for (let i = 1; i < curve.length; i++) {
    const d = curve[i].date, prev = curve[i - 1];
    if (!d) continue;
    const key = `${d.getFullYear()}-${d.getMonth()}`;
    if (!monthly[key]) monthly[key] = { year: d.getFullYear(), month: d.getMonth(), start: prev.equity };
    monthly[key].end = curve[i].equity;
  }
  const rows = {};
  Object.values(monthly).forEach(m => {
    if (!rows[m.year]) rows[m.year] = {};
    rows[m.year][m.month] = m.end && m.start ? (m.end / m.start - 1) * 100 : null;
  });
  const years = Object.keys(rows).sort();
  const getColor = (v) => {
    if (v === null) return "#0d1a0d";
    if (v > 5) return "#00ff8888"; if (v > 2) return "#00ff8855"; if (v > 0) return "#00ff8833";
    if (v > -2) return "#ff446633"; if (v > -5) return "#ff446655"; return "#ff446688";
  };
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", width: "100%", fontFamily: "Courier New, monospace", fontSize: 9 }}>
        <thead>
          <tr>
            <th style={{ color: "#2a4a2a", padding: "4px 6px", textAlign: "left" }}>YEAR</th>
            {months.map(m => <th key={m} style={{ color: "#2a4a2a", padding: "4px 4px", textAlign: "center" }}>{m}</th>)}
          </tr>
        </thead>
        <tbody>
          {years.map(year => (
            <tr key={year}>
              <td style={{ color: "#4a7a4a", padding: "3px 6px", fontWeight: "bold" }}>{year}</td>
              {months.map((_, mi) => {
                const v = rows[year][mi];
                return (
                  <td key={mi} style={{ background: getColor(v), padding: "3px 2px", textAlign: "center", color: v === null ? "transparent" : v >= 0 ? "#00ff88" : "#ff6644", fontWeight: "bold", border: "1px solid #0a140a" }}>
                    {v !== null ? `${v > 0 ? "+" : ""}${v.toFixed(1)}` : "—"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = {
  root: { background: "#030803", minHeight: "100vh", color: "#c0d8c0", fontFamily: "'Courier New', Courier, monospace", position: "relative", overflowX: "hidden" },
  scanlineOverlay: { position: "fixed", top: 0, left: 0, right: 0, bottom: 0, backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px)", pointerEvents: "none", zIndex: 9999 },
  loadingScreen: { display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh", background: "#030803" },
  loadingContent: { textAlign: "center" },
  loadingLogo: { color: "#00ff88", fontFamily: "Courier New, monospace", fontSize: 36, fontWeight: "bold", letterSpacing: 8, marginBottom: 8 },
  loadingBar: { width: 320, height: 3, background: "#0d1a0d", borderRadius: 2, overflow: "hidden", margin: "0 auto" },
  loadingFill: { height: "100%", background: "linear-gradient(90deg, #00ff88, #00ccff)", transition: "width 0.1s", boxShadow: "0 0 8px #00ff88" },
  header: { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 20px", borderBottom: "1px solid #0d1a0d", background: "#050d05" },
  headerLeft: {},
  logo: { color: "#00ff88", fontSize: 18, fontWeight: "bold", letterSpacing: 4, textShadow: "0 0 12px #00ff8866" },
  headerSub: { color: "#1a3a1a", fontSize: 9, letterSpacing: 3, marginTop: 2 },
  headerSymbols: { display: "flex", gap: 8 },
  symPill: { background: "transparent", border: "1px solid", padding: "4px 12px", cursor: "pointer", display: "flex", flexDirection: "column", alignItems: "center", gap: 2, fontFamily: "Courier New, monospace", transition: "all 0.2s", letterSpacing: 1 },
  headerRight: { display: "flex", alignItems: "center", gap: 12 },
  statusDot: { width: 6, height: 6, borderRadius: "50%", boxShadow: "0 0 6px currentColor", animation: "pulse 2s infinite" },
  toggleBtn: { background: "transparent", border: "1px solid #1a3a1a", color: "#2a4a2a", padding: "3px 8px", cursor: "pointer", fontFamily: "Courier New, monospace", fontSize: 9, letterSpacing: 1 },
  tickerBar: { background: "#030803", borderBottom: "1px solid #0d1a0d", padding: "6px 0", overflow: "hidden", position: "relative" },
  tickerScroll: { display: "inline-block", whiteSpace: "nowrap", fontFamily: "Courier New, monospace", fontSize: 11, animation: "scroll 40s linear infinite", paddingLeft: "100%" },
  nav: { display: "flex", borderBottom: "1px solid #0d1a0d", background: "#040c04", paddingLeft: 12 },
  navBtn: { background: "transparent", border: "none", padding: "10px 20px", cursor: "pointer", fontFamily: "Courier New, monospace", fontSize: 10, letterSpacing: 2, transition: "all 0.2s" },
  main: { padding: 20, maxWidth: 1400, margin: "0 auto" },
  fadeIn: { animation: "fadeIn 0.4s ease" },
  kpiGrid: { display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 16 },
  kpiCard: { background: "#050d05", border: "1px solid #0d1a0d", padding: "14px 16px", position: "relative", overflow: "hidden" },
  kpiLabel: { color: "#2a4a2a", fontSize: 9, letterSpacing: 2, marginBottom: 6 },
  kpiValue: { fontSize: 20, fontWeight: "bold", lineHeight: 1 },
  kpiSub: { color: "#1a3a1a", fontSize: 9, marginTop: 4 },
  gaugeRow: { display: "grid", gridTemplateColumns: "auto 1fr", gap: 12, marginBottom: 16 },
  gaugeCard: { background: "#050d05", border: "1px solid #0d1a0d", padding: 16 },
  statsGrid: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 1, background: "#0d1a0d" },
  statRow: { background: "#050d05", padding: "8px 12px", display: "flex", justifyContent: "space-between", alignItems: "center" },
  sectionTitle: { color: "#2a5a2a", fontSize: 10, letterSpacing: 3, marginBottom: 12, fontWeight: "bold" },
  chartPanel: { background: "#050d05", border: "1px solid #0d1a0d", padding: 16 },
  chartHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 },
  table: { width: "100%", borderCollapse: "collapse", fontFamily: "Courier New, monospace", fontSize: 11 },
  th: { color: "#2a5a2a", fontSize: 9, letterSpacing: 1, padding: "8px 12px", textAlign: "left", borderBottom: "1px solid #0d1a0d", background: "#030803" },
  td: { padding: "8px 12px", color: "#6a8a6a", fontSize: 11 },
  miniStat: { background: "#050d05", border: "1px solid #0d1a0d", padding: 16 },
  footer: { borderTop: "1px solid #0d1a0d", padding: "10px 20px", display: "flex", justifyContent: "space-between", color: "#1a3a1a", fontSize: 9, letterSpacing: 1, background: "#030803", marginTop: 20 },
};

const cssAnimations = `
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
  @keyframes scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }
  * { box-sizing: border-box; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #030803; }
  ::-webkit-scrollbar-thumb { background: #0d2a0d; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #00ff8844; }
`;
