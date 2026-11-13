"""
==========================================
BACKTEST EUR/USD - MA CROSSOVER + IA (v2)
==========================================

CORRECTIONS APPORTÉES PAR RAPPORT À LA V1 :

1. RATIO TP/SL COHÉRENT AVEC LE WIN RATE VISÉ
   - Le seuil de rentabilité (breakeven win rate) = SL / (SL + TP)
   - v1 avait TP=3*ATR, SL=1.5*ATR -> ratio 2:1 -> il fallait 33.3% de win rate
     rien que pour ne pas perdre d'argent. Viser 30% de win rate était donc
     mathématiquement voué à l'échec avec ce ratio.
   - Ici RR_RATIO est un paramètre explicite et on calcule le seuil
     d'équilibre pour que tu voies toujours où tu te situes par rapport à lui.

2. PLUS DE SIGNAUX
   - v1 ne tradait que sur les croisements MA20/MA50 (rarissimes en daily,
     ~15-30 occurrences sur 1500 bougies -> 5-10 trades en test = pas
     statistiquement significatif).
   - Ici on ajoute un filtre de tendance + RSI + volatilité pour générer un
     signal chaque fois que les conditions sont réunies (pas seulement au
     croisement exact), ce qui donne un échantillon de test exploitable.

3. WALK-FORWARD VALIDATION au lieu d'un split unique
   - On ré-entraîne le modèle sur une fenêtre glissante plusieurs fois au
     lieu d'un seul split 70/30, pour une estimation plus robuste.

4. COÛTS DE TRANSACTION (spread) inclus dans le PnL.

5. Clé API retirée du code -> à mettre dans une variable d'environnement.

IMPORTANT : aucun réglage de code ne peut garantir un win rate ou une
rentabilité donnés sur des données futures. Ce script vise à te donner une
mesure honnête et statistiquement plus solide de la performance, pas un
résultat "consigné à l'avance".
"""

import os
from dotenv import load_dotenv

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

load_dotenv()
API_KEY = os.getenv("API_KEY")

# --- Paires tradées ---
# Sur le premier test, seule EUR/USD a montré un edge net et robuste
# (win rate 54.2%, nettement au-dessus du seuil d'équilibre 50%).
# GBP/USD était à l'équilibre (bruit), USD/JPY et AUD/USD étaient perdants.
# On se concentre donc sur EUR/USD. Réactive les autres paires seulement
# après les avoir validées individuellement sur tes propres données.
SYMBOLS = ["EUR/USD"]
INTERVAL = "1day"   # passe à "4h" pour encore plus de trades (voir note plus bas)
INITIAL_CAPITAL = 10000.0

# --- Paramètres de risk management ---
SL_ATR_MULT = 3
TP_ATR_MULT = 6   # ratio 1:1 par défaut -> seuil d'équilibre = 50% de win rate
                      # (change TP_ATR_MULT pour ajuster le compromis WR / RR, voir plus bas)
PROBA_THRESHOLD = 0.50

# --- Coûts de transaction ---
SPREAD_PIPS = 1.0          # spread typique en pips (à ajuster selon ton broker)


def get_pip_size(symbol):
    """Les paires cotées avec le YEN (JPY) utilisent 2 décimales (1 pip = 0.01)
    au lieu de 4 décimales (1 pip = 0.0001) pour les autres paires majeures."""
    return 0.01 if "JPY" in symbol else 0.0001

# --- Sizing basé sur la confiance du modèle ---
# On risque un % du capital par trade, ce % augmentant avec la confiance
# de l'IA (proba_success), entre BASE_RISK_PCT (au seuil) et MAX_RISK_PCT
# (proba = 1.0). C'est plus réaliste que "tout miser à chaque trade" et ça
# donne plus de poids aux trades où le modèle est le plus confiant.
BASE_RISK_PCT = 0.02   # 2% du capital risqué au minimum (proba juste au seuil)
MAX_RISK_PCT = 0.05    # 5% du capital risqué au maximum (proba = 1.0)
# NOTE : ce réglage ("Modéré") vise plus de rendement contre plus de
# drawdown. Regarde le nouveau panneau de drawdown sur le graphique pour
# juger si tu peux psychologiquement tenir ce niveau de recul de capital.
# Si c'est trop violent, redescends vers 0.01 / 0.03 (profil "Actuel"/prudent).

# --- Walk-forward ---
N_FOLDS = 5
MIN_TRAIN_SIZE = 300

BREAKEVEN_WR = SL_ATR_MULT / (SL_ATR_MULT + TP_ATR_MULT)
print(f"Ratio TP:SL = {TP_ATR_MULT}:{SL_ATR_MULT}  ->  win rate d'équilibre requis = {BREAKEVEN_WR:.1%}")


# ==========================================
# 2. RÉCUPÉRATION DES DONNÉES
# ==========================================
def fetch_forex_data(symbol, interval, api_key):
    if not api_key:
        raise ValueError(
            "Aucune clé API trouvée. Définis la variable d'environnement "
            "TWELVEDATA_API_KEY avant de lancer le script."
        )
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=5000&apikey={api_key}"
    response = requests.get(url).json()

    if "values" not in response:
        raise ValueError(f"Erreur API: {response.get('message', 'Inconnue')}")

    df = pd.DataFrame(response["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    return df


# ==========================================
# 3. FEATURE ENGINEERING & CIBLE TP/SL
# ==========================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def prepare_ml_features(df):
    df = df.copy()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["close"], 14)

    df["tr"] = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = df["tr"].rolling(14).mean()

    # Features contextuelles
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["ma_diff"] = (df["MA20"] - df["MA50"]) / df["close"]
    df["ma20_slope"] = df["MA20"].diff(3) / df["close"]
    df["ma50_slope"] = df["MA50"].diff(3) / df["close"]
    df["dist_ma20"] = (df["close"] - df["MA20"]) / df["close"]
    df["norm_atr"] = df["ATR"] / df["close"]
    df["rsi_norm"] = (df["RSI"] - 50) / 50

    # --- SIGNAUX ---
    # Au lieu du seul croisement exact (trop rare), on trade tant que la
    # tendance MA20>MA50 est établie ET que le prix vient de "pull back"
    # vers la MA20 (zone d'entrée classique en tendance), avec un filtre
    # RSI pour éviter les entrées en zone de surachat/survente extrême.
    trend_up = df["MA20"] > df["MA50"]
    trend_down = df["MA20"] < df["MA50"]
    pullback_up = (df["close"] <= df["MA20"] * 1.001) & (df["close"].shift(1) > df["MA20"].shift(1) * 1.001)
    pullback_down = (df["close"] >= df["MA20"] * 0.999) & (df["close"].shift(1) < df["MA20"].shift(1) * 0.999)

    df["raw_signal"] = 0
    df.loc[trend_up & pullback_up & (df["RSI"] < 65), "raw_signal"] = 1
    df.loc[trend_down & pullback_down & (df["RSI"] > 35), "raw_signal"] = -1

    # Cible : 1 si TP atteint avant SL
    df["target"] = np.nan
    close = df["close"].values
    atr = df["ATR"].values
    high = df["high"].values
    low = df["low"].values
    sig = df["raw_signal"].values

    for i in range(len(df) - 30):
        s = sig[i]
        if s == 0 or np.isnan(atr[i]):
            continue

        entry = close[i]
        a = atr[i]

        if s == 1:
            sl = entry - SL_ATR_MULT * a
            tp = entry + TP_ATR_MULT * a
            for j in range(i + 1, min(i + 30, len(df))):
                if low[j] <= sl:
                    df.iloc[i, df.columns.get_loc("target")] = 0
                    break
                if high[j] >= tp:
                    df.iloc[i, df.columns.get_loc("target")] = 1
                    break
        else:
            sl = entry + SL_ATR_MULT * a
            tp = entry - TP_ATR_MULT * a
            for j in range(i + 1, min(i + 30, len(df))):
                if high[j] >= sl:
                    df.iloc[i, df.columns.get_loc("target")] = 0
                    break
                if low[j] <= tp:
                    df.iloc[i, df.columns.get_loc("target")] = 1
                    break

    return df.dropna(subset=["MA50", "ATR", "RSI"]).reset_index(drop=True)


FEATURE_COLS = [
    "day_of_week", "ma_diff", "ma20_slope", "ma50_slope",
    "dist_ma20", "norm_atr", "rsi_norm",
]


# ==========================================
# 4. WALK-FORWARD TRAIN + PREDICT
# ==========================================
def walk_forward_predict(df, n_folds=N_FOLDS, min_train_size=MIN_TRAIN_SIZE):
    """
    Découpe la série en n_folds segments de test consécutifs. Pour chaque
    segment, le modèle est entraîné UNIQUEMENT sur les données antérieures
    (jamais sur le futur), puis on prédit sur le segment suivant.
    Évite le biais d'un split unique et donne plus de trades de test.
    """
    n = len(df)
    test_size = (n - min_train_size) // n_folds
    if test_size < 20:
        raise ValueError("Pas assez de données pour ce nombre de folds.")

    all_test_frames = []

    for fold in range(n_folds):
        train_end = min_train_size + fold * test_size
        test_start = train_end
        test_end = min(train_end + test_size, n)
        if test_end <= test_start:
            continue

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end].copy().reset_index(drop=True)

        train_signals = train_df[(train_df["raw_signal"] != 0) & (train_df["target"].notna())]
        if len(train_signals) < 20:
            continue  # pas assez d'exemples pour entraîner ce fold

        X_train = train_signals[FEATURE_COLS]
        y_train = train_signals["target"].astype(int)
        X_test = test_df[FEATURE_COLS]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        test_df["proba_success"] = model.predict_proba(X_test_scaled)[:, 1]
        test_df["fold"] = fold
        all_test_frames.append(test_df)

    if not all_test_frames:
        raise ValueError("Aucun fold valide - pas assez de signaux dans les données.")

    return pd.concat(all_test_frames, ignore_index=True)


# ==========================================
# 5. BACKTEST (avec coûts de transaction)
# ==========================================
def confidence_risk_pct(proba, threshold=PROBA_THRESHOLD, base_risk=BASE_RISK_PCT, max_risk=MAX_RISK_PCT):
    """
    Convertit la proba du modèle en % de capital risqué sur ce trade :
    - proba == threshold -> base_risk
    - proba == 1.0       -> max_risk
    - interpolation linéaire entre les deux
    """
    if proba <= threshold:
        return base_risk
    confidence = (proba - threshold) / (1 - threshold)
    confidence = min(max(confidence, 0.0), 1.0)
    return base_risk + confidence * (max_risk - base_risk)


def run_backtest(df_test, capital, threshold=PROBA_THRESHOLD, base_risk=BASE_RISK_PCT, max_risk=MAX_RISK_PCT, verbose=True, symbol="EUR/USD"):
    trades = []
    equity_curve = [{"date": df_test["datetime"].iloc[0], "capital": capital}]
    position = 0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    risk_pct = 0.0
    spread_cost = SPREAD_PIPS * get_pip_size(symbol)

    n_raw_signals = len(df_test[df_test["raw_signal"] != 0])
    n_ia_signals = len(df_test[(df_test["raw_signal"] != 0) & (df_test["proba_success"] >= threshold)])
    if verbose:
        print(f"--> Signaux bruts (avant filtre IA) : {n_raw_signals}")
        print(f"--> Signaux retenus (après filtre IA >= {threshold:.0%}) : {n_ia_signals}")

    for i in range(len(df_test)):
        row = df_test.iloc[i]
        date = row["datetime"]
        close = row["close"]
        high = row["high"]
        low = row["low"]
        atr = row["ATR"]
        proba = row["proba_success"]
        raw_signal = row["raw_signal"]

        if position == 1:
            hit = None
            if low <= stop_loss:
                hit, exit_price = "SL Long", stop_loss
            elif high >= take_profit:
                hit, exit_price = "TP Long", take_profit
            if hit:
                price_pnl_pct = (exit_price - entry_price) / entry_price - spread_cost / entry_price
                r_multiple = price_pnl_pct / entry_sl_dist_pct
                capital *= 1 + risk_pct * r_multiple
                trades.append({"date": date, "type": hit, "price": exit_price, "pnl": price_pnl_pct, "proba": entry_proba})
                equity_curve.append({"date": date, "capital": capital})
                position = 0

        elif position == -1:
            hit = None
            if high >= stop_loss:
                hit, exit_price = "SL Short", stop_loss
            elif low <= take_profit:
                hit, exit_price = "TP Short", take_profit
            if hit:
                price_pnl_pct = (entry_price - exit_price) / entry_price - spread_cost / entry_price
                r_multiple = price_pnl_pct / entry_sl_dist_pct
                capital *= 1 + risk_pct * r_multiple
                trades.append({"date": date, "type": hit, "price": exit_price, "pnl": price_pnl_pct, "proba": entry_proba})
                equity_curve.append({"date": date, "capital": capital})
                position = 0

        if position == 0 and raw_signal != 0 and proba >= threshold:
            risk_pct = confidence_risk_pct(proba, threshold, base_risk, max_risk)
            entry_proba = proba
            if raw_signal == 1:
                position = 1
                entry_price = close
                stop_loss = entry_price - SL_ATR_MULT * atr
                take_profit = entry_price + TP_ATR_MULT * atr
                entry_sl_dist_pct = (SL_ATR_MULT * atr) / entry_price
                trades.append({"date": date, "type": "BUY (IA)", "price": entry_price, "pnl": 0.0, "proba": proba})
            else:
                position = -1
                entry_price = close
                stop_loss = entry_price + SL_ATR_MULT * atr
                take_profit = entry_price - TP_ATR_MULT * atr
                entry_sl_dist_pct = (SL_ATR_MULT * atr) / entry_price
                trades.append({"date": date, "type": "SELL (IA)", "price": entry_price, "pnl": 0.0, "proba": proba})

    return pd.DataFrame(trades), df_test, capital, pd.DataFrame(equity_curve)


# ==========================================
# 6. STATISTIQUES + AFFICHAGE
# ==========================================
def print_stats(trades_df, initial_capital, final_capital):
    closed = trades_df[trades_df["type"].str.contains("SL|TP")] if not trades_df.empty else pd.DataFrame()
    total = len(closed)
    wins = len(closed[closed["pnl"] > 0]) if total else 0
    win_rate = wins / total * 100 if total else 0
    total_return = (final_capital - initial_capital) / initial_capital * 100

    gross_win = closed.loc[closed["pnl"] > 0, "pnl"].sum() if total else 0
    gross_loss = -closed.loc[closed["pnl"] <= 0, "pnl"].sum() if total else 0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("nan")
    avg_win = closed.loc[closed["pnl"] > 0, "pnl"].mean() if wins else 0
    avg_loss = closed.loc[closed["pnl"] <= 0, "pnl"].mean() if (total - wins) else 0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    print("\n" + "=" * 55)
    print("      RÉSULTATS - WALK-FORWARD MA + IA (v2)      ")
    print("=" * 55)
    print(f"Capital Initial         : ${initial_capital:,.2f}")
    print(f"Capital Final           : ${final_capital:,.2f}")
    print(f"Rendement Total         : {total_return:+.2f}%")
    print(f"Nombre de Trades        : {total}")
    print(f"Win Rate                : {win_rate:.2f}%")
    print(f"Win Rate d'équilibre    : {BREAKEVEN_WR:.1%}  (avec ce ratio TP:SL)")
    print(f"Profit Factor           : {profit_factor:.2f}" if total else "Profit Factor           : N/A")
    print(f"Espérance / trade       : {expectancy:.4%}")
    print("=" * 55 + "\n")

    if win_rate < BREAKEVEN_WR * 100:
        print("⚠️  Win rate SOUS le seuil d'équilibre -> stratégie perdante avec ce ratio.")
    else:
        print("✅ Win rate au-dessus du seuil d'équilibre -> marge de rentabilité positive.")


def plot_results(df_test, trades_df, equity_df):
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(15, 13), sharex=True,
        gridspec_kw={"height_ratios": [2, 3, 1]},
    )

    # --- Panneau 1 : courbe d'équité (le graphique "argent gagné") ---
    ax0.plot(equity_df["date"], equity_df["capital"], color="#1f77b4", linewidth=1.8)
    ax0.fill_between(equity_df["date"], equity_df["capital"], equity_df["capital"].iloc[0],
                      where=(equity_df["capital"] >= equity_df["capital"].iloc[0]),
                      color="green", alpha=0.15, interpolate=True)
    ax0.fill_between(equity_df["date"], equity_df["capital"], equity_df["capital"].iloc[0],
                      where=(equity_df["capital"] < equity_df["capital"].iloc[0]),
                      color="red", alpha=0.15, interpolate=True)
    ax0.axhline(equity_df["capital"].iloc[0], color="grey", linestyle=":", linewidth=1)
    ax0.set_title("Courbe d'équité (évolution du capital)", fontsize=13)
    ax0.set_ylabel("Capital ($)")

    # Drawdown en repère visuel (annotation du pire recul)
    running_max = equity_df["capital"].cummax()
    drawdown = (equity_df["capital"] - running_max) / running_max
    max_dd_idx = drawdown.idxmin()
    if drawdown.min() < 0:
        ax0.annotate(
            f"Max Drawdown: {drawdown.min():.1%}",
            xy=(equity_df["date"].iloc[max_dd_idx], equity_df["capital"].iloc[max_dd_idx]),
            xytext=(10, -20), textcoords="offset points",
            fontsize=9, color="darkred",
        )

    # --- Panneau 2 : prix + MAs + trades ---
    ax1.plot(df_test["datetime"], df_test["close"], label="EUR/USD", color="black", alpha=0.5)
    ax1.plot(df_test["datetime"], df_test["MA20"], label="MA 20", color="green", linewidth=1.2)
    ax1.plot(df_test["datetime"], df_test["MA50"], label="MA 50", color="red", linewidth=1.2)

    if not trades_df.empty:
        buys = trades_df[trades_df["type"] == "BUY (IA)"]
        sells = trades_df[trades_df["type"] == "SELL (IA)"]
        if not buys.empty:
            ax1.scatter(buys["date"], buys["price"], marker="^", color="darkgreen", s=120, label="Achat IA", zorder=5)
        if not sells.empty:
            ax1.scatter(sells["date"], sells["price"], marker="v", color="crimson", s=120, label="Vente IA", zorder=5)

    ax1.set_title("EUR/USD - Backtest Walk-Forward MA + IA (v2)", fontsize=13)
    ax1.legend(loc="best")

    # --- Panneau 3 : proba IA ---
    ax2.plot(df_test["datetime"], df_test["proba_success"], color="purple", alpha=0.7, label="Probabilité Succès IA")
    ax2.axhline(PROBA_THRESHOLD, color="green", linestyle="--", label=f"Seuil IA ({PROBA_THRESHOLD:.0%})")
    ax2.set_ylabel("Proba IA")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="best")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    plt.savefig("backtest_result.png", dpi=120)
    print("Graphique sauvegardé -> backtest_result.png")
    plt.show()


# ==========================================
# 7. BALAYAGE DE SEUIL (threshold sweep)
# ==========================================
def threshold_sweep(df_test, capital, thresholds=None):
    """
    Teste plusieurs valeurs de PROBA_THRESHOLD sur les MÊMES prédictions
    walk-forward (donc sans re-entraîner) et affiche un tableau comparatif.
    Sert à repérer le seuil qui maximise le profit factor / rendement,
    SANS le sur-optimiser à l'excès (reste sur 5-8 valeurs raisonnables).
    """
    if thresholds is None:
        thresholds = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]

    print("\n" + "=" * 70)
    print("BALAYAGE DE SEUIL (sur les mêmes prédictions walk-forward)")
    print("=" * 70)
    print(f"{'Seuil':>6} | {'Trades':>7} | {'Win Rate':>9} | {'Rendement':>10} | {'Profit Factor':>13}")
    print("-" * 70)

    results = []
    for t in thresholds:
        trades_df, _, final_cap, _ = run_backtest(df_test, capital, threshold=t, verbose=False)
        closed = trades_df[trades_df["type"].str.contains("SL|TP")] if not trades_df.empty else pd.DataFrame()
        total = len(closed)
        wins = len(closed[closed["pnl"] > 0]) if total else 0
        wr = wins / total * 100 if total else 0
        ret = (final_cap - capital) / capital * 100
        gw = closed.loc[closed["pnl"] > 0, "pnl"].sum() if total else 0
        gl = -closed.loc[closed["pnl"] <= 0, "pnl"].sum() if total else 0
        pf = gw / gl if gl > 0 else float("nan")
        results.append({"threshold": t, "trades": total, "win_rate": wr, "return": ret, "profit_factor": pf})
        print(f"{t:>6.2f} | {total:>7} | {wr:>8.2f}% | {ret:>+9.2f}% | {pf:>13.2f}")

    print("=" * 70)
    print("⚠️  Choisis le seuil avec le PLUS de trades parmi les meilleurs résultats,")
    print("    pas juste le chiffre le plus élevé (risque de sur-optimisation sur peu de trades).\n")
    return pd.DataFrame(results)


# ==========================================
# 8. COMPARATIF DE PROFILS DE RISQUE
# ==========================================
def risk_profile_comparison(df_test, capital, threshold=PROBA_THRESHOLD):
    """
    Montre le compromis rendement / drawdown pour différents niveaux de
    risque par trade. Plus de risque = plus de rendement POTENTIEL mais
    aussi un drawdown (recul de capital) bien plus violent.
    Aucun de ces profils n'est "le bon" -> c'est un choix personnel de
    tolérance au risque, pas un paramètre à maximiser aveuglément.
    """
    profiles = [
        ("Prudent",    0.005, 0.015),
        ("Actuel",     BASE_RISK_PCT, MAX_RISK_PCT),
        ("Modéré",     0.02, 0.05),
        ("Agressif",   0.04, 0.08),
        ("Très agressif", 0.06, 0.12),
    ]

    print("\n" + "=" * 78)
    print("COMPARATIF DE PROFILS DE RISQUE (même edge, sizing différent)")
    print("=" * 78)
    print(f"{'Profil':>15} | {'Risque/trade':>13} | {'Rendement':>10} | {'Max Drawdown':>13}")
    print("-" * 78)

    for name, br, mr in profiles:
        trades_df, _, final_cap, equity_df = run_backtest(
            df_test, capital, threshold=threshold, base_risk=br, max_risk=mr, verbose=False
        )
        ret = (final_cap - capital) / capital * 100
        running_max = equity_df["capital"].cummax()
        dd = ((equity_df["capital"] - running_max) / running_max).min() * 100
        print(f"{name:>15} | {br:.1%}-{mr:.1%}   | {ret:>+9.2f}% | {dd:>12.2f}%")

    print("=" * 78)
    print("Plus le risque par trade augmente, plus le rendement ET le drawdown")
    print("augmentent ensemble. Un drawdown de -40% ou plus est psychologiquement")
    print("très difficile à tenir en réel, même si le rendement final est bon.\n")


def combine_portfolio_equity(equity_dfs, capital_per_symbol):
    """
    Fusionne les courbes d'équité de plusieurs paires en une seule courbe de
    portefeuille. Chaque paire évolue indépendamment à partir du même
    capital de départ (capital_per_symbol) ; on additionne les capitaux
    interpolés sur l'union des dates pour obtenir le capital total.
    """
    all_dates = sorted(set().union(*[set(e["date"]) for e in equity_dfs.values()]))
    portfolio = pd.DataFrame({"date": all_dates})

    total = pd.Series(0.0, index=portfolio.index)
    for symbol, eq in equity_dfs.items():
        eq = eq.drop_duplicates(subset="date").set_index("date").reindex(all_dates)
        eq["capital"] = eq["capital"].ffill().fillna(capital_per_symbol)
        total += eq["capital"].values

    portfolio["capital"] = total.values
    return portfolio


def plot_portfolio_equity(portfolio_equity, initial_capital):
    plt.style.use("default")
    final_capital = portfolio_equity["capital"].iloc[-1]
    total_return = (final_capital - initial_capital) / initial_capital * 100
    running_max = portfolio_equity["capital"].cummax()
    drawdown = (portfolio_equity["capital"] - running_max) / running_max * 100

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("white")

    # --- Panneau 1 : courbe d'équité ---
    line_color = "#2ca02c" if total_return >= 0 else "#d62728"
    ax0.plot(portfolio_equity["date"], portfolio_equity["capital"], color=line_color, linewidth=2)
    ax0.fill_between(portfolio_equity["date"], portfolio_equity["capital"], initial_capital,
                      where=(portfolio_equity["capital"] >= initial_capital),
                      color="#2ca02c", alpha=0.12, interpolate=True)
    ax0.fill_between(portfolio_equity["date"], portfolio_equity["capital"], initial_capital,
                      where=(portfolio_equity["capital"] < initial_capital),
                      color="#d62728", alpha=0.12, interpolate=True)
    ax0.axhline(initial_capital, color="grey", linestyle="--", linewidth=1, alpha=0.6, label="Capital initial")

    # Pic de capital atteint
    peak_idx = portfolio_equity["capital"].idxmax()
    ax0.scatter(portfolio_equity["date"].iloc[peak_idx], portfolio_equity["capital"].iloc[peak_idx],
                color="#1f77b4", zorder=5, s=50, label=f"Pic: ${portfolio_equity['capital'].iloc[peak_idx]:,.0f}")

    ax0.set_title(
        f"Portefeuille {'/'.join(SYMBOLS)} — Capital final : ${final_capital:,.0f} ({total_return:+.2f}%)",
        fontsize=14, fontweight="bold", loc="left",
    )
    ax0.set_ylabel("Capital ($)")
    ax0.yaxis.set_major_formatter(lambda x, pos: f"${x:,.0f}")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper left", frameon=False)

    # --- Panneau 2 : drawdown (recul sous le pic) ---
    ax1.fill_between(portfolio_equity["date"], drawdown, 0, color="#d62728", alpha=0.35)
    ax1.plot(portfolio_equity["date"], drawdown, color="#d62728", linewidth=1)
    ax1.set_ylabel("Drawdown (%)")
    ax1.set_ylim(min(drawdown.min() * 1.15, -1), 1)
    ax1.grid(True, alpha=0.25)
    ax1.annotate(
        f"Max: {drawdown.min():.1f}%",
        xy=(portfolio_equity["date"].iloc[drawdown.idxmin()], drawdown.min()),
        xytext=(8, -12), textcoords="offset points", fontsize=9, color="#d62728", fontweight="bold",
    )

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("portfolio_equity.png", dpi=140, facecolor="white")
    print("Graphique portefeuille sauvegardé -> portfolio_equity.png")
    plt.show()


# ==========================================
# EXECUTION PRINCIPALE
# ==========================================
if __name__ == "__main__":
    capital_per_symbol = INITIAL_CAPITAL / len(SYMBOLS)
    all_trades = []
    equity_dfs = {}

    for symbol in SYMBOLS:
        print(f"\n########## {symbol} ##########")
        try:
            df_raw = fetch_forex_data(symbol, INTERVAL, API_KEY)
            df_ml = prepare_ml_features(df_raw)
            df_test = walk_forward_predict(df_ml)
        except ValueError as e:
            print(f"⚠️  {symbol} ignoré : {e}")
            continue

        trades_df, df_bt, final_capital, equity_df = run_backtest(
            df_test, capital_per_symbol, threshold=PROBA_THRESHOLD, symbol=symbol
        )
        trades_df["symbol"] = symbol
        all_trades.append(trades_df)
        equity_dfs[symbol] = equity_df

        closed = trades_df[trades_df["type"].str.contains("SL|TP")] if not trades_df.empty else pd.DataFrame()
        wr = (closed["pnl"] > 0).mean() * 100 if len(closed) else 0
        print(f"--> {symbol} : {len(closed)} trades, win rate {wr:.1f}%, "
              f"capital {capital_per_symbol:,.0f} -> {final_capital:,.0f}")

        # Outils de diagnostic : à quel seuil / niveau de risque cette paire
        # donnerait-elle le meilleur compromis ? (affichage informatif seul,
        # ne change PAS automatiquement PROBA_THRESHOLD/BASE_RISK_PCT).
        threshold_sweep(df_test, capital_per_symbol)
        risk_profile_comparison(df_test, capital_per_symbol, threshold=PROBA_THRESHOLD)

    if not all_trades:
        raise SystemExit("Aucune paire n'a pu être traitée - vérifie ta clé API / connexion.")

    combined_trades = pd.concat(all_trades, ignore_index=True)
    final_total_capital = sum(eq["capital"].iloc[-1] for eq in equity_dfs.values())

    print("\n" + "#" * 60)
    print("RÉSULTATS COMBINÉS - PORTEFEUILLE MULTI-PAIRES")
    print("#" * 60)
    print_stats(combined_trades, INITIAL_CAPITAL, final_total_capital)

    portfolio_equity = combine_portfolio_equity(equity_dfs, capital_per_symbol)
    plot_portfolio_equity(portfolio_equity, INITIAL_CAPITAL)