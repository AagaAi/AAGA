# agents/pdf_report.py
import sqlite3, datetime

def generate_daily_report():
    con = sqlite3.connect("journal.db")
    today = datetime.date.today().isoformat()
    rows = con.execute("""
        SELECT pair, COUNT(*), SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END),
               SUM(pnl), AVG(pnl)
        FROM trade_journal WHERE date(timestamp) = ? AND outcome IS NOT NULL
        GROUP BY pair
    """, (today,)).fetchall()
    con.close()
    report = f"📄 **A.A.G.A AI Daily Report – {today}**\n\n"
    total_pnl = 0
    for r in rows:
        wr = (r[2] / r[1] * 100) if r[1] > 0 else 0
        total_pnl += r[3] or 0
        report += f"*{r[0]}*: {r[1]} trades, {wr:.0f}% WR, P&L ${r[3]:.2f}\n"
    report += f"\n**Total P&L: ${total_pnl:.2f}**\n"
    # Add market regime
    from main import latest_signals_cache
    for pair, cache in latest_signals_cache.items():
        report += f"{pair} regime: {cache.get('regime','Unknown')}\n"
    return report
