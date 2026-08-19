"""Close job: decide, and on HOLD place a market-on-close BUY (paper). Runs ~15:45 ET."""
import logging, json
from decide import decide
import broker, config, ledger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("close")

def main():
    d = decide(); log.info("decision: %s", json.dumps(d))
    if d["stale_days"] > 3:
        ledger.record(event="close_skip", asof=d["asof"], decision=d["decision"],
                      note=f"STALE features ({d['stale_days']}d) — refusing to trade"); log.warning("stale data, skip"); return
    if not broker.clock().is_open:
        ledger.record(event="close_skip", asof=d["asof"], decision=d["decision"], note="market closed"); log.info("market closed, skip"); return
    if d["decision"] == "HOLD":
        eq = float(broker.account().equity); qty = int((eq*config.TARGET_FRAC)//d["spy_close"])
        if qty < 1: ledger.record(event="close_skip", asof=d["asof"], note="qty<1"); return
        o = broker.buy_on_close(config.SYMBOL, qty)
        ledger.record(event="close_buy", asof=d["asof"], decision="HOLD", p_gap=d["p_gap"],
                      threshold=d["threshold"], symbol=config.SYMBOL, qty=qty, order_id=str(o.id),
                      price=d["spy_close"], note="MOC buy")
        log.info("placed MOC BUY %d %s (order %s)", qty, config.SYMBOL, o.id)
    else:
        ledger.record(event="close_aside", asof=d["asof"], decision="STAND_ASIDE",
                      p_gap=d["p_gap"], threshold=d["threshold"], note="flat overnight")
        log.info("STAND_ASIDE — no position tonight")

if __name__ == "__main__":
    main()
