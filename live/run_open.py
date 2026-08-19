"""Open job: sell any overnight position at the open (market). Runs ~09:31 ET."""
import logging
import broker, config, ledger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("open")

def main():
    qty = broker.position_qty(config.SYMBOL)
    if not broker.clock().is_open:
        ledger.record(event="open_skip", symbol=config.SYMBOL, qty=qty, note="market closed"); log.info("market closed, skip"); return
    if qty and qty > 0:
        o = broker.sell_market(config.SYMBOL, int(qty))
        ledger.record(event="open_sell", symbol=config.SYMBOL, qty=int(qty), order_id=str(o.id), note="market sell at open")
        log.info("placed SELL %d %s (order %s)", int(qty), config.SYMBOL, o.id)
    else:
        ledger.record(event="open_flat", symbol=config.SYMBOL, note="no position to close")
        log.info("no position to close")

if __name__ == "__main__":
    main()
