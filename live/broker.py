"""Alpaca PAPER broker adapter for the overnight bot (reuses the covered_call pattern).
Paper-only hard gate: refuses any non-'PK' key. Market-on-close entry, market exit."""
import os
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/overnight_bot/.env"))
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

PAPER = "https://paper-api.alpaca.markets"

def client():
    k = os.environ["ALPACA_API_KEY_ID"]; s = os.environ["ALPACA_API_SECRET_KEY"]
    if not k.startswith("PK"):
        raise SystemExit("REFUSING: ALPACA_API_KEY_ID lacks paper 'PK' prefix — will not trade a live account.")
    ep = os.environ.get("ALPACA_PAPER_ENDPOINT", PAPER)
    if ep.rstrip("/") != PAPER:
        raise SystemExit(f"REFUSING: endpoint {ep!r} is not the paper URL.")
    return TradingClient(k, s, paper=True)

def account():  return client().get_account()
def clock():    return client().get_clock()

def position_qty(symbol):
    try:
        return float(client().get_open_position(symbol).qty)
    except Exception:
        return 0.0

def buy_on_close(symbol, qty):
    return client().submit_order(MarketOrderRequest(
        symbol=symbol, qty=int(qty), side=OrderSide.BUY, time_in_force=TimeInForce.CLS))

def sell_market(symbol, qty):
    return client().submit_order(MarketOrderRequest(
        symbol=symbol, qty=int(qty), side=OrderSide.SELL, time_in_force=TimeInForce.DAY))

def smoke_test():
    """Prove order placement without a fill: place a far-below limit buy, confirm, cancel."""
    c = client()
    a = c.get_account()
    o = c.submit_order(LimitOrderRequest(symbol="SPY", qty=1, side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY, limit_price=1.00))   # $1 limit never fills
    got = c.get_order_by_id(o.id)
    c.cancel_order_by_id(o.id)
    return dict(account=a.account_number, equity=str(a.equity),
                test_order_id=str(o.id), status=str(got.status), cancelled=True)

if __name__ == "__main__":
    import json, sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        print(json.dumps(smoke_test(), indent=1))
    else:
        a = account(); clk = clock()
        print(f"account={a.account_number} equity=${a.equity} bp=${a.buying_power}")
        print(f"market open={clk.is_open}  next_open={clk.next_open}  next_close={clk.next_close}")
        print(f"SPY position: {position_qty('SPY')}")
