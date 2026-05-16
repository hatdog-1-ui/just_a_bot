"""Position sizing and order-level risk calculations."""

from dataclasses import dataclass

from config import RiskConfig


@dataclass
class OrderPlan:
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float      # base-asset units (e.g. BTC)
    position_value: float  # quote-asset value (e.g. USDT)


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def plan_long(self, entry_price: float, portfolio_value: float) -> OrderPlan:
        """
        Calculate stop-loss, take-profit and position size for a long entry.

        Position size is the smaller of:
          - max_risk_per_trade / (stop_loss distance as fraction of entry)
          - max_position_pct of portfolio
        """
        stop_loss = entry_price * (1 - self.cfg.stop_loss_pct)
        take_profit = entry_price * (1 + self.cfg.take_profit_pct)

        risk_per_unit = entry_price - stop_loss  # quote currency lost if SL hit
        max_loss_amount = portfolio_value * self.cfg.max_risk_per_trade

        # Units we can afford to lose at SL and still stay within risk budget
        qty_by_risk = max_loss_amount / risk_per_unit

        # Hard cap: never put more than max_position_pct of portfolio in one trade
        max_value = portfolio_value * self.cfg.max_position_pct
        qty_by_size = max_value / entry_price

        quantity = min(qty_by_risk, qty_by_size)
        position_value = quantity * entry_price

        return OrderPlan(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            position_value=position_value,
        )

    def plan_short(self, entry_price: float, portfolio_value: float) -> OrderPlan:
        """Short variant: stop above entry, take-profit below."""
        stop_loss = entry_price * (1 + self.cfg.stop_loss_pct)
        take_profit = entry_price * (1 - self.cfg.take_profit_pct)

        risk_per_unit = stop_loss - entry_price
        max_loss_amount = portfolio_value * self.cfg.max_risk_per_trade

        qty_by_risk = max_loss_amount / risk_per_unit
        max_value = portfolio_value * self.cfg.max_position_pct
        qty_by_size = max_value / entry_price

        quantity = min(qty_by_risk, qty_by_size)
        position_value = quantity * entry_price

        return OrderPlan(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            position_value=position_value,
        )

    def should_stop_loss(self, position_side: str, current_price: float, stop_loss: float) -> bool:
        if position_side == "long":
            return current_price <= stop_loss
        return current_price >= stop_loss

    def should_take_profit(self, position_side: str, current_price: float, take_profit: float) -> bool:
        if position_side == "long":
            return current_price >= take_profit
        return current_price <= take_profit
