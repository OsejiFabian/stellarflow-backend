import math
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TradePoint:
    timestamp: datetime
    price: float
    volume: float


@dataclass
class SwapLeg:
    """A single child order in a Twap execution schedule."""
    index: int
    timestamp: datetime
    asset_in: str
    asset_out: str
    amount_in: float
    expected_price: float
    expected_out: float
    market_impact: float
    gas_cost_eth: float


@dataclass
class DiversificationPlan:
    """Result of simulating a treasury diversification swap route."""
    legs: List[SwapLeg]
    total_amount_in: float
    total_expected_out: float
    average_market_impact: float
    max_market_impact: float
    total_gas_eth: float
    total_gas_usd: float
    gas_breakdown: Dict[str, float]
    schedule_start: datetime
    schedule_end: datetime


class TWAPENgine:
    """Calculates Time-Weighted Average Price (TWAP) with outlier filtering."""

    @staticmethod
    def filter_outliers(trades: List[TradePoint], variance_threshold: float = 0.50) -> List[TradePoint]:
        """Filters out trades exceeding `variance_threshold` (default 50%) from moving median."""
        if not trades:
            return []

        prices = [t.price for t in trades]
        median_price = float(np.median(prices))

        if median_price == 0:
            return trades

        filtered_trades = []
        for trade in trades:
            variance = abs(trade.price - median_price) / median_price
            if variance <= variance_threshold:
                filtered_trades.append(trade)

        return filtered_trades

    @classmethod
    def calculate_twap(
        cls,
        trades: List[TradePoint],
        window: timedelta,
        current_time: Optional[datetime] = None,
    ) -> float:
        """Calculates time-weighted average price over a given time window."""
        if not trades:
            return 0.0

        if current_time is None:
            current_time = datetime.now(timezone.utc)

        start_time = current_time - window
        
        # Sort trades by timestamp ascending
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        
        # Filter trades within the window
        window_trades = [t for t in sorted_trades if t.timestamp >= start_time]

        # Filter price outliers
        clean_trades = cls.filter_outliers(window_trades)

        if not clean_trades:
            return 0.0

        # Compute time-weighted average price using linear interval integration
        total_time_weighted_price = 0.0
        total_time_delta = 0.0

        for i in range(len(clean_trades)):
            current = clean_trades[i]
            # Determine interval duration to the next trade or current_time
            if i < len(clean_trades) - 1:
                next_time = clean_trades[i + 1].timestamp
            else:
                next_time = current_time

            duration = (next_time - current.timestamp).total_seconds()
            if duration > 0:
                total_time_weighted_price += current.price * duration
                total_time_delta += duration

        if total_time_delta == 0:
            return clean_trades[-1].price

        return round(total_time_weighted_price / total_time_delta, 6)


class TreasuryDiversificationSimulator:
    """Simulates market impact of multi-asset treasury diversification swaps.

    Splits large diversification swaps into Twap orders to minimize market impact
    and outputs an execution schedule with an estimated gas cost breakdown.
    """

    # Default liquidity depth factors (in USD) for common treasury assets.
    DEFAULT_LIQUIDITY_DEPTH: Dict[str, float] = {
        "USDT": 500_000_000.0,
        "USDC": 200_000_000.0,
        "DAI": 100_000_000.0,
        "WBTC": 50_000_000.0,
        "WETH": 50_000_000.0,
        "STOSE": 5_000_000.0,
        "ARB": 2_000_000.0,
        "OP": 2_000_000.0,
        "MATIC": 1_000_000.0,
        "LARGE": 1_000_000.0,
    }

    # Default gas cost estimates (ETH) per swap leg by protocol.
    DEFAULT_GAS_PER_LEG: Dict[str, float] = {
        "uniswap_v2": 0.0015,
        "uniswap_v3": 0.0025,
        "curve": 0.0020,
        "balancer": 0.0030,
    }

    # Default gas price in USD per ETH.
    DEFAULT_GAS_PRICE_USD: float = 3000.0

    def __init__(
        self,
        liquidity_depth: Optional[Dict[str, float]] = None,
        gas_per_leg: Optional[Dict[str, float]] = None,
        gas_price_usd: float = DEFAULT_GAS_PRICE_USD,
        max_impact: float = 0.005,
    ) -> None:
        """
        Args:
            liquidity_depth: Map of asset symbol -> available liquidity depth in USD.
            gas_per_leg: Map of protocol -> gas cost in ETH per swap leg.
            gas_price_usd: Price of one ETH in USD.
            max_impact: Maximum acceptable market impact per leg (0.5% default).
        """
        self.liquidity_depth = dict(self.DEFAULT_LIQUIDITY_DEPTH)
        if liquidity_depth:
            self.liquidity_depth.update(liquidity_depth)

        self.gas_per_leg = dict(self.DEFAULT_GAS_PER_LEG)
        if gas_per_leg:
            self.gas_per_leg.update(gas_per_leg)

        self.gas_price_usd = gas_price_usd
        self.max_impact = max_impact

    def _estimate_market_impact(self, amount_usd: float, asset: str, liquidity_depth_usd: Optional[float] = None) -> float:
        """Estimates price impact for a swap using a constant-product like model."""
        if amount_usd <= 0:
            return 0.0

        depth = liquidity_depth_usd
        if depth is None:
            depth = self.liquidity_depth.get(asset, 1_000_000.0)

        if depth <= 0:
            return 1.0

        # Constant-product impact approximation: dx / (depth + dx)
        impact = amount_usd / (depth + amount_usd)
        return min(impact, 1.0)

    def _choose_slice_count(
        self,
        total_amount_usd: float,
        asset: str,
        liquidity_depth_usd: Optional float = None,
    ) -> int:
        """Determines the minimum number of Twap slices to keep impact <= max_impact."""
        if total_amount_usd <= 0:
            return 1

        depth = liquidity_depth_usd
        if depth is None:
            depth = self.liquidity_depth.get(asset, 1_000_000.0)

        if depth <= 0:
            return 1

        # For constant-product, impact = x / (depth + x) <= max_impact
        # x <= max_impact * depth / (1 - max_impact)
        max_slice_usd = self.max_impact * depth / (1.0 - self.max_impact)
        if max_slice_usd <= 0:
            return 1

        slices = int(math.ceil(total_amount_usd / max_slice_usd))
        return max(slices, 1)

    def simulate_diversification(
        self,
        amount_in: float,
        asset_in: str,
        target_allocations: Dict[str, float],
        asset_prices: Dict[str, float],
        duration_hours: float = 24.0,
        num_slices: Optional[int] = None,
        protocol: str = "uniswap_v3",
        start_time: Optional[datetime] = None,
        liquidity_depth_usd: Optional float = None,
    ) -> DiversificationPlan:
        """Simulates a multi-asset treasury diversification swap via Twap.

        Args:
            amount_in: Total amount of the input asset to swap.
            asset_in: Symbol of the input asset.
            target_allocations: Map of output asset -> fraction of total value (1.0).
            asset_prices: Map of asset symbol -> USD price.
            duration_hours: Duration of the Twap execution in hours.
            num_slices: Explicit number of Twap slices. Auto-computed if None.
            protocol: DEX/AGM protocol used for gas estimation.
            start_time: Schedule start time. Defaults to now (UTC).
            liquidity_depth_usd: Optional override for the input asset liquidity depth.

        Returns:
            DiversificationPlan with execution schedule and gas cost breakdown.
        """
        if amount_in <= 0:
            raise ValueError("amount_in must be positive")
        if not target_allocations:
            raise ValueError("target_allocations must not be empty")
        if asset_in not in asset_prices:
            raise ValueError(f"missing price for input asset {asset_in}")

        alloc_sum = sum(target_allocations.values())
        if alloc_sum <= 0:
            raise ValueError("target_allocations must sum to a positive value")

        normalized_allocs = {k: v / alloc_sum for k, v in target_allocations.items()}

        for asset in normalized_allocs:
            if asset not in asset_prices:
                raise ValueError(f"missing price for output asset {asset}")

        if start_time is None:
            start_time = datetime.now(timezone.utc)
        elif start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        if duration_hours <= 0:
            raise ValueError("duration_hours must be positive")

        price_in = asset_prices[asset_in]
        total_value_usd = amount_in * price_in
        depth_in = liquidity_depth_usd if liquidity_depth_usd is not None else self.liquidity_depth.get(asset_in, 1_000_000.0)

        # Determine the number of Twap slices required to keep impact <= max_impact.
        if num_slices is None:
            num_slices = self._choose_slice_count(total_value_usd, asset_in, depth_in)
        else:
            num_slices = max(int(num_slices), 1)

        slice_value_usd = total_value_usd / num_slices if num_slices else total_value_usd
        slice_amount_in = amount_in / num_slices if num_slices else amount_in

        gas_per_leg_eth = self.gas_per_leg.get(protocol, self.gas_per_leg["uniswap_v3"])
        gas_price_usd = self.gas_price_usd

        interval = timedelta(hours=duration_hours / num_slices)
        legs: List[SwapLeg] = []
        gas_breakdown: Dict[str, float] = {}
        total_out = 0.0
        total_impact = 0.0
        max_impact = 0.0
        leg_idx = 0

        for slice_idx in range(num_slices):
            slice_time = start_time + timedelta(hours=slice_idx * duration_hours / num_slices)
            for asset_out, alloc in normalized_allocs.items():
                if alloc <= 0:
                    continue

                amount_out_usd = slice_value_usd * alloc
                if amount_out_usd <= 0:
                    continue

                price_out = asset_prices[asset_out]
                if price_out <= 0:
                    continue

                depth_out = self.liquidity_depth.get(asset_out, 1_000_000.0)
                impact_in = self._estimate_market_impact(amount_out_usd, asset_in, depth_in)
                impact_out = self._estimate_market_impact(amount_out_usd, asset_out, depth_out)
                leg_impact = min(impact_in + impact_out, 1.0)

                amount_out = (amount_out_usd / price_out) * (1.0 - leg_impact)
                amount_in_leg = slice_amount_in * alloc
                gas_eth = gas_per_leg_eth

                legs.append(
                    SwapLeg(
                        index=leg_idx,
                        timestamp=slice_time,
                        asset_in=asset_in,
                        asset_out=asset_out,
                        amount_in=amount_in_leg,
                        expected_price=price_out,
                        expected_out=amount_out,
                        market_impact=leg_impact,
                        gas_cost_eth=gas_eth,
                    )
                )
                leg_idx += 1
                total_out += amount_out
                total_impact += leg_impact
                max_impact = max(max_impact, leg_impact)
                gas_breakdown[asset_out] = gas_breakdown.get(asset_out, 0.0) + gas_eth

        total_gas_eth = sum(gas_breakdown.values())
        total_gas_usd = total_gas_eth * gas_price_usd
        avg_impact = total_impact / len_legs if (len_legs := len(legs)) else 0.0
        schedule_end = start_time + timedelta(hours=duration_hours)

        return DiversificationPlan(
            legs=legs,
            total_amount_in=amount_in,
            total_expected_out=total_out,
            average_market_impact=avg_impact,
            max_market_impact=max_impact,
            total_gas_eth=total_gas_eth,
            total_gas_usd=total_gas_usd,
            gas_breakdown=gas_breakdown,
            schedule_start=start_time,
            schedule_end=schedule_end,
        )
