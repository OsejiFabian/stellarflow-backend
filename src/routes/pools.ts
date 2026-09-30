import { Router } from "express";
import { sendApiError } from "../lib/apiError";
import prisma from "../lib/prisma";
import {
  DEFAULT_FEE_TIER,
  FeeProjectionError,
  projectConcentratedLiquidityFees,
} from "../services/liquidity/feeProjection";

const router = Router();

type FeeTierSource = "request" | "pool" | "default";

/** Read a required positive number from a query string. */
function readPositiveQueryNumber(raw: unknown, field: string): number {
  if (typeof raw !== "string" || raw.trim() === "") {
    throw new FeeProjectionError(`${field} is required.`);
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new FeeProjectionError(`${field} must be a positive number.`);
  }
  return value;
}

/** Read an optional positive number from a query string. */
function readOptionalPositiveQueryNumber(
  raw: unknown,
  field: string,
): number | undefined {
  if (raw === undefined) return undefined;
  return readPositiveQueryNumber(raw, field);
}

/**
 * Choose the fee tier to project with.
 *
 * Priority: an explicit caller override, then the pool's *realised* effective
 * tier (`fees24h / volume24h`), then the 0.30% default. Deriving the tier from
 * observed fees keeps the projection correct when a pool's tier changes and
 * avoids hardcoding a per-pool constant that would silently drift.
 */
function resolveFeeTier(
  requested: number | undefined,
  volume24h: number,
  fees24h: number,
): { feeTier: number; feeTierSource: FeeTierSource } {
  if (requested !== undefined) {
    if (!(requested > 0 && requested < 1)) {
      throw new FeeProjectionError(
        "feeTier must be a decimal fraction between 0 and 1 (exclusive).",
      );
    }
    return { feeTier: requested, feeTierSource: "request" };
  }

  const realised = volume24h > 0 ? fees24h / volume24h : Number.NaN;
  if (Number.isFinite(realised) && realised > 0 && realised < 1) {
    return { feeTier: realised, feeTierSource: "pool" };
  }
  return { feeTier: DEFAULT_FEE_TIER, feeTierSource: "default" };
}

/**
 * @swagger
 * /api/v1/pools/{address}/project-fees:
 *   get:
 *     tags:
 *       - Pools
 *     summary: Project concentrated liquidity swap fee earnings
 *     description: >
 *       Estimates the swap fees a prospective LP position would earn over 24
 *       hours using S_fee = (L_user / L_total) * Volume_24h * Fee_tier.
 *       When `currentPrice` is supplied and falls outside the requested range
 *       the position is out of range and the projection is zero.
 *     parameters:
 *       - in: path
 *         name: address
 *         required: true
 *         schema: { type: string }
 *         description: Pool address / pool id
 *       - in: query
 *         name: liquidityAmount
 *         required: true
 *         schema: { type: number }
 *         description: Target liquidity the LP would provide (L_user)
 *       - in: query
 *         name: priceLower
 *         required: true
 *         schema: { type: number }
 *         description: Lower bound of the position price range (P_lower)
 *       - in: query
 *         name: priceUpper
 *         required: true
 *         schema: { type: number }
 *         description: Upper bound of the position price range (P_upper)
 *       - in: query
 *         name: currentPrice
 *         schema: { type: number }
 *         description: Current pool price used to decide whether the position is in range
 *       - in: query
 *         name: feeTier
 *         schema: { type: number }
 *         description: Fee tier override as a decimal (e.g. 0.003 for 0.30%)
 *     responses:
 *       '200':
 *         description: Fee projection computed
 *       '400':
 *         description: Invalid query parameters
 *       '404':
 *         description: Pool has no recorded liquidity or volume analytics
 *       '500':
 *         description: Internal server error
 */
router.get("/:address/project-fees", async (req, res) => {
  const address = req.params.address?.trim();
  if (!address) {
    sendApiError(
      res,
      400,
      "BAD_REQUEST",
      "A pool address path parameter is required.",
    );
    return;
  }

  try {
    const liquidityAmount = readPositiveQueryNumber(
      req.query.liquidityAmount,
      "liquidityAmount",
    );
    const priceLower = readPositiveQueryNumber(
      req.query.priceLower,
      "priceLower",
    );
    const priceUpper = readPositiveQueryNumber(
      req.query.priceUpper,
      "priceUpper",
    );
    const currentPrice = readOptionalPositiveQueryNumber(
      req.query.currentPrice,
      "currentPrice",
    );
    const requestedFeeTier = readOptionalPositiveQueryNumber(
      req.query.feeTier,
      "feeTier",
    );

    const [liquidityRow, volumeRow] = await Promise.all([
      prisma.poolLiquidity.findFirst({
        where: { poolId: address },
        orderBy: { timestamp: "desc" },
      }),
      prisma.poolVolumeAnalytics.findFirst({
        where: { poolId: address },
        orderBy: { timestamp: "desc" },
      }),
    ]);

    if (!liquidityRow || !volumeRow) {
      sendApiError(
        res,
        404,
        "POOL_NOT_FOUND",
        `No liquidity or volume analytics are recorded for pool '${address}'.`,
      );
      return;
    }

    const totalLiquidity = Number(liquidityRow.liquidity);
    const volume24h = Number(volumeRow.volume24h);
    const fees24h = Number(volumeRow.fees24h);

    const { feeTier, feeTierSource } = resolveFeeTier(
      requestedFeeTier,
      volume24h,
      fees24h,
    );

    const projection = projectConcentratedLiquidityFees({
      liquidityAmount,
      totalLiquidity,
      volume24h,
      feeTier,
      priceLower,
      priceUpper,
      ...(currentPrice === undefined ? {} : { currentPrice }),
    });

    res.json({
      success: true,
      data: {
        poolAddress: address,
        liquidityAmount,
        totalLiquidity,
        liquidityShare: projection.liquidityShare,
        volume24h,
        fees24h,
        feeTier,
        feeTierSource,
        priceLower,
        priceUpper,
        currentPrice: currentPrice ?? null,
        inRange: projection.inRange,
        projectedFees24h: projection.projectedFees24h,
        projectedFeesAnnualized: projection.projectedFeesAnnualized,
        projectedFeeAprPercent: projection.projectedFeeAprPercent,
        liquidityAsOf: liquidityRow.timestamp,
        volumeAsOf: volumeRow.timestamp,
      },
    });
  } catch (error) {
    if (error instanceof FeeProjectionError) {
      sendApiError(res, 400, "VALIDATION_ERROR", error.message);
      return;
    }
    console.error(
      "Error computing concentrated liquidity fee projection:",
      error,
    );
    sendApiError(res, 500, "INTERNAL_SERVER_ERROR");
  }
});

export default router;
