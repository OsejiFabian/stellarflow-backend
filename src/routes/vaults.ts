import { Router } from "express";
import { VaultController } from "../controllers/vaultController";

const router = Router();
const vaultController = new VaultController();

router.get("/positions/:account_id", (req, res) => {
  vaultController.getPosition(req, res);
});

router.get("/auction-price", (req, res) => {
  vaultController.getAuctionPrice(req, res);
});

export default router;
