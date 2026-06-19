# Gold-standard survivorship-free crypto universe via crypto2 (full CoinMarketCap).
# crypto_history works (validated: FTT→$0.84, LUNA death captured). crypto_listings(limit=)
# is buggy (returns all 5001) so we head() it; historical-listings is CMC-gated.
# Universe = top-N current (excl. stables) + a curated list of the famous dead/crashed
# coins → survivorship-aware. Pulls daily close + market cap for a point-in-time backtest.
suppressMessages(library(crypto2))

all <- crypto_list(only_active = FALSE)
lat <- crypto_listings(which = "latest", convert = "USD", limit = 200, quote = TRUE)
lat <- lat[order(lat$cmc_rank), ]
STABLES <- c("USDT","USDC","DAI","BUSD","TUSD","USDD","FDUSD","USDE","PYUSD","USDP","USD1","FRAX","USDS","GUSD","LUSD")
lat <- lat[!lat$symbol %in% STABLES, ]
top <- head(lat, 110)$id

# curated survivorship-critical dead/crashed coins (Terra, FTX, Celsius eras, etc.)
dead_syms <- c("LUNA","LUNC","UST","USTC","FTT","CEL","SRM","RAY","HNT","MIR","ANC","WAVES",
               "STEP","SCRT","TLM","SLP","BICO","REEF","TOMO","CTSI","KEEP","NU","OMG","BTG")
dead_ids <- all[all$symbol %in% dead_syms, ]$id

uni <- all[all$id %in% unique(c(top, dead_ids)), ]
uni <- uni[!duplicated(uni$id), ]
cat("universe:", nrow(uni), "coins ( top", length(top), "+ curated dead", length(dead_ids), ")\n"); flush.console()

hist <- crypto_history(coin_list = uni, convert = "USD",
                       start_date = "20200101", end_date = "20260601",
                       interval = "1d", finalWait = FALSE)

keep <- intersect(c("timestamp", "id", "symbol", "name", "close", "volume", "market_cap"), names(hist))
write.csv(hist[, keep], "data/crypto/crypto2_history.csv", row.names = FALSE)
cat("SAVED", nrow(hist), "rows,", length(unique(hist$symbol)), "coins → crypto2_history.csv\n")
