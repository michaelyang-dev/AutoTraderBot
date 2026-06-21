# FULL-survivorship universe pull — the make-or-break test for whether crypto momentum's
# 126% is real or a survivorship mirage. Inactive coins have no date/mcap metadata in
# crypto2, so we pull a broad set (top-150 current + the ~450 NEWEST-id dead coins = the
# 2022-25 pump-and-die era + famous deaths) and let the backtest's point-in-time mcap
# filter decide which were ever top-N. If momentum holds with all these death spirals
# included, it's a real edge; if it collapses, the 126% was survivorship.
suppressMessages(library(crypto2))

all <- crypto_list(only_active = FALSE)
lat <- crypto_listings(which = "latest", convert = "USD", limit = 300, quote = TRUE)
lat <- lat[order(lat$cmc_rank), ]
STABLES <- c("USDT","USDC","DAI","BUSD","TUSD","USDD","FDUSD","USDE","PYUSD","USDP","USD1","FRAX","USDS","GUSD","LUSD")
lat <- lat[!lat$symbol %in% STABLES, ]
top <- head(lat, 150)$id

inact <- all[all$is_active == 0, ]
inact <- inact[order(inact$id), ]
dead_recent <- tail(inact$id, 450)                                  # newest-id = recent pump-and-die era
dead_syms <- c("LUNA","LUNC","UST","USTC","FTT","CEL","SRM","RAY","HNT","MIR","ANC","WAVES","SNT","TLM")
dead_curated <- all[all$symbol %in% dead_syms, ]$id

uni <- all[all$id %in% unique(c(top, dead_recent, dead_curated)), ]
uni <- uni[!duplicated(uni$id), ]
cat("universe:", nrow(uni), "coins ( top 150 + ~450 recent dead + curated )\n"); flush.console()

hist <- crypto_history(coin_list = uni, convert = "USD",
                       start_date = "20200101", end_date = "20260601",
                       interval = "1d", finalWait = FALSE)

keep <- intersect(c("timestamp", "id", "symbol", "close", "market_cap"), names(hist))
write.csv(hist[, keep], "data/crypto/crypto2_history_full.csv", row.names = FALSE)
cat("SAVED", nrow(hist), "rows,", length(unique(hist$symbol)), "coins → crypto2_history_full.csv\n")
