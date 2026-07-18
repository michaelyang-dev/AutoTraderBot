# Thread S — sector caps + vol-managed momentum — FINDINGS (deeper push)

**2026-07-15. Research-only.** Two theory-backed, previously-untested momentum risk controls.
Harness: `threadS_sector_volmom.py` + sector-cap/vol-managed-mom wired into livemirror. Full
stack (1.49x integer $50k), both periods, multi-start.

## Verdict: both null-to-negative on the both-period bar. Same structural pattern as Thread R.

### Sector-concentration cap (max K of top-5 momentum per 2-digit SIC — the book had NO sector control)
- max 2/SIC: **null** (short −0.1pp, long −0.0pp) — rarely binds; top-5 momentum seldom has
  >2 in one 2-digit SIC.
- max 1/SIC (force 5 different sectors): **mixed** — long +0.8pp CAGR/+0.02 Sharpe, but short
  −1.9pp CAGR/−0.04 Sharpe. Forcing diversification dilutes the concentrated momentum that IS
  the alpha in recent regimes (2020 tech). Not a both-period win.

### Vol-managed momentum (Barroso: scale mom sleeve by target/own-vol)
- t0.25: short −4.3pp CAGR (over-de-risks), long −0.8pp/+0.02 Sharpe.
- t0.35: short −2.3pp CAGR, long +0.1pp/+0.02 Sharpe.
- Marginal Sharpe help on LONG, consistent CAGR/Sharpe cost on SHORT. Not a both-period win.
- **Why it doesn't help here (it's a documented winner elsewhere):** the book ALREADY has TWO
  momentum-crash protections — the price-UMD crash detector (switches to defensive weights) and
  portfolio vol-scaling. Sleeve-specific vol-management is largely REDUNDANT with those, so its
  extra de-risking just costs CAGR in normal times.

## The structural pattern (across Threads B, C, R, S)
Every INTERNAL lever — reweighting sleeves, regime mix, quality (sleeve/filter), sector caps,
vol-managed momentum, hold-times, lever-up — is **null-to-negative on the both-period bar**.
The ONLY additions that work are signals ORTHOGONAL to what the equity book already sees:
the **credit de-risk gate** (external macro the book is blind to). The equity cross-section's
alpha (momentum, its regime timing, its crash protection) is already fully extracted; you
can't squeeze more by rearranging it. Gains require an orthogonal EXTERNAL signal.

Bonus (modeling): the credit gate is slightly BETTER live than backtested — the backtest
credits de-levered cash 0%, but the live IBKR account earns ~4-5% on idle cash during the
gate's de-risked periods. So real-world the gate's carry is positive, not zero.

## Net
Confirms the book is at its efficient frontier on all internal axes. The credit down-gate
remains the single validated, deployable addition. The next orthogonal frontier (if pursued)
is OTHER external macro-stress signals (funding/TED, treasury-vol, dollar) — but Thread B
already showed credit dominates and two-signal stacking (credit∧PC1) didn't beat credit alone,
so expected marginal.
