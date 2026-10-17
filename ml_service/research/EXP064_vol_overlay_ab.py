"""A/B on the DEPLOYED package: vol-scaling ON (live) vs OFF, and a gentler floor (0.60). 24 starts, EXP-059 harness."""
import sys
HZ = sys.argv[1]
sys.argv = ["x", HZ, "stage2", "pkg_noVol", "pkg_floor0.60"]
import EXP059_frontier as X
pkg = X.ARMS["cap0.15_package"]
X.ARMS["pkg_noVol"] = dict(pkg, vol_overlay=False)        # never de-risks: constant 1.49x x credit gate
X.ARMS["pkg_floor0.60"] = dict(pkg, vs_floor=0.60)        # de-risks at most to 0.60 x (live floor 0.30)
X.main()
