"""Shared, torch-independent legal instantaneous lane/type patterns."""
STATE_NAMES=('none','light','heavy','extra_heavy','gear','gemini')
PATTERNS=tuple((g,a) for g in range(5) for a in range(5)
               if (g or a) and not (g==3 and a!=0 or a==3 and g!=0))+((5,5),)
