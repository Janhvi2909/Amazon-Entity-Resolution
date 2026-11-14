# Independent continuation audit — 2026-09-27 ~22:08 IST

The earlier execution plan is now partly implemented by the other session. Do not
repeat the M2 fusion work or launch overlapping GPU jobs.

## Verified current results

- Local `SUB-FINAL-V3b_es_protocol.stack_eval.json`: validation macro F0.5
  **0.9896440163**, India 0.9883473692, US 0.9905084607. Gate .65 selected on ES;
  485 stacker trees. Its recorded candidate export is still 15 per S1.
- Completed job `ber-final-c`: v3c exit 0 at 22:04:50 IST, peak container memory
  54,108,880,896 bytes. It exports **115.7093 candidates per S1**, consistent with
  the full M2 scored set rather than top 15. Output directory:
  `/data/ber/logs/ber-final-c/v3c`.
- V3c is **not identical to V3b**: 468 trees, ES-selected gate .70, 5,750,738 test
  links versus V3b's 5,750,745. Do not pair V3c metadata/model with V3b matching
  output or claim a byte-identical reproduction. Full-output validation still needs
  checking with the owning session.
- No new leaderboard score has been confirmed to this session. User was asked which
  variants were uploaded, their scores, and remaining attempts. Until answered,
  0.979604 remains the last confirmed public score here.

## V3b test changes versus SUB-N3

| Country | Added links | Removed links | Changed S1 entities |
|---|---:|---:|---:|
| US | 10,095 | 1,888 | 11,513 |
| India | 22,085 | 2,471 | 22,910 |
| France | 12,550 | 18,821 | 29,097 |

Total changed S1s: **63,520 / 1,732,544 = approximately 3.666%**. Unlike SUB-R,
this has enough changed entities for a +.01 gain to be mathematically possible,
but this provides no evidence that those changes are correct. France changes
affect about 11.2% of its S1s and remove more links than they add. Raw similarities
cannot establish whether those removals fix false positives or introduce misses.

Full label-free comparison and raw samples:
`analysis/score_audit/final_v3b_changes/`.

## Next action

1. Obtain actual portal feedback before choosing another architecture or upload.
2. Verify the owner's full V3c export and validation; preserve output/model/hash identity.
3. If the reviewed-France variant regresses, compare a country-spliced alternative
   using improved US/India and frozen SUB-N3 France, rather than guessing from
   address/name similarity alone.
4. Keep the 22:45 stop-new-heavy-work and 23:35 upload-buffer checkpoints.
