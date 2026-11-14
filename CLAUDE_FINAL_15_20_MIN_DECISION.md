# Claude: final 15–20 minutes — upload and preserve the best result

## Accepted evidence: do not repeat completed work

- Best confirmed public score: **V3 = 0.982564**.
- Uploaded fr08: **0.982529**, below V3.
- Neural-anchor experiment is complete and rejected: paired eval **−0.000005**, CI
  **[−0.000027, +0.000016]**; ES below V3s; US worse; only 2,377 French S1s changed.
- **Do not rerun anchors, seed searches, calibration multipliers or speculative new models.**
- V3s support has a reported small measured validation gain, approximately **+0.000055**.
  Its test gain is unknown. The estimate 0.98256–0.98265 is not an observed score or guarantee.

**The remaining gap to 0.99 is 0.007436. No completed experiment supports that gain.
There is no evidence-backed way to guarantee it in this window. The strongest next
action is to use the last opportunity correctly, not invent another unvalidated model.**

## Minute 0–3: resolve portal rules before touching the last attempt

1. Confirm the number of remaining attempts, scoring turnaround and hard deadline.
2. Check the portal's documented final-selection behavior: best public score, last
   upload, or explicit selection of a prior submission. Do not infer it from habit.
3. Confirm whether the previously uploaded fr08 has displaced V3 as the selected file.
   **If the portal keeps the last upload, doing nothing may retain fr08, not V3.**
4. Record which exact submission is currently selected and whether that can be changed
   without consuming an upload. Public-best selection does not guarantee private-best
   performance, but it preserves the strongest measured public evidence.

## Minute 3–5: verify the one available new candidate

Candidate:

`submissions/SUB-FINAL-V3s_support/matching_results.tsv`

Expected MD5:

`078eed8c2b10e38d2845392f5ff23bcf`

- Confirm the local file hash and the official validator report, including `--check-ids`.
- Confirm the reported paired gain belongs to this exact model/output lineage, and
  identify its comparator. Do not attach another variant's gain or confidence interval.
- Confirm that inference is complete for all countries and all 1,732,544 S1s.
- Preserve the scored V3 unchanged; its matching MD5 is
  `fffea65157b36546bbf566889ce23ca4`.
- Do not retrain or modify thresholds while preparing the upload.

## Minute 5–10: make the final upload decision

| Verified portal behavior | Action |
|---|---|
| Best scored submission retained automatically | Upload validated V3s support; wait for actual score; verify V3 remains available if it loses. |
| Any previous submission can be selected explicitly | Upload V3s support if time permits scoring; select based on actual results, retaining V3 as fallback. |
| Last upload is final, with no rollback/selection | Do not claim V3s is safer or better on test. Default to preserving/restoring V3 using the available selection or re-upload mechanism. Check whether fr08 is currently last. |
| Rules cannot be verified promptly | Do not spend the last attempt blindly. Ask the team to inspect the portal and prioritize a valid selected submission before the deadline. |

If a last-upload-only portal would consume the last attempt on V3s, that is an
irreversible test-score gamble with only a tiny validation advantage. Make that tradeoff
explicit to the user; do not present it as a route likely to reach 0.99.

Record the actual returned score and submission ID. Do not round a prediction into a
claimed result. Do not wait past the package/upload buffer for delayed scoring.

## Minute 10–15: make the package match the selected submission

Expected arriving package:

`submissions/FINAL_PACKAGE_V3s/team_submission.zip`

1. Confirm copying has actually finished; verify the archive against the producer's
   checksum and test archive integrity before uploading.
2. Inspect the matching file **inside the zip**. Its hash must equal the exact selected
   final matching file. A V3s package is not automatically the right package if V3 is selected.
3. Confirm that the candidate file is the **full exact scored candidate set**, with every
   final match included. A top-15 post-score export is insufficient provenance.
4. Confirm code, models/ensemble configuration or reproduction instructions, dependency
   pins and documentation describe that selected variant. Do not replace only the TSV
   while leaving a different model and gate described as its producer.
5. If V3 remains selected, use or prepare the matching V3 package with its proper
   candidate provenance. Do not silently submit a V3s package against V3 portal predictions.
6. Submit the correct package and verify the portal acknowledges successful completion.

## Minute 15–20: verify receipts, not more experiments

- Verify final selected submission ID, actual public score, matching MD5, zip checksum
  and successful package receipt.
- Keep V3 and all immutable scored files, validation reports and reproduction artifacts.
- Update the handoff with the failed anchor result so nobody repeats it.
- Record the final measured score honestly. **0.99 remains unachieved unless the portal
  actually reports it.** Do not sacrifice the valid submission chasing an unsupported claim.

## Exact instruction to Claude

> Do not repeat the completed anchor experiment or launch another model. It failed its
> gate. We have at most 15–20 minutes. Verify remaining attempts and whether the portal
> uses best, last, or explicit selection. Check whether fr08 is currently selected:
> on a last-upload portal, inaction may keep fr08 rather than V3. Verify V3s support
> matching MD5 078eed8c2b10e38d2845392f5ff23bcf and its full validator report. If the
> portal safely retains/allows selecting V3, use the remaining attempt on V3s support
> and select using actual feedback. If no rollback exists, default to preserving or
> restoring measured-best V3; make any gamble explicit. Verify the arriving zip is
> complete and that its internal matching file, full candidates, code and documentation
> correspond to the exact final selection. Upload the matching package before the
> deadline and record receipts/hashes. No experiment has demonstrated a route to
> +0.007436 in this window; do not promise 0.99 or mislabel validation as a portal gain.
