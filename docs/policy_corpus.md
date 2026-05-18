# Sentinel Policy Corpus

## Critical Risk Handling

Transactions classified as critical risk should be blocked before authorization when technically possible. The analyst case must include the model score, top risk signals, device information, and any retrieved policy context.

## High Risk Step-Up

High-risk transactions should trigger strong customer authentication. If authentication succeeds, the transaction may proceed with enhanced monitoring unless additional compliance rules require manual review.

## Medium Risk Monitoring

Medium-risk transactions can be approved with post-transaction monitoring. Analysts should sample cases for quality review and inspect repeated medium-risk behavior across related accounts.

## Device Trust Policy

Low device trust is a strong operational signal when paired with recent failed logins, new-device activity, or cross-border behavior. Device trust should not be used as the only reason for permanent account action.

## Cross-Border Review

Cross-border transactions require extra attention when merchant risk, country risk, and transaction amount are also elevated. The review should consider customer history and prior travel patterns.

## Drift Response

When feature drift exceeds the attention threshold, the model owner should compare recent traffic against the reference training population, inspect business changes, and decide whether retraining or threshold adjustment is needed.
