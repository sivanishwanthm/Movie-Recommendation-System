# Browser QA results

Local verification on October 31, 2026: **6 passed**, with three scenarios each in desktop Chromium and Pixel 7 Chromium emulation. Tests used the isolated Docker app at `127.0.0.1:3001`, not the original profile database. Frontend TypeScript checking also passed.

The complete profile/rating/recommendation flow passed on both viewports, including visible-card click attribution, persistence after reload, three-positive-rating personalization, rated-item exclusion, rating updates, fallback with fewer qualifying ratings, and Activity counts. Search empty-state handling and recovery from a browser-injected 503 passed. All seven main page types loaded without page-level horizontal overflow or uncaught JavaScript errors; the synthetic experiment tab remained clearly labeled and prohibited rollout conclusions.

Manual screenshot inspection covered the mobile home/profile form, mobile Activity, mobile synthetic experiment results, and desktop Models page. No visible overlaps or broken wrapping were found in those samples. Wide experiment tables remain inside horizontal scroll containers. Full-page screenshots and the runner's JSON output are saved locally under `frontend/test-results/`; CI uploads the same evidence as `browser-qa`.

The first run exposed test-harness issues, not confirmed application defects: a broad alert selector also matched Next.js's route announcer, and the mobile scenario navigated before the asynchronous click-attribution handler completed. Main-content selectors and explicit navigation synchronization fixed these; the final run passed without retries. No application UI changes were needed.

This establishes the tested Chromium flows and sampled layouts only. Physical mobile devices, Safari/Firefox, comprehensive accessibility, browser-restart/account recovery, load testing and public pilot hosting remain separate checks. Reload persistence was tested; it must not be described as browser-restart verification. Synthetic QA activity is not user engagement evidence.

Hosted verification also passed: [run 41233023754](https://github.com/sivanishwanthm/Movie-Recommendation-System/actions/runs/41233023754) on commit `55a45b99ff499611f87bf0b227a6ac24d1f46628`. It ran **53 Python tests** (15 warnings), frontend checks, the fresh Compose deployment and **6 browser tests** (17.3 seconds). The `browser-qa` artifact contains screenshots/results with seven-day retention. See [ci-verification.json](ci-verification.json).
