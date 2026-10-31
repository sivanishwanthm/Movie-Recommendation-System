# Browser verification

Playwright tests run the real Next.js/FastAPI/PostgreSQL flow with a fresh browser context per test. Two projects exercise desktop Chromium (1440×1000 CSS pixels) and Pixel 7 mobile emulation in Chromium. This is not a physical Android device or Safari/WebKit test.

## Coverage

- Create a profile; verify cold-start recommendations.
- Click a visible recommendation and wait for attributed navigation.
- Rate three recommended movies highly; verify personalized mode and rated-item exclusion.
- Reload the profile and confirm saved ratings and session-cookie properties.
- Change a rating; verify the current rating count stays fixed and the recommendation fallback changes appropriately.
- Check attributed clicks, visible impressions and rated-movie counts in Activity.
- Search for an absent title, simulate a 503 response in the browser, and recover with Retry.
- Load home, Explore, Models, Experiments, Profile, Activity and movie details on both viewports; check page-level overflow and uncaught JavaScript errors.
- Verify live enrollment is off and synthetic experiment results clearly state that they cannot support rollout conclusions.

The tests create synthetic profiles and events. **Run only against a disposable QA database or the isolated test stack.** Never aim them at your original profile database or production. The configuration requires `RECSYSLAB_E2E=1` and an explicit URL; neither is implied by the runner.

## Run locally

With Node installed and the isolated app already running:

```bash
cd frontend
npm ci
node node_modules/@playwright/test/cli.js install chromium
RECSYSLAB_E2E=1 E2E_BASE_URL=http://127.0.0.1:3001 npm run test:e2e
```

This workspace's bundled Node/browser installation can instead be used from the repository root:

```bash
RECSYSLAB_E2E=1 E2E_BASE_URL=http://127.0.0.1:3001 \
PLAYWRIGHT_BROWSERS_PATH="$PWD/.tools/playwright" \
.tools/node/bin/node frontend/node_modules/@playwright/test/cli.js test \
  --config frontend/playwright.config.ts
```

JSON results and screenshots are under ignored `frontend/test-results/`. Failures retain screenshots and traces for diagnosis. Successful navigation tests also save screenshots for visual inspection; this is not a screenshot-baseline comparison. Screenshots/traces can contain session information, so keep them limited to synthetic QA contexts. QA profiles remain in the disposable stack until it is reset; user-profile data is never migrated into that stack.

## CI and scope

The Compose CI job runs the same suite against its fresh localhost:3000 deployment and uploads `browser-qa` evidence with seven-day retention, then tears down the disposable stack. Browser checks run with no retries so a failure is visible. The test suite deliberately scopes error selectors to the main application, excluding Next.js's accessibility announcer, and waits for attributed navigation before performing subsequent actions.

Current scope does not establish Safari/Firefox compatibility, physical-device touch behavior, complete keyboard/screen-reader accessibility, production identity, or performance under load. See [observed results](../reports/browser-qa/results.md). Playwright's official [project/device documentation](https://playwright.dev/docs/test-projects) describes the emulation model.
