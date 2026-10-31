import {defineConfig, devices} from '@playwright/test';

// Explicit opt-in prevents accidentally writing synthetic activity to a real deployment.
if (process.env.RECSYSLAB_E2E !== '1') {
  throw new Error('Set RECSYSLAB_E2E=1 and E2E_BASE_URL to a disposable QA app.');
}
const baseURL = process.env.E2E_BASE_URL;
if (!baseURL) throw new Error('E2E_BASE_URL must identify the disposable QA app.');
export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: {timeout: 10_000},
  workers: 1,
  retries: 0,
  reporter: [['list'], ['json', {outputFile: 'test-results/results.json'}]],
  use: {baseURL, trace: 'retain-on-failure', screenshot: 'only-on-failure'},
  projects: [
    {name: 'desktop-chromium', use: {...devices['Desktop Chrome'], viewport: {width: 1440, height: 1000}}},
    {name: 'mobile-chromium', use: {...devices['Pixel 7']}},
  ],
});
