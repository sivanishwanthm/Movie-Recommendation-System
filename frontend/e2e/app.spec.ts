import {test, expect, Page} from '@playwright/test';

async function createProfile(page: Page) {
  await page.goto('/');
  await page.getByLabel('What should we call you?').fill('Synthetic browser QA');
  await page.getByRole('button', {name: 'Create my profile'}).click();
  await expect(page.getByText('Popularity · getting to know you', {exact: true})).toBeVisible();
}

async function noOverflow(page: Page) {
  const widths = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    content: document.documentElement.scrollWidth,
  }));
  expect(widths.content).toBeLessThanOrEqual(widths.viewport + 1);
}

test('ratings personalize picks, persist, and generate attributable activity', async ({page, context}, info) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await createProfile(page);
  const links = page.locator('a.movie-cover');
  await expect(links).toHaveCount(12);
  const hrefs = await links.evaluateAll(elements => elements.slice(0, 3).map(e => e.getAttribute('href')!));
  // First navigation must come from a visible recommendation to exercise attribution.
  await links.first().click();
  await page.waitForURL(`**${hrefs[0]}`);
  for (const href of hrefs) {
    if (!page.url().endsWith(href)) await page.goto(href);
    await page.getByLabel('Your rating', {exact: true}).selectOption('5');
    await page.getByRole('button', {name: 'Save rating', exact: true}).click();
    await expect(page.getByText('Saved: 5 / 5.', {exact: false})).toBeVisible();
  }
  await page.goto('/');
  await expect(page.getByText('Personalized · item CF', {exact: true})).toBeVisible();
  await expect(links).toHaveCount(12);
  for (const href of hrefs) await expect(page.locator(`a.movie-cover[href="${href}"]`)).toHaveCount(0);
  await page.goto('/profile');
  await expect(page.locator('.rating-list a')).toHaveCount(3);
  await page.reload();
  await expect(page.getByLabel('Display name')).toHaveValue('Synthetic browser QA');
  await expect(page.locator('.rating-list a')).toHaveCount(3);
  const cookie = (await context.cookies()).find(c => c.name === 'recsys_session');
  expect(cookie?.httpOnly).toBe(true);
  expect(cookie?.sameSite).toBe('Strict');
  // Updating an existing rating must keep one current rating per movie.
  await page.goto(hrefs[0]);
  await page.getByLabel('Your rating', {exact: true}).selectOption('3');
  await page.getByRole('button', {name: 'Save rating', exact: true}).click();
  await expect(page.getByText('Saved: 3 / 5.', {exact: false})).toBeVisible();
  await page.goto('/');
  await expect(page.getByText('Popularity · getting to know you', {exact: true})).toBeVisible();
  await page.goto('/activity');
  await expect(page.locator('.stat').filter({hasText: 'Movies rated'}).locator('strong')).toHaveText('3');
  await expect(page.locator('.stat').filter({hasText: 'Recommendation clicks'}).locator('strong')).toHaveText('1');
  expect(Number(await page.locator('.stat').filter({hasText: 'Visible cards'}).locator('strong').innerText())).toBeGreaterThan(0);
  await noOverflow(page);
  await page.screenshot({path: info.outputPath('activity.png'), fullPage: true});
  expect(errors).toEqual([]);
});

test('search empty state, service failure and retry recover', async ({page}) => {
  await page.goto('/explore');
  await page.getByLabel('Search movie titles').fill('ZZZ_NO_MATCH_BROWSER_QA_98421');
  await page.getByRole('button', {name: 'Search'}).click();
  await expect(page.getByText('No titles matched.', {exact: false})).toBeVisible();
  await page.route('**/api/movies?**', route => route.fulfill({status: 503, contentType: 'application/json', body: JSON.stringify({detail: 'Synthetic service interruption'})}));
  await page.getByLabel('Search movie titles').fill('Toy Story');
  await page.getByRole('button', {name: 'Search'}).click();
  await expect(page.locator('main').getByRole('alert')).toContainText('Synthetic service interruption');
  await page.unroute('**/api/movies?**');
  await page.getByRole('button', {name: 'Retry', exact: true}).click();
  await expect(page.getByRole('link', {name: 'View Toy Story (1995)', exact: true})).toBeVisible();
  await expect(page.locator('main').getByRole('alert')).toHaveCount(0);
});

test('navigation and responsive layouts remain usable', async ({page}, info) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  for (const [path, name] of [['/', 'home'], ['/explore', 'explore'], ['/models', 'models'], ['/experiments', 'experiments'], ['/profile', 'profile'], ['/activity', 'activity'], ['/movies/1', 'detail']]) {
    await page.goto(path);
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.getByRole('navigation', {name: 'Main navigation'})).toBeVisible();
    await expect(page.getByRole('navigation').getByRole('link', {name: 'Profile', exact: true})).toBeVisible();
    // Wait for fetched content, including wide result tables, before measuring layout.
    await page.waitForLoadState('networkidle');
    await noOverflow(page);
    await page.screenshot({path: info.outputPath(`${name}.png`), fullPage: true});
    if (path === '/experiments') {
      await expect(page.getByText('LIVE ENROLLMENT IS OFF', {exact: true})).toBeVisible();
      await page.getByRole('button', {name: 'Synthetic diagnostics', exact: true}).click();
      await expect(page.getByText('SYNTHETIC · GENERATED OUTCOMES', {exact: true})).toBeVisible();
      await expect(page.getByRole('heading', {name: 'No rollout conclusion', exact: true})).toBeVisible();
      await noOverflow(page);
      await page.screenshot({path: info.outputPath('experiments-synthetic.png'), fullPage: true});
    }
  }
  expect(errors).toEqual([]);
});
