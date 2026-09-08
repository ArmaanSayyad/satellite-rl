// Capture the running application, not a mockup. Requires frontend dependencies.
import { createRequire } from 'node:module';
import { mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const require = createRequire(new URL('../web/frontend/package.json', import.meta.url));
const { chromium } = require('playwright');
const output = fileURLToPath(new URL('../docs/screenshots/', import.meta.url));
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome' });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 }, reducedMotion: 'reduce' });
  await page.goto(process.env.APSIS_CAPTURE_ORIGIN || 'http://localhost:8000', { waitUntil: 'domcontentloaded' });
  await page.getByText('VERIFIED REPLAY', { exact: true }).waitFor();
  await page.locator('canvas').waitFor();
  await page.waitForTimeout(4000); // Let Earth textures settle before capturing.
  await page.screenshot({ path: `${output}/mission-control.jpg`, type: 'jpeg', quality: 90 });
  await page.getByRole('button', { name: 'Research & evidence' }).click();
  await page.locator('.experiment-evidence h2').scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${output}/research-evidence.jpg`, type: 'jpeg', quality: 90 });
  await page.getByRole('button', { name: 'Mission control', exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: `${output}/mobile-mission.jpg`, type: 'jpeg', quality: 90, fullPage: true });
  console.log(`Captured live application screenshots in ${output}`);
} finally {
  await browser.close();
}
