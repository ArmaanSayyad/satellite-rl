import { expect, test } from "@playwright/test";

test("live service job executes through the application and preserves cache labeling", async ({
  page,
}) => {
  const origin = process.env.APSIS_E2E_LIVE_ORIGIN;
  test.skip(!origin, "Optional real Basilisk service integration");
  test.setTimeout(120000);
  await page.route("**/api/v2/**", async (route) => {
    const url = new URL(route.request().url());
    const response = await route.fetch({
      url: `${origin}${url.pathname}${url.search}`,
      timeout: 110000,
    });
    await route.fulfill({ response });
  });
  await page.goto("/?event=8767&policy=never_maneuver");
  await expect(
    page.getByRole("button", { name: "Run live simulation" }),
  ).toBeVisible();
  await page.getByRole("spinbutton", { name: "Simulation seed" }).fill("33");
  await page
    .getByRole("combobox", { name: "Live radius scale" })
    .selectOption("0.5");
  await page.getByRole("button", { name: "Run live simulation" }).click();
  await expect(
    page.getByText(
      /LIVE RUN COMPLETE|Loaded a deterministic cached simulation/,
    ),
  ).toBeVisible({ timeout: 110000 });
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(
    page.getByRole("group", { name: "Replay policy" }).getByRole("button"),
  ).toHaveCount(1);
  await page
    .getByRole("button", { name: "Return to verified comparisons" })
    .click();
  await expect(
    page.getByRole("group", { name: "Replay policy" }).getByRole("button"),
  ).toHaveCount(5);
});
test("verified artifact opens, briefing teaches, timeline is keyboard accessible", async ({
  page,
}, testInfo) => {
  const pageErrors: string[] = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Room to maneuver." }),
  ).toBeVisible();
  await expect(page.getByText("VERIFIED REPLAY", { exact: true })).toBeVisible({
    timeout: 20000,
  });
  await page.waitForTimeout(3000);
  await page.screenshot({
    path: `/private/tmp/apsis-${testInfo.project.name}.png`,
    fullPage: true,
  });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(page.viewportSize()!.width);
  const overflow = await page.evaluate(() =>
    [...document.querySelectorAll("main *")]
      .filter((e) => e.getBoundingClientRect().right > window.innerWidth + 1)
      .slice(0, 12)
      .map((e) => ({
        tag: e.tagName,
        class: e.className,
        width: e.getBoundingClientRect().width,
        right: e.getBoundingClientRect().right,
        viewport: window.innerWidth,
      })),
  );
  expect(
    overflow.filter(
      (e) =>
        e.class === "workspace" ||
        e.class === "scenario-rail" ||
        e.class === "mission-body",
    ),
    JSON.stringify(overflow),
  ).toEqual([]);
  const seek = page.getByRole("slider", { name: "Mission time", exact: true });
  await seek.focus();
  await page.keyboard.press("End");
  await expect(seek).toHaveValue((await seek.getAttribute("max")) ?? "");
  await page.getByRole("button", { name: "No maneuver", exact: true }).click();
  await expect(seek).toHaveValue((await seek.getAttribute("max")) ?? "");
  await page
    .getByRole("button", { name: "Terminal simulator truth", exact: true })
    .click();
  await expect(page.getByText("Terminal truth", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "The briefing", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "A close approach is not a collision." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Next field note" }).click();
  await expect(
    page.getByRole("heading", { name: "Every warning has an uncertainty." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Research & evidence" }).click();
  await expect(
    page.getByRole("heading", { name: /Does the policy react/ }),
  ).toBeVisible();
  const evidenceHeading = page.locator(".experiment-evidence h2");
  await expect(evidenceHeading).toBeVisible();
  await evidenceHeading.scrollIntoViewIfNeeded();
  await page.screenshot({
    path: `/private/tmp/apsis-evidence-${testInfo.project.name}.png`,
  });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(page.viewportSize()!.width);
  expect(pageErrors).toEqual([]);
});
