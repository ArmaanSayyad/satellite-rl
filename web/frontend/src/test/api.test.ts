import { afterEach, expect, it, vi } from "vitest";
import { createJob, loadReplay } from "../api";
afterEach(() => vi.unstubAllGlobals());
it("submits deterministic variant and radius settings through the versioned job API", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ id: "job-1", status: "running" }),
  });
  vi.stubGlobal("fetch", fetchMock);
  await createJob("kelvins-8767", "planner", 17, "posterior_variant", 0.5);
  expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
    scenario_id: "kelvins-8767",
    policy_id: "planner",
    seed: 17,
    variant: "posterior_variant",
    radius_scale: 0.5,
    schedule_mode: "controlled",
  });
  expect(fetchMock.mock.calls[0][0]).toBe("/api/v2/jobs");
});
it("rejects a malformed replay before playback can dereference missing trajectory data", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ dense_frames: [] }),
    }),
  );
  await expect(loadReplay("/replays/broken.json")).rejects.toThrow(
    "incomplete",
  );
});
