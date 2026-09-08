import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { EncounterPlane } from "../components/ResearchPlots";
import { ellipseGeometry } from "../evidence";
import { explainDecision } from "../decisionExplain";
import { frameAtTime } from "../physics";
import type { Decision, Encounter, Frame, Keyframe } from "../types";

const encounter: Encounter = {
  miss_vector_m: [10, 20],
  covariance_m2: [
    [5, 4],
    [4, 5],
  ],
  time_to_tca_s: 100,
};
describe("scientific evidence boundaries", () => {
  it("does not classify missing risk observations as safe", () => {
    const explanation = explainDecision(
      { action_magnitude_ms: 0.1 } as Decision,
      null,
      1e-4,
    );
    expect(explanation).toContain("not recorded");
    expect(explanation).not.toContain("below threshold");
  });
  it("preserves orientation and anisotropy for correlated covariance", () => {
    const geometry = ellipseGeometry(encounter);
    expect(geometry.major).toBeCloseTo(3);
    expect(geometry.minor).toBeCloseTo(1);
    expect(geometry.angle).toBeCloseTo(Math.PI / 4);
  });
  it("labels radius changes as visual only and supports keyboard adjustment", async () => {
    function Test() {
      const [factor, setFactor] = useState(1);
      return (
        <EncounterPlane
          encounter={encounter}
          radius={10}
          radiusFactor={factor}
          setRadiusFactor={setFactor}
        />
      );
    }
    render(<Test />);
    expect(screen.getByText(/recorded Pc is unchanged/)).toBeInTheDocument();
    const slider = screen.getByRole("slider", { name: /radius/ });
    slider.focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(slider).toHaveFocus();
    expect(screen.getByRole("img")).toHaveAccessibleName(/10.0, 20.0 meters/);
  });
  it("reports a below-threshold burn without inventing policy intention", () => {
    const d = { action_magnitude_ms: 0.5 } as Decision;
    const explanation = explainDecision(
      d,
      { pc_estimate: 1e-8 } as Keyframe,
      1e-4,
    );
    expect(explanation).toContain("below threshold");
    expect(explanation).toContain("rationale is unknown");
    expect(explanation).not.toMatch(/judged|waiting|worth/);
  });
  it("declares missing covariance instead of drawing a substitute", () => {
    render(
      <EncounterPlane
        radius={10}
        radiusFactor={1}
        setRadiusFactor={() => {}}
      />,
    );
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("Geometry not recorded")).toBeInTheDocument();
  });
  it("interpolates recorded frame boundaries and clamps playback outside the run", () => {
    const frame = (t: number): Frame => ({
      t_s: t,
      ego_r: [t, 0, 0],
      ego_v: [1, 0, 0],
      sec_r: [0, t, 0],
      sec_v: [0, 1, 0],
    });
    expect(frameAtTime([frame(0), frame(10)], 5).ego_r[0]).toBe(5);
    expect(frameAtTime([frame(0), frame(10)], -1).t_s).toBe(0);
    expect(frameAtTime([frame(0), frame(10)], 99).t_s).toBe(10);
  });
});
