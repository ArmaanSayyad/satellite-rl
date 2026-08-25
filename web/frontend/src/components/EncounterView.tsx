import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { useMemo, useRef } from "react";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import type { Frame } from "../types";
import { subtract, splitFramesAtTime } from "../physics";
import { useManeuverFlashToken } from "../useManeuverFlashToken";
import ManeuverFlash from "./ManeuverFlash";

const PREVIEW_FRACTION = 0.08;
const EGO_ORIGIN: [number, number, number] = [0, 0, 0];

interface Props {
  frame: Frame | null;
  missionTimeS: number;
  denseFrames: Frame[];
  combinedRadiusM: number;
  maneuversSoFar: number;
}

export default function EncounterView({ frame, missionTimeS, denseFrames, combinedRadiusM, maneuversSoFar }: Props) {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const flashToken = useManeuverFlashToken(maneuversSoFar);

  // Real danger scales here run from tens to ~thousand meters -- nothing
  // like the 500km/unit scale the orbital view uses would show anything.
  // Auto-fit: find the largest relative separation across the whole
  // episode and size the scene/camera to it, so every scenario frames
  // itself sensibly regardless of its specific real miss distance.
  const maxSeparation = useMemo(() => {
    let maxSep = combinedRadiusM;
    for (const f of denseFrames) {
      const rel = subtract(f.sec_r, f.ego_r);
      const sep = Math.hypot(rel[0], rel[1], rel[2]);
      if (sep > maxSep) maxSep = sep;
    }
    return maxSep;
  }, [denseFrames, combinedRadiusM]);

  // Scale so the largest separation comfortably fills the view (~15
  // scene units), and convert everything else (danger sphere, current
  // marker) through the same real-meters-per-unit factor.
  const metersPerUnit = maxSeparation / 15;
  const toUnits = (m: number) => m / metersPerUnit;

  const missionDurationS = denseFrames[denseFrames.length - 1]?.t_s ?? 0;
  const previewWindowS = missionDurationS * PREVIEW_FRACTION;
  const { traveled, preview } = useMemo(
    () => splitFramesAtTime(denseFrames, missionTimeS, previewWindowS),
    [denseFrames, missionTimeS, previewWindowS]
  );

  const traveledPath = useMemo<[number, number, number][]>(
    () =>
      traveled.map((f) => {
        const rel = subtract(f.sec_r, f.ego_r);
        return [rel[0] / metersPerUnit, rel[1] / metersPerUnit, rel[2] / metersPerUnit];
      }),
    [traveled, metersPerUnit]
  );
  const previewPath = useMemo<[number, number, number][]>(
    () =>
      preview.map((f) => {
        const rel = subtract(f.sec_r, f.ego_r);
        return [rel[0] / metersPerUnit, rel[1] / metersPerUnit, rel[2] / metersPerUnit];
      }),
    [preview, metersPerUnit]
  );

  const secPos: [number, number, number] = frame
    ? (() => {
        const rel = subtract(frame.sec_r, frame.ego_r);
        return [toUnits(rel[0]), toUnits(rel[1]), toUnits(rel[2])];
      })()
    : (traveledPath[0] ?? [10, 0, 0]);

  const dangerRadiusUnits = Math.max(toUnits(combinedRadiusM), 0.08);

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Encounter Geometry (relative to satellite)</span>
        <div className="panel-header-controls">
          <button className="panel-btn" onClick={() => controlsRef.current?.reset()}>
            Reset view
          </button>
        </div>
      </div>
      <Canvas camera={{ position: [12, 8, 12], fov: 45 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 10]} intensity={1.2} />

        {/* Our satellite, fixed at the origin of this relative frame */}
        <mesh>
          <sphereGeometry args={[0.28, 20, 20]} />
          <meshStandardMaterial color="#e8f6ff" emissive="#4fd1ff" emissiveIntensity={0.9} />
        </mesh>

        {/* Real physical collision threshold -- if the other object's
            center enters this sphere, it's a collision, per the
            combined hard-body radius reported for this scenario. */}
        <mesh>
          <sphereGeometry args={[dangerRadiusUnits, 24, 24]} />
          <meshBasicMaterial color="#ff4d4d" wireframe transparent opacity={0.35} />
        </mesh>

        {/* Other object's real relative trajectory: already-flown (bright)
            plus a short preview ahead (faint), not the whole outcome
            revealed up front. */}
        {traveledPath.length > 1 && (
          <Line points={traveledPath} color="#ffb84d" lineWidth={1.8} transparent opacity={0.85} />
        )}
        {previewPath.length > 1 && (
          <Line points={previewPath} color="#ffb84d" lineWidth={1.2} transparent opacity={0.2} />
        )}

        {/* Live line connecting the two objects right now */}
        {frame && (
          <Line
            points={[EGO_ORIGIN, secPos]}
            color="#ff5c5c"
            lineWidth={1}
            transparent
            opacity={0.5}
            dashed
            dashSize={0.15}
            gapSize={0.1}
          />
        )}

        <mesh position={secPos}>
          <sphereGeometry args={[0.22, 16, 16]} />
          <meshStandardMaterial color="#ffdca8" emissive="#ff8a3d" emissiveIntensity={0.9} />
        </mesh>
        <ManeuverFlash trigger={flashToken} position={EGO_ORIGIN} baseRadius={0.4} />

        <OrbitControls ref={controlsRef} enablePan={false} minDistance={2} maxDistance={40} />
      </Canvas>
    </div>
  );
}
