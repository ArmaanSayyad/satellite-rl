import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls, Stars } from "@react-three/drei";
import { Suspense, useMemo, useRef, useState } from "react";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import type { Frame } from "../types";
import { splitFramesAtTime } from "../physics";
import { useManeuverFlashToken } from "../useManeuverFlashToken";
import Earth from "./Earth";
import CameraFollow from "./CameraFollow";
import ManeuverFlash from "./ManeuverFlash";

// 1 scene unit = 500km. Chosen so Earth (radius 6378.1km) and a ~500km-
// altitude LEO orbit both render at a sensible, legible size.
const METERS_PER_UNIT = 500_000;
const toUnits = (m: number) => m / METERS_PER_UNIT;

// How far ahead of the current mission time to draw a faint trajectory
// preview, as a fraction of total mission duration -- keeps the preview
// proportionally sized whether the mission is 20 minutes or 5 hours.
const PREVIEW_FRACTION = 0.08;

interface Props {
  frame: Frame | null;
  missionTimeS: number;
  earthRadiusM: number;
  denseFrames: Frame[];
  comparisonFrames?: Frame[];
  maneuversSoFar: number;
}

export default function OrbitalContextView({
  frame,
  missionTimeS,
  earthRadiusM,
  denseFrames,
  comparisonFrames,
  maneuversSoFar,
}: Props) {
  const earthRadiusUnits = toUnits(earthRadiusM);
  const missionDurationS = denseFrames[denseFrames.length - 1]?.t_s ?? 0;
  const previewWindowS = missionDurationS * PREVIEW_FRACTION;
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const [follow, setFollow] = useState(false);
  const flashToken = useManeuverFlashToken(maneuversSoFar);
  const comparisonPath = useMemo<[number, number, number][]>(
    () =>
      (comparisonFrames ?? [])
        .filter((f) => f.t_s <= missionTimeS)
        .map((f) => [
          toUnits(f.ego_r[0]),
          toUnits(f.ego_r[1]),
          toUnits(f.ego_r[2]),
        ]),
    [comparisonFrames, missionTimeS],
  );

  const { traveled, preview } = useMemo(
    () => splitFramesAtTime(denseFrames, missionTimeS, previewWindowS),
    [denseFrames, missionTimeS, previewWindowS],
  );

  const egoTraveled = useMemo<[number, number, number][]>(
    () =>
      traveled.map((f) => [
        toUnits(f.ego_r[0]),
        toUnits(f.ego_r[1]),
        toUnits(f.ego_r[2]),
      ]),
    [traveled],
  );
  const egoPreview = useMemo<[number, number, number][]>(
    () =>
      preview.map((f) => [
        toUnits(f.ego_r[0]),
        toUnits(f.ego_r[1]),
        toUnits(f.ego_r[2]),
      ]),
    [preview],
  );
  const secTraveled = useMemo<[number, number, number][]>(
    () =>
      traveled.map((f) => [
        toUnits(f.sec_r[0]),
        toUnits(f.sec_r[1]),
        toUnits(f.sec_r[2]),
      ]),
    [traveled],
  );
  const secPreview = useMemo<[number, number, number][]>(
    () =>
      preview.map((f) => [
        toUnits(f.sec_r[0]),
        toUnits(f.sec_r[1]),
        toUnits(f.sec_r[2]),
      ]),
    [preview],
  );

  const egoPos: [number, number, number] = frame
    ? [
        toUnits(frame.ego_r[0]),
        toUnits(frame.ego_r[1]),
        toUnits(frame.ego_r[2]),
      ]
    : [earthRadiusUnits + 1, 0, 0];
  const secPos: [number, number, number] = frame
    ? [
        toUnits(frame.sec_r[0]),
        toUnits(frame.sec_r[1]),
        toUnits(frame.sec_r[2]),
      ]
    : egoPos;

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Orbital Context</span>
        <div className="panel-header-controls">
          <button
            className={`panel-btn ${follow ? "active" : ""}`}
            onClick={() => setFollow((f) => !f)}
          >
            {follow ? "Following ●" : "Follow satellite"}
          </button>
          <button
            className="panel-btn"
            onClick={() => controlsRef.current?.reset()}
          >
            Reset view
          </button>
        </div>
      </div>
      <Canvas
        camera={{ position: [32, 20, 32], fov: 42 }}
        fallback={
          <div className="loading">
            Orbital rendering needs WebGL. The encounter plane, charts, and
            recorded decisions remain available.
          </div>
        }
      >
        <ambientLight intensity={0.25} />
        <directionalLight position={[60, 20, 30]} intensity={1.8} />
        <Stars
          radius={250}
          depth={80}
          count={1600}
          factor={1.5}
          fade
          speed={0}
        />

        <Suspense
          fallback={
            <mesh>
              <sphereGeometry args={[earthRadiusUnits, 48, 48]} />
              <meshStandardMaterial color="#35483a" />
            </mesh>
          }
        >
          <Earth radiusUnits={earthRadiusUnits} />
        </Suspense>

        {denseFrames.length > 1 && (
          <Line
            points={denseFrames.map((f) => [
              toUnits(f.ego_r[0]),
              toUnits(f.ego_r[1]),
              toUnits(f.ego_r[2]),
            ])}
            color="#9fbea9"
            lineWidth={1}
            transparent
            opacity={0.18}
          />
        )}
        {comparisonPath.length > 1 && (
          <Line
            points={comparisonPath}
            color="#aaa7dc"
            lineWidth={2}
            transparent
            opacity={0.7}
          />
        )}

        {/* Ego satellite: already-flown path (bright), short preview ahead (faint) */}
        {egoTraveled.length > 1 && (
          <Line
            points={egoTraveled}
            color="#a3d7bd"
            lineWidth={1.6}
            transparent
            opacity={0.85}
          />
        )}
        {egoPreview.length > 1 && (
          <Line
            points={egoPreview}
            color="#4fd1ff"
            lineWidth={1}
            transparent
            opacity={0.18}
          />
        )}

        {/* Secondary object: same treatment, distinct color -- its real
            orbital path, not just the satellite's. */}
        {secTraveled.length > 1 && (
          <Line
            points={secTraveled}
            color="#ffb84d"
            lineWidth={1.6}
            transparent
            opacity={0.85}
          />
        )}
        {secPreview.length > 1 && (
          <Line
            points={secPreview}
            color="#ffb84d"
            lineWidth={1}
            transparent
            opacity={0.18}
          />
        )}

        {/* Live line connecting the two objects right now, so the
            encounter reads clearly even at orbital scale where the real
            gap between them is normally sub-pixel. */}
        {frame && (
          <Line
            points={[egoPos, secPos]}
            color="#ff5c5c"
            lineWidth={1}
            transparent
            opacity={0.5}
            dashed
            dashSize={0.4}
            gapSize={0.3}
          />
        )}

        <mesh position={egoPos}>
          <sphereGeometry args={[0.35, 16, 16]} />
          <meshStandardMaterial
            color="#e8f6ff"
            emissive="#4fd1ff"
            emissiveIntensity={0.8}
          />
        </mesh>
        <mesh position={secPos}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshStandardMaterial
            color="#ffdca8"
            emissive="#ff8a3d"
            emissiveIntensity={0.8}
          />
        </mesh>
        <ManeuverFlash
          trigger={flashToken}
          position={egoPos}
          baseRadius={0.5}
        />

        <OrbitControls
          ref={controlsRef}
          enablePan={false}
          minDistance={earthRadiusUnits + 3}
          maxDistance={140}
        />
        <CameraFollow
          controlsRef={controlsRef}
          target={egoPos}
          enabled={follow}
        />
      </Canvas>
    </div>
  );
}
