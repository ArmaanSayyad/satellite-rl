import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { useMemo } from "react";
import type { Frame } from "../types";
import { subtract } from "../physics";

interface Props {
  frame: Frame | null;
  denseFrames: Frame[];
  combinedRadiusM: number;
}

export default function EncounterView({ frame, denseFrames, combinedRadiusM }: Props) {
  // Real danger scales here run from tens to ~thousand meters -- nothing
  // like the 500km/unit scale the orbital view uses would show anything.
  // Auto-fit: find the largest relative separation across the whole
  // episode and size the scene/camera to it, so every scenario frames
  // itself sensibly regardless of its specific real miss distance.
  const { relativePath, maxSeparation } = useMemo(() => {
    let maxSep = combinedRadiusM;
    const path: [number, number, number][] = denseFrames.map((f) => {
      const rel = subtract(f.sec_r, f.ego_r);
      const sep = Math.hypot(rel[0], rel[1], rel[2]);
      if (sep > maxSep) maxSep = sep;
      return rel;
    });
    return { relativePath: path, maxSeparation: maxSep };
  }, [denseFrames, combinedRadiusM]);

  // Scale so the largest separation comfortably fills the view (~15
  // scene units), and convert everything else (danger sphere, current
  // marker) through the same real-meters-per-unit factor.
  const metersPerUnit = maxSeparation / 15;
  const toUnits = (m: number) => m / metersPerUnit;

  const scaledPath = useMemo<[number, number, number][]>(
    () => relativePath.map(([x, y, z]) => [toUnits(x), toUnits(y), toUnits(z)]),
    [relativePath, metersPerUnit]
  );

  const secPos: [number, number, number] = frame
    ? (() => {
        const rel = subtract(frame.sec_r, frame.ego_r);
        return [toUnits(rel[0]), toUnits(rel[1]), toUnits(rel[2])];
      })()
    : [scaledPath[0]?.[0] ?? 10, scaledPath[0]?.[1] ?? 0, scaledPath[0]?.[2] ?? 0];

  const dangerRadiusUnits = Math.max(toUnits(combinedRadiusM), 0.08);

  return (
    <div className="panel">
      <div className="panel-header">Encounter Geometry (relative to satellite)</div>
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

        {/* Other object's real relative trajectory */}
        {scaledPath.length > 1 && <Line points={scaledPath} color="#ffb84d" lineWidth={1.5} transparent opacity={0.75} />}

        <mesh position={secPos}>
          <sphereGeometry args={[0.22, 16, 16]} />
          <meshStandardMaterial color="#ffdca8" emissive="#ff8a3d" emissiveIntensity={0.9} />
        </mesh>

        <OrbitControls enablePan={false} minDistance={2} maxDistance={40} />
      </Canvas>
    </div>
  );
}
