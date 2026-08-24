import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls, Stars } from "@react-three/drei";
import { useMemo } from "react";
import type { Frame } from "../types";

// 1 scene unit = 500km. Chosen so Earth (radius 6378.1km) and a ~500km-
// altitude LEO orbit both render at a sensible, legible size.
const METERS_PER_UNIT = 500_000;
const toUnits = (m: number) => m / METERS_PER_UNIT;

interface Props {
  frame: Frame | null;
  earthRadiusM: number;
  denseFrames: Frame[];
}

export default function OrbitalContextView({ frame, earthRadiusM, denseFrames }: Props) {
  const earthRadiusUnits = toUnits(earthRadiusM);

  const orbitPath = useMemo<[number, number, number][]>(
    () => denseFrames.map((f) => [toUnits(f.ego_r[0]), toUnits(f.ego_r[1]), toUnits(f.ego_r[2])]),
    [denseFrames]
  );

  const egoPos: [number, number, number] = frame
    ? [toUnits(frame.ego_r[0]), toUnits(frame.ego_r[1]), toUnits(frame.ego_r[2])]
    : [earthRadiusUnits + 1, 0, 0];

  return (
    <div className="panel">
      <div className="panel-header">Orbital Context</div>
      <Canvas camera={{ position: [40, 25, 40], fov: 45 }}>
        <ambientLight intensity={0.35} />
        <directionalLight position={[60, 20, 30]} intensity={1.6} />
        <Stars radius={200} depth={60} count={3000} factor={2} fade speed={0.3} />

        {/* Earth */}
        <mesh>
          <sphereGeometry args={[earthRadiusUnits, 48, 48]} />
          <meshStandardMaterial color="#1b4f8a" roughness={0.85} metalness={0.1} />
        </mesh>
        <mesh>
          <sphereGeometry args={[earthRadiusUnits * 1.001, 48, 48]} />
          <meshBasicMaterial color="#4da3ff" wireframe transparent opacity={0.12} />
        </mesh>

        {/* Ego satellite full orbit path */}
        {orbitPath.length > 1 && <Line points={orbitPath} color="#4fd1ff" lineWidth={1.2} transparent opacity={0.55} />}

        {/* Ego satellite marker */}
        <mesh position={egoPos}>
          <sphereGeometry args={[0.35, 16, 16]} />
          <meshStandardMaterial color="#e8f6ff" emissive="#4fd1ff" emissiveIntensity={0.8} />
        </mesh>

        <OrbitControls enablePan={false} minDistance={earthRadiusUnits + 3} maxDistance={120} />
      </Canvas>
    </div>
  );
}
