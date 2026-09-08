import { useFrame } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import * as THREE from "three";

interface Props {
  trigger: number;
  position: [number, number, number];
  baseRadius?: number;
  color?: string;
}

const FLASH_DURATION_S = 0.6;

/** A brief expanding, fading ring at `position`, replayed once each time
 * `trigger` changes -- visual feedback for the instant a maneuver fires,
 * decoupled from mission-time (always plays at the same real wall-clock
 * pace regardless of playback speed). */
export default function ManeuverFlash({
  trigger,
  position,
  baseRadius = 0.4,
  color = "#ffde59",
}: Props) {
  const meshRef = useRef<THREE.Mesh>(null);
  const startRef = useRef<number | null>(null);
  const isFirstRender = useRef(true);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    startRef.current = performance.now() / 1000;
  }, [trigger]);

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    if (startRef.current === null) {
      mesh.visible = false;
      return;
    }
    const elapsed = performance.now() / 1000 - startRef.current;
    if (elapsed > FLASH_DURATION_S) {
      mesh.visible = false;
      startRef.current = null;
      return;
    }
    const t = elapsed / FLASH_DURATION_S;
    mesh.visible = true;
    mesh.position.set(position[0], position[1], position[2]);
    mesh.scale.setScalar(baseRadius * (1 + t * 5));
    const material = mesh.material as THREE.MeshBasicMaterial;
    material.opacity = 1 - t;
  });

  return (
    <mesh ref={meshRef} visible={false}>
      <sphereGeometry args={[1, 16, 16]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0}
        depthWrite={false}
      />
    </mesh>
  );
}
