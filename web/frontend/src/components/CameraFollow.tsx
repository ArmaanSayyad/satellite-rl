import { useFrame } from "@react-three/fiber";
import type { RefObject } from "react";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

interface Props {
  controlsRef: RefObject<OrbitControlsImpl | null>;
  target: [number, number, number];
  enabled: boolean;
}

/** Re-centers OrbitControls' orbit target on `target` every frame while
 * `enabled` -- camera keeps the same relative angle/zoom the user chose,
 * just recentered on the moving satellite instead of the origin. */
export default function CameraFollow({ controlsRef, target, enabled }: Props) {
  useFrame(() => {
    if (!enabled) return;
    const controls = controlsRef.current;
    if (!controls) return;
    controls.target.set(target[0], target[1], target[2]);
    controls.update();
  });
  return null;
}
