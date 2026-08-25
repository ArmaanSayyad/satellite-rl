import { useEffect, useRef, useState } from "react";

/** Bumps a counter each time maneuversSoFar increases -- 3D views watch
 * this to trigger a one-shot visual flash exactly when a burn fires
 * during playback (not on mount, not on rewind). */
export function useManeuverFlashToken(maneuversSoFar: number): number {
  const [token, setToken] = useState(0);
  const prevRef = useRef(maneuversSoFar);

  useEffect(() => {
    if (maneuversSoFar > prevRef.current) {
      setToken((t) => t + 1);
    }
    prevRef.current = maneuversSoFar;
  }, [maneuversSoFar]);

  return token;
}
