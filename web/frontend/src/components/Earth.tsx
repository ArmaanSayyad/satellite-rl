import { useFrame, useLoader } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

interface Props {
  radiusUnits: number;
}

// Real NASA Blue Marble / Black Marble imagery (day map, night lights,
// ocean specular, clouds) -- public-domain, sourced here from three.js's
// own official example assets. Purely visual: this project's physics
// works entirely in an Earth-Centered Inertial frame and never models
// Earth's rotational orientation or a real solar ephemeris (see
// satellite_rl.scenario.targeting), so the slow spin and fixed "sun"
// direction below are stylistic choices, not a simulated day/night cycle.
const SUN_DIRECTION = new THREE.Vector3(60, 20, 30).normalize();

const DAY_NIGHT_VERTEX_SHADER = `
  varying vec2 vUv;
  varying vec3 vWorldNormal;
  void main() {
    vUv = uv;
    vWorldNormal = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const DAY_NIGHT_FRAGMENT_SHADER = `
  uniform sampler2D dayMap;
  uniform sampler2D nightMap;
  uniform sampler2D specularMap;
  uniform vec3 sunDirection;
  varying vec2 vUv;
  varying vec3 vWorldNormal;

  void main() {
    vec3 normal = normalize(vWorldNormal);
    float sunDot = dot(normal, sunDirection);
    float dayMix = smoothstep(-0.15, 0.15, sunDot);

    vec3 dayColor = texture2D(dayMap, vUv).rgb;
    vec3 nightColor = texture2D(nightMap, vUv).rgb * 1.8;
    vec3 color = mix(nightColor * 0.7, dayColor, dayMix);

    float ocean = texture2D(specularMap, vUv).r;
    float rim = pow(max(sunDot, 0.0), 8.0) * ocean * dayMix;
    color += vec3(rim) * 0.5;

    gl_FragColor = vec4(color, 1.0);
  }
`;

const ATMOSPHERE_VERTEX_SHADER = `
  varying vec3 vViewNormal;
  void main() {
    vViewNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const ATMOSPHERE_FRAGMENT_SHADER = `
  varying vec3 vViewNormal;
  void main() {
    float intensity = pow(0.7 - dot(vViewNormal, vec3(0.0, 0.0, 1.0)), 3.5);
    gl_FragColor = vec4(0.35, 0.65, 1.0, 1.0) * clamp(intensity, 0.0, 1.0);
  }
`;

export default function Earth({ radiusUnits }: Props) {
  const [dayMap, nightMap, specularMap, cloudsMap] = useLoader(THREE.TextureLoader, [
    "/textures/earth_atmos_2048.jpg",
    "/textures/earth_lights_2048.png",
    "/textures/earth_specular_2048.jpg",
    "/textures/earth_clouds_1024.png",
  ]);

  const dayNightMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          dayMap: { value: dayMap },
          nightMap: { value: nightMap },
          specularMap: { value: specularMap },
          sunDirection: { value: SUN_DIRECTION },
        },
        vertexShader: DAY_NIGHT_VERTEX_SHADER,
        fragmentShader: DAY_NIGHT_FRAGMENT_SHADER,
      }),
    [dayMap, nightMap, specularMap]
  );

  const atmosphereMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: ATMOSPHERE_VERTEX_SHADER,
        fragmentShader: ATMOSPHERE_FRAGMENT_SHADER,
        blending: THREE.AdditiveBlending,
        side: THREE.BackSide,
        transparent: true,
      }),
    []
  );

  useEffect(
    () => () => {
      dayNightMaterial.dispose();
      atmosphereMaterial.dispose();
    },
    [dayNightMaterial, atmosphereMaterial]
  );

  const earthRef = useRef<THREE.Mesh>(null);
  const cloudsRef = useRef<THREE.Mesh>(null);

  useFrame((_state, delta) => {
    if (earthRef.current) earthRef.current.rotation.y += delta * 0.015;
    if (cloudsRef.current) cloudsRef.current.rotation.y += delta * 0.022;
  });

  return (
    <group>
      <mesh ref={earthRef} material={dayNightMaterial}>
        <sphereGeometry args={[radiusUnits, 64, 64]} />
      </mesh>

      <mesh ref={cloudsRef}>
        <sphereGeometry args={[radiusUnits * 1.006, 64, 64]} />
        <meshStandardMaterial map={cloudsMap} transparent opacity={0.75} depthWrite={false} />
      </mesh>

      <mesh scale={1.12} material={atmosphereMaterial}>
        <sphereGeometry args={[radiusUnits, 48, 48]} />
      </mesh>
    </group>
  );
}
