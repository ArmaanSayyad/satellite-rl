# Apsis application

From the repository root, `npm run dev` installs the frontend lockfile and opens a
Vite development server. Node 24.15+ is required; Node 24.20 is tested. Verified Basilisk
examples are committed under `web/frontend/public/replays/` and require no backend.

For a built application and API, use `docker compose up --build`, then visit
http://localhost:8000. For live physics, see [deployment](../docs/DEPLOYMENT.md).
The old blocking `/api/simulate?seed=` API has been replaced by `/api/v2/jobs`.

```sh
cd web/frontend
npm run lint
npm run test
npm run build
npx playwright install chromium
npm run test:e2e
```

The interface distinguishes recorded historical geometry, generated orbital scenarios,
noisy policy observations, simulator outcomes and cached replays. Orbit samples are
actual spacecraft message recordings; smooth browser playback interpolates between
them. The Earth globe is contextual artwork, not an Earth-fixed coordinate measurement.

Earth texture assets were inherited from the original demo and depict NASA Blue Marble
imagery. NASA attribution and imagery usage: [Visible Earth](https://visibleearth.nasa.gov/)
and [NASA imagery guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/).
These imagery assets are separate from the ESA CC-BY-4.0 conjunction data and MIT code.
