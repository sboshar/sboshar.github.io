# Personal site (Astro, static)

## Host for free (GitHub Pages)

1. In the repo on GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Push to `main`; the **Deploy static site to GitHub Pages** workflow builds and publishes `dist/`.
3. **`astro.config.mjs`** must match your URL:
   - **Project site** (default): `https://<user>.github.io/<repo>/`  
     Set `site` to `https://<user>.github.io` and `base` to `'/<repo>'` (e.g. `'/portfolio'`).
   - **User site** (`<user>.github.io` from a `username.github.io` repo) or **custom domain at root**: set `base: '/'` and set `site` to that full URL.

After changing `site` / `base`, commit and push so the next deploy picks it up.

## Local

```bash
npm install
npm run dev
npm run build   # output in dist/
```
