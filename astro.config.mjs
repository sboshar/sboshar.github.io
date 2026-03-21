import tailwind from '@astrojs/tailwind';
import { defineConfig } from 'astro/config';

// Static site — free on GitHub Pages (see README).
// Project URL: https://<user>.github.io/<repo>/  →  set base to '/<repo>' (no trailing slash).
// Custom domain or user site at root: base: '/'
export default defineConfig({
  site: 'https://sboshar.github.io',
  base: '/portfolio',
  integrations: [tailwind()],
  output: 'static',
});
