import tailwind from '@astrojs/tailwind';
import { defineConfig } from 'astro/config';

// User site: https://sboshar.github.io/  — GitHub repo MUST be named `sboshar.github.io` (not `portfolio`).
export default defineConfig({
  site: 'https://sboshar.github.io',
  base: '/',
  integrations: [tailwind()],
  output: 'static',
});
