import tailwind from '@astrojs/tailwind';
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  integrations: [tailwind()],
  site: 'https://example.com',
  output: 'hybrid', // Enable SSR for API endpoints while keeping pages static
});
