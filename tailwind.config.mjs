/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'bg-light': '#D9CDC3',
        'bg-dark': '#15101A',
        accent: 'var(--color-accent)',
        'accent-soft': 'var(--color-accent-soft)',
        divider: 'var(--color-divider)',
      },
    },
  },
  plugins: [],
}
