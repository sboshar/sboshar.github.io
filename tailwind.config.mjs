/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'bg-light': '#ffffff',
        'bg-dark': '#1C1C1D',
      },
    },
  },
  plugins: [],
}
