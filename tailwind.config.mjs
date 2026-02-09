/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'bg-light': '#FAF8F3',
        'bg-dark': '#1C1C1D',
      },
    },
  },
  plugins: [],
}
