# sboshar.github.io

Personal website for **Sam Boshar** — ML research engineer at [InstaDeep](https://www.instadeep.com/) / [BioNTech](https://www.biontech.com/).

Built with [Astro](https://astro.build/) and [Tailwind CSS](https://tailwindcss.com/), deployed via GitHub Pages.

**Live:** [sboshar.github.io](https://sboshar.github.io/)

## Development

```bash
npm install
npm run dev       # local dev server
npm run build     # production build → dist/
npm run preview   # preview the build locally
```

## Deployment

Pushes to `main` trigger the GitHub Actions workflow (`.github/workflows/deploy.yml`), which builds the site and publishes `dist/` to GitHub Pages.
