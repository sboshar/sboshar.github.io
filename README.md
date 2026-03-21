# Personal site (Astro, static)

Live at **`https://sboshar.github.io/`** (user site — see below).

## Host for free (GitHub Pages)

1. Repo on GitHub must be named **`sboshar.github.io`** (exactly) under account **`sboshar`**.  
   - That is the only way to get **`https://sboshar.github.io/`** without a custom domain.  
   - A repo named `portfolio` always publishes to **`https://sboshar.github.io/portfolio/`** — you cannot remove `/portfolio` while deploying from that repo name.

2. **Settings → Pages → Source: GitHub Actions**; push **`main`** so the workflow deploys `dist/`.

3. **`astro.config.mjs`** uses `site: 'https://sboshar.github.io'` and **`base: '/'`**.  
   If you ever deploy from a **project** repo again, set `base: '/<repo-name>'` and update `public/sitemap.xml` + `public/robots.txt`.

### Moving from `portfolio` to root URL

1. On GitHub: **New repository** → name **`sboshar.github.io`** → public.  
2. Copy this project in (push the same code), including **`.github/workflows/deploy.yml`**.  
3. Turn on **Pages → GitHub Actions** on the new repo.  
4. Optional: archive or delete the old **`portfolio`** repo, or keep it and add a note that the site moved.

5. Locally, point git at the new remote, e.g.  
   `git remote set-url origin https://github.com/sboshar/sboshar.github.io.git`

## Local

```bash
npm install
npm run dev
npm run build   # output in dist/
```
