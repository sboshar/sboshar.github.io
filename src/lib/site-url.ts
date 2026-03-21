/** Astro / Vite base path; normalize so joins never drop a slash. */
function baseWithSlash(): string {
  let b = import.meta.env.BASE_URL;
  if (b !== '/' && !b.endsWith('/')) b = `${b}/`;
  return b;
}

/** Internal URL under site base (GitHub project page or `/`). */
export function siteHref(path = ''): string {
  const base = baseWithSlash();
  if (!path || path === '/') return base;
  const clean = path.replace(/^\//, '');
  return `${base}${clean}`;
}

export function normalizePathname(pathname: string): string {
  return pathname.replace(/\/$/, '') || '/';
}
