import { promises as fs } from 'fs';
import path from 'path';

const CACHE_DIR = path.resolve(process.cwd(), '.cache');
const CACHE_FILE = path.join(CACHE_DIR, 'strava-activities.json');
const CACHE_TTL_MS = 6 * 60 * 60 * 1000; // 6 hours

type CachedActivities = {
  lastSync: number;
  activities: any[];
};

async function refreshAccessToken(refreshToken: string): Promise<string | null> {
  const clientId = import.meta.env.STRAVA_CLIENT_ID;
  const clientSecret = import.meta.env.STRAVA_CLIENT_SECRET;

  if (!clientId || !clientSecret) {
    return null;
  }

  try {
    const params = new URLSearchParams({
      client_id: clientId,
      client_secret: clientSecret,
      refresh_token: refreshToken,
      grant_type: 'refresh_token',
    });

    const response = await fetch('https://www.strava.com/api/v3/oauth/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: params.toString(),
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json();
    return data.access_token;
  } catch (error) {
    console.error('Token refresh error:', error);
    return null;
  }
}

async function getAccessToken(cookies: any): Promise<string | null> {
  const accessToken = cookies.get('strava_access_token')?.value;
  const refreshToken = cookies.get('strava_refresh_token')?.value;
  const expiresAt = cookies.get('strava_expires_at')?.value;

  if (!accessToken) {
    return null;
  }

  if (expiresAt && parseInt(expiresAt) < Date.now() + 5 * 60 * 1000) {
    if (refreshToken) {
      const newToken = await refreshAccessToken(refreshToken);
      if (newToken) {
        cookies.set('strava_access_token', newToken, {
          httpOnly: true,
          secure: import.meta.env.PROD,
          sameSite: 'lax',
          maxAge: 21600,
          path: '/',
        });
        return newToken;
      }
    }
    return null;
  }

  return accessToken;
}

function buildAuthUrl(origin: string): string | null {
  const clientId = import.meta.env.STRAVA_CLIENT_ID;
  if (!clientId) return null;
  const redirectUri = `${origin}/api/strava/callback`;
  const scope = 'activity:read';
  return `https://www.strava.com/oauth/authorize?client_id=${clientId}&redirect_uri=${encodeURIComponent(redirectUri)}&response_type=code&scope=${scope}`;
}

export async function fetchStravaActivities({
  cookies,
  origin,
  forceRefresh = false,
}: {
  cookies: any;
  origin: string;
  forceRefresh?: boolean;
}): Promise<{ activities: any[]; authUrl?: string; error?: string; cached?: boolean; lastSync?: number }> {
  const accessToken = await getAccessToken(cookies);

  if (!accessToken) {
    const authUrl = buildAuthUrl(origin);
    return authUrl
      ? { activities: [], authUrl }
      : { activities: [], error: 'Strava not configured' };
  }

  let cached: CachedActivities | null = null;
  try {
    const cacheRaw = await fs.readFile(CACHE_FILE, 'utf-8');
    const parsed = JSON.parse(cacheRaw) as CachedActivities;
    if (parsed && Array.isArray(parsed.activities) && typeof parsed.lastSync === 'number') {
      cached = parsed;
    }
  } catch {
    cached = null;
  }

  const cacheFresh = cached && Date.now() - cached.lastSync < CACHE_TTL_MS;
  if (cached && cacheFresh && !forceRefresh) {
    return { activities: cached.activities, cached: true, lastSync: cached.lastSync };
  }

  const cachedActivities = cached?.activities || [];
  const cachedIds = new Set(cachedActivities.map((activity) => activity.id));
  const newActivities: any[] = [];

  const perPage = 200;
  let page = 1;
  let reachedKnown = false;

  while (true) {
    const response = await fetch(
      `https://www.strava.com/api/v3/athlete/activities?per_page=${perPage}&page=${page}`,
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );

    if (!response.ok) {
      if (response.status === 401) {
        cookies.delete('strava_access_token', { path: '/' });
        cookies.delete('strava_refresh_token', { path: '/' });
        cookies.delete('strava_expires_at', { path: '/' });
        const authUrl = buildAuthUrl(origin);
        return authUrl
          ? { activities: [], authUrl }
          : { activities: [], error: 'Strava authorization required' };
      }

      return { activities: [], error: 'Failed to fetch activities' };
    }

    const pageActivities = await response.json();
    if (!Array.isArray(pageActivities) || pageActivities.length === 0) {
      break;
    }

    for (const activity of pageActivities) {
      if (cachedIds.has(activity.id)) {
        reachedKnown = true;
        continue;
      }
      newActivities.push(activity);
    }

    if (reachedKnown) {
      break;
    }

    page += 1;
  }

  const activities = [...newActivities, ...cachedActivities];
  activities.sort((a, b) => {
    const aDate = new Date(a.start_date || a.start_date_local || 0).getTime();
    const bDate = new Date(b.start_date || b.start_date_local || 0).getTime();
    return bDate - aDate;
  });

  try {
    await fs.mkdir(CACHE_DIR, { recursive: true });
    const payload: CachedActivities = {
      lastSync: Date.now(),
      activities,
    };
    await fs.writeFile(CACHE_FILE, JSON.stringify(payload, null, 2), 'utf-8');
  } catch (error) {
    console.warn('Failed to write Strava cache:', error);
  }

  return { activities, cached: false, lastSync: Date.now() };
}
