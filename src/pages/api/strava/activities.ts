import type { APIRoute } from 'astro';

async function refreshAccessToken(refreshToken: string): Promise<string | null> {
  const clientId = import.meta.env.STRAVA_CLIENT_ID;
  const clientSecret = import.meta.env.STRAVA_CLIENT_SECRET;

  if (!clientId || !clientSecret) {
    return null;
  }

  try {
    // Strava expects form-encoded data, not JSON
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

  // Check if token is expired (with 5 minute buffer)
  if (expiresAt && parseInt(expiresAt) < Date.now() + 5 * 60 * 1000) {
    if (refreshToken) {
      const newToken = await refreshAccessToken(refreshToken);
      if (newToken) {
        // Update cookies with new token
        cookies.set('strava_access_token', newToken, {
          httpOnly: true,
          secure: import.meta.env.PROD,
          sameSite: 'lax',
          maxAge: 21600, // 6 hours
          path: '/',
        });
        return newToken;
      }
    }
    return null;
  }

  return accessToken;
}

export const GET: APIRoute = async ({ url, cookies }) => {
  const accessToken = await getAccessToken(cookies);

  if (!accessToken) {
    // Return auth URL for user to connect
    const clientId = import.meta.env.STRAVA_CLIENT_ID;
    if (!clientId) {
      return new Response(
        JSON.stringify({ error: 'Strava not configured' }),
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const redirectUri = `${url.origin}/api/strava/callback`;
    const scope = 'activity:read';
    const authUrl = `https://www.strava.com/oauth/authorize?client_id=${clientId}&redirect_uri=${encodeURIComponent(redirectUri)}&response_type=code&scope=${scope}`;

    return new Response(
      JSON.stringify({ authUrl }),
      { status: 401, headers: { 'Content-Type': 'application/json' } }
    );
  }

  try {
    // Fetch athlete activities
    const perPage = 30;
    const response = await fetch(
      `https://www.strava.com/api/v3/athlete/activities?per_page=${perPage}`,
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );

    if (!response.ok) {
      if (response.status === 401) {
        // Token expired, clear cookies and return auth URL
        cookies.delete('strava_access_token', { path: '/' });
        cookies.delete('strava_refresh_token', { path: '/' });
        cookies.delete('strava_expires_at', { path: '/' });

        const clientId = import.meta.env.STRAVA_CLIENT_ID;
        const redirectUri = `${url.origin}/api/strava/callback`;
        const scope = 'activity:read';
        const authUrl = `https://www.strava.com/oauth/authorize?client_id=${clientId}&redirect_uri=${encodeURIComponent(redirectUri)}&response_type=code&scope=${scope}`;

        return new Response(
          JSON.stringify({ authUrl }),
          { status: 401, headers: { 'Content-Type': 'application/json' } }
        );
      }

      return new Response(
        JSON.stringify({ error: 'Failed to fetch activities' }),
        { status: response.status, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const activities = await response.json();

    return new Response(
      JSON.stringify({ activities }),
      { headers: { 'Content-Type': 'application/json' } }
    );
  } catch (error) {
    console.error('Error fetching activities:', error);
    return new Response(
      JSON.stringify({ error: 'Server error' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
};
