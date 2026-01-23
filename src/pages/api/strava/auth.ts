import type { APIRoute } from 'astro';

export const GET: APIRoute = async ({ url, redirect }) => {
  const clientId = import.meta.env.STRAVA_CLIENT_ID;
  const redirectUri = `${url.origin}/api/strava/callback`;
  
  if (!clientId) {
    return new Response(
      JSON.stringify({ error: 'Strava client ID not configured' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }

  const scope = 'activity:read';
  const authUrl = `https://www.strava.com/oauth/authorize?client_id=${clientId}&redirect_uri=${encodeURIComponent(redirectUri)}&response_type=code&scope=${scope}`;
  
  return redirect(authUrl, 302);
};
