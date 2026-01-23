import type { APIRoute } from 'astro';

// Ensure this endpoint is not prerendered
export const prerender = false;

export const GET: APIRoute = async ({ url, cookies, redirect }) => {
  const code = url.searchParams.get('code');
  const error = url.searchParams.get('error');
  const scope = url.searchParams.get('scope');
  const state = url.searchParams.get('state');

  // Log callback for debugging if needed
  if (!code && !error) {
    console.log('Callback received without code or error:', {
      origin: url.origin,
      search: url.search,
    });
  }

  if (error) {
    console.error('Strava returned error:', error);
    return redirect('/activities?error=access_denied', 302);
  }

  if (!code) {
    console.error('No code parameter in callback URL');
    return redirect('/activities?error=no_code', 302);
  }

  const clientId = import.meta.env.STRAVA_CLIENT_ID;
  const clientSecret = import.meta.env.STRAVA_CLIENT_SECRET;
  const redirectUri = `${url.origin}/api/strava/callback`;

  if (!clientId || !clientSecret) {
    return redirect('/activities?error=config_error', 302);
  }

  try {
    // Exchange code for access token
    // Strava expects form-encoded data, not JSON
    // Include redirect_uri for security validation
    const params = new URLSearchParams({
      client_id: clientId,
      client_secret: clientSecret,
      code,
      grant_type: 'authorization_code',
      redirect_uri: redirectUri,
    });

    const tokenResponse = await fetch('https://www.strava.com/api/v3/oauth/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: params.toString(),
    });

    if (!tokenResponse.ok) {
      const errorData = await tokenResponse.json().catch(() => ({}));
      console.error('Token exchange failed:', {
        status: tokenResponse.status,
        statusText: tokenResponse.statusText,
        error: errorData,
        redirectUri,
        code: code.substring(0, 10) + '...',
      });
      return redirect(`/activities?error=token_exchange_failed&details=${encodeURIComponent(errorData.message || tokenResponse.statusText)}`, 302);
    }

    const tokenData = await tokenResponse.json();
    
    // Store tokens securely in httpOnly cookies
    cookies.set('strava_access_token', tokenData.access_token, {
      httpOnly: true,
      secure: import.meta.env.PROD,
      sameSite: 'lax',
      maxAge: tokenData.expires_in || 21600, // Default 6 hours
      path: '/',
    });

    cookies.set('strava_refresh_token', tokenData.refresh_token, {
      httpOnly: true,
      secure: import.meta.env.PROD,
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 365, // 1 year
      path: '/',
    });

    cookies.set('strava_expires_at', String(Date.now() + (tokenData.expires_in * 1000)), {
      httpOnly: true,
      secure: import.meta.env.PROD,
      sameSite: 'lax',
      maxAge: tokenData.expires_in || 21600,
      path: '/',
    });

    return redirect('/activities?success=true', 302);
  } catch (error: any) {
    console.error('Strava callback error:', error);
    return redirect(`/activities?error=server_error&details=${encodeURIComponent(error.message || 'Unknown error')}`, 302);
  }
};
