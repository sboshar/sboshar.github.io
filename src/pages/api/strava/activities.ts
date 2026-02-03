import type { APIRoute } from 'astro';
import { fetchStravaActivities } from '../../../lib/strava';

export const GET: APIRoute = async ({ url, cookies }) => {
  try {
    const forceRefresh = url.searchParams.get('refresh') === '1';
    const data = await fetchStravaActivities({
      cookies,
      origin: url.origin,
      forceRefresh,
    });

    if (data.authUrl) {
      return new Response(
        JSON.stringify({ authUrl: data.authUrl }),
        { status: 401, headers: { 'Content-Type': 'application/json' } }
      );
    }

    if (data.error) {
      return new Response(
        JSON.stringify({ error: data.error }),
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      );
    }

    return new Response(
      JSON.stringify({ activities: data.activities, cached: data.cached, lastSync: data.lastSync }),
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
