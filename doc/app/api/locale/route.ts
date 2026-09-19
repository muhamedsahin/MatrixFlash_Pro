import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

function countryFromRequest(request: NextRequest): string | null {
  const headers = request.headers
  const raw =
    headers.get('x-vercel-ip-country') ||
    headers.get('cf-ipcountry') ||
    headers.get('x-country-code') ||
    (request as NextRequest & { geo?: { country?: string } }).geo?.country ||
    null

  if (!raw || raw === 'XX' || raw === 'T1') return null
  return raw.toUpperCase()
}

/**
 * Returns suggested UI language from request geo.
 * TR → tr, any other known country → en, unknown → null.
 */
export function GET(request: NextRequest) {
  const country = countryFromRequest(request)
  const lang = country === 'TR' ? 'tr' : country ? 'en' : null

  return NextResponse.json(
    { country, lang },
    {
      headers: {
        'Cache-Control': 'private, max-age=3600',
      },
    },
  )
}
