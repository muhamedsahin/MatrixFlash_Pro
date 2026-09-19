import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

const GEO_COOKIE = 'mfp_geo_lang'
const PREF_COOKIE = 'mfp_lang'

function countryFromRequest(request: NextRequest): string | null {
  const headers = request.headers
  const raw =
    headers.get('x-vercel-ip-country') ||
    headers.get('cf-ipcountry') ||
    headers.get('x-country-code') ||
    // Next.js / Vercel geo helper when available
    (request as NextRequest & { geo?: { country?: string } }).geo?.country ||
    null

  if (!raw || raw === 'XX' || raw === 'T1') return null
  return raw.toUpperCase()
}

export function middleware(request: NextRequest) {
  const response = NextResponse.next()

  // Manual language preference always wins — do not overwrite.
  if (request.cookies.get(PREF_COOKIE)?.value) {
    return response
  }

  // Already have a geo hint for this session.
  if (request.cookies.get(GEO_COOKIE)?.value) {
    return response
  }

  const country = countryFromRequest(request)
  if (!country) return response

  const lang = country === 'TR' ? 'tr' : 'en'
  response.cookies.set(GEO_COOKIE, lang, {
    path: '/',
    maxAge: 60 * 60 * 24 * 30,
    sameSite: 'lax',
  })

  return response
}

export const config = {
  matcher: [
    /*
     * Skip static assets and Next internals.
     */
    '/((?!_next/static|_next/image|favicon.ico|data/|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico)$).*)',
  ],
}
