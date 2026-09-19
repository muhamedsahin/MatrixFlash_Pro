import { NextResponse } from 'next/server'
import course from '@/content/matrix-course.json'

/** GET /api/matrix-course — serves the university-level matrix course JSON. */
export async function GET() {
  return NextResponse.json(course)
}
