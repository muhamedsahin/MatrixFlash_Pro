import { NextResponse } from 'next/server'
import gettingStarted from '@/content/docs/getting-started.json'
import apiReference from '@/content/docs/api-reference.json'
import examples from '@/content/docs/examples.json'
import benchmarksComparison from '@/content/docs/benchmarks-comparison.json'

const DOCS: Record<string, unknown> = {
  'getting-started': gettingStarted,
  'api-reference': apiReference,
  examples,
  'benchmarks-comparison': benchmarksComparison,
}

/** GET /api/docs/[slug] — serves the block-based documentation JSON. */
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ slug: string }> },
) {
  const { slug } = await params
  const data = DOCS[slug]

  if (!data) {
    return NextResponse.json({ error: `Unknown doc: ${slug}` }, { status: 404 })
  }

  return NextResponse.json(data)
}
