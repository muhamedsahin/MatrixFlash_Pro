/**
 * Shared documentation content types.
 * These describe the block-based JSON schema served by /api/docs/* and
 * /api/matrix-course. Keep in sync with the files under content/.
 */

/** Bilingual text (Turkish / English). */
export type LText = { tr: string; en: string }

export type CodeLang = 'cpp' | 'bash' | 'text'

/** A single content block inside a documentation section. */
export type DocBlock =
  | { type: 'p'; text: LText }
  | { type: 'h3'; text: LText }
  | { type: 'code'; lang: CodeLang; filename?: string; code: string }
  | { type: 'callout'; variant: 'info' | 'tip' | 'warn'; title?: LText; text: LText }
  | { type: 'math'; tex: string; display?: boolean }
  | { type: 'list'; ordered?: boolean; items: LText[] }
  | { type: 'table'; headers: LText[]; rows: LText[][] }
  | {
      type: 'figure'
      data: (string | number)[][]
      caption?: LText
      highlight?: [number, number][]
      accent?: 'primary' | 'accent' | 'chart-3'
    }
  | { type: 'steps'; items: { title: LText; text: LText }[] }
  | { type: 'simulator' }

/** One documented API surface (function / class / kernel). */
export type ApiParam = { name: string; type: string; desc: LText }

export type ApiEntryData = {
  name: string
  signature: string
  badge?: string
  returns?: string
  params?: ApiParam[]
  text: LText
  example?: string
}

/** A titled section of a documentation page. */
export type DocSectionData = {
  id: string
  title: LText
  intro?: LText
  blocks?: DocBlock[]
  entries?: ApiEntryData[]
}

export type DocPageMeta = {
  eyebrow: LText
  title: LText
  description: LText
}

/** Full payload served for /api/docs/[slug]. */
export type DocPageData = {
  slug: string
  meta: DocPageMeta
  sections: DocSectionData[]
}

/** Full payload served for /api/matrix-course. */
export type CourseData = {
  slug: 'matrix-course'
  meta: DocPageMeta
  sections: DocSectionData[]
}
