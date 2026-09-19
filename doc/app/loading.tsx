import { Logo } from '@/components/logo'

/** Route-level loading state with a boot-style pulse. */
export default function DocsLoading() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-6">
      <div className="relative animate-pulse-glow">
        <Logo className="size-14" />
      </div>
      <div className="flex items-center gap-3 font-mono text-xs text-muted-foreground">
        <span className="size-1.5 animate-ping rounded-full bg-primary" />
        <span>kernel&apos;ler yükleniyor / loading kernels ...</span>
      </div>
      <div className="h-1 w-48 overflow-hidden rounded-full bg-secondary/60">
        <div className="h-full w-1/2 animate-shimmer rounded-full bg-gradient-to-r from-primary to-accent" />
      </div>
    </div>
  )
}
