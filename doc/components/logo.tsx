import { cn } from '@/lib/utils'

export function Logo({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        'grid place-items-center rounded-md bg-primary/10 ring-1 ring-primary/30',
        className,
      )}
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        className="size-[62%]"
        aria-hidden="true"
      >
        {/* matrix bracket + flash bolt */}
        <path
          d="M8 4H5v16h3M16 4h3v16h-3"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="text-primary"
        />
        <path
          d="M13 6l-4 6h3l-1 6 4-6h-3l1-6z"
          fill="currentColor"
          className="text-primary"
        />
      </svg>
    </span>
  )
}
