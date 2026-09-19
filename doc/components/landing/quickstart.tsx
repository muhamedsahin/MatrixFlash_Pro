'use client'

import { useState } from 'react'
import { useLanguage } from '@/lib/hooks/use-language'
import { CodeBlock } from '@/components/code-block'
import { cn } from '@/lib/utils'

const QUICKSTART_SNIPPETS = {
  basic: {
    label: 'Quickstart GEMM',
    filename: 'quickstart.cpp',
    code: `#include "matrix_pro/matrix_pro.hpp"
#include <iostream>

int main() {
    using matrix_pro::Matrix;
    using matrix_pro::MemoryMode;

    // Device-resident matris (PCIe yükü yok)
    Matrix a{{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix b = Matrix::identity(2);

    // cuBLAS GEMM + ReLU aktivasyonu
    Matrix c = (a * b).relu();

    // Fail-fast kuralı: download() çağrılmadan host okuması yapılamaz
    c.download();
    std::cout << "c(0,0): " << c.at(0, 0) << "\\n";
    return 0;
}`,
  },
  autograd: {
    label: 'Autograd Tape',
    filename: 'autograd_demo.cpp',
    code: `#include "matrix_pro/matrix_pro.hpp"

using matrix_pro::Variable;
using matrix_pro::Matrix;

int main() {
    // Otomatik türev değişkenleri
    Variable x(Matrix::randn(64, 128), /*requires_grad=*/true);
    Variable w(Matrix::randn(128, 10), /*requires_grad=*/true);

    // İleri yayılım: Y = x * W
    Variable y = x.matmul(w).relu();

    // Geri yayılım (Backpropagation)
    y.backward();

    // x ve w için gradyanlar GPU üzerinde hazır!
    const Matrix& grad_w = w.grad();
    return 0;
}`,
  },
  fused: {
    label: 'Fused & In-Place',
    filename: 'fused_perf.cpp',
    code: `#include "matrix_pro/matrix_pro.hpp"

int main() {
    using namespace matrix_pro;

    Matrix x = Matrix::uniform(512, 512, -1.0f, 1.0f);
    Matrix bias = Matrix::zeros(512, 512);

    // Tek GPU kernel'ında bias + GeLU birleşimi
    Matrix activated = fused_bias_gelu(x, bias);

    // Bellek tasarruflu yerinde (in-place) işlem
    relu_(activated);
    add_(activated, 0.5f);
    return 0;
}`,
  },
}

/** Tabbed terminal showcase with animated active state. */
export function Quickstart() {
  const { t } = useLanguage()
  const [activeTab, setActiveTab] = useState<keyof typeof QUICKSTART_SNIPPETS>('basic')

  return (
    <section className="relative mx-auto max-w-7xl px-4 pb-16 sm:px-6 lg:px-8">
      <div className="mb-6 text-center">
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-primary">
          {t('30 SANİYEDE İLK ÇALIŞTIRMA', 'YOUR FIRST RUN IN 30 SECONDS')}
        </p>
        <h2 className="fluid-section-title mt-2 text-foreground">
          {t('Terminalde Gör, GPU’da Hesapla', 'See It in the Terminal, Compute on the GPU')}
        </h2>
      </div>

      <div className="relative mx-auto max-w-3xl">
        <div className="absolute -inset-1 -z-10 rounded-2xl bg-gradient-to-r from-primary/20 via-transparent to-accent/20 blur-lg" aria-hidden />
        <div className="relative rounded-2xl border border-border/80 bg-gradient-to-b from-card to-card/60 p-2 shadow-2xl backdrop-blur-xl">
          <div className="flex items-center justify-between border-b border-border/60 px-3 py-2">
            <div className="flex items-center gap-1.5">
              <span className="size-3 rounded-full bg-destructive/60" />
              <span className="size-3 rounded-full bg-amber-500/60" />
              <span className="size-3 rounded-full bg-emerald-500/60" />
            </div>
            <div className="flex items-center gap-1">
              {(Object.keys(QUICKSTART_SNIPPETS) as Array<keyof typeof QUICKSTART_SNIPPETS>).map(
                (key) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setActiveTab(key)}
                    className={cn(
                      'rounded-md px-2.5 py-1 font-mono text-[11px] font-semibold transition-all',
                      activeTab === key
                        ? 'border border-primary/40 bg-primary/20 text-primary'
                        : 'text-muted-foreground hover:bg-white/[0.04] hover:text-foreground',
                    )}
                  >
                    {QUICKSTART_SNIPPETS[key].label}
                  </button>
                ),
              )}
            </div>
          </div>

          <div className="p-2">
            <CodeBlock
              code={QUICKSTART_SNIPPETS[activeTab].code}
              filename={QUICKSTART_SNIPPETS[activeTab].filename}
            />
          </div>

          <div className="flex items-center justify-between border-t border-border/50 px-3 py-2 font-mono text-[10px] text-muted-foreground">
            <span className="flex items-center gap-1.5 text-primary">
              <span className="size-1.5 animate-ping rounded-full bg-primary" />
              GPU Memory: Resident
            </span>
            <span>nvcc -O3 -std=c++17</span>
          </div>
        </div>
      </div>
    </section>
  )
}
