// tools/manual/styles.js
// Professional publication-grade print CSS for MatrixFlash-Pro Technical Manual

module.exports = `
  @page {
    size: A4;
    margin: 16mm 14mm 18mm 14mm;
    @bottom-center {
      content: "MatrixFlash-Pro Technical Manual | Page " counter(page);
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      font-size: 8pt;
      color: #718096;
      border-top: 1px solid #E2E8F0;
      padding-top: 4mm;
      width: 100%;
    }
  }

  * {
    box-sizing: border-box;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 9.8pt;
    line-height: 1.55;
    color: #1A202C;
    background-color: #FFFFFF;
    margin: 0;
    padding: 0;
  }

  .cover-page {
    height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 30mm 15mm 20mm 15mm;
    background: linear-gradient(145deg, #0F172A 0%, #020617 60%, #064E3B 100%);
    color: #F8FAFC;
    page-break-after: always;
    break-after: page;
  }

  .cover-header {
    border-bottom: 2px solid #10B981;
    padding-bottom: 20px;
  }

  .cover-badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10B981;
    color: #34D399;
    font-weight: 700;
    font-size: 9pt;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 4px;
    margin-bottom: 16px;
  }

  .cover-title {
    font-size: 34pt;
    font-weight: 900;
    line-height: 1.15;
    letter-spacing: -1px;
    margin: 0 0 12px 0;
    color: #FFFFFF;
  }

  .cover-title span {
    color: #10B981;
  }

  .cover-subtitle {
    font-size: 14pt;
    font-weight: 400;
    color: #94A3B8;
    margin: 0 0 20px 0;
    line-height: 1.4;
  }

  .cover-spec-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 40px;
  }

  .cover-spec-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 14px;
  }

  .cover-spec-card .label {
    font-size: 7.5pt;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94A3B8;
    margin-bottom: 4px;
  }

  .cover-spec-card .val {
    font-size: 13pt;
    font-weight: 700;
    color: #38BDF8;
  }

  .cover-footer {
    border-top: 1px solid rgba(255, 255, 255, 0.15);
    padding-top: 16px;
    display: flex;
    justify-content: space-between;
    font-size: 8.5pt;
    color: #64748B;
  }

  .chapter {
    page-break-before: always;
    break-before: page;
    padding-top: 6mm;
  }

  .chapter-header {
    border-bottom: 2.5px solid #0F172A;
    padding-bottom: 8px;
    margin-bottom: 18px;
  }

  .chapter-num {
    font-size: 10pt;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #059669;
    margin-bottom: 4px;
  }

  .chapter-title {
    font-size: 20pt;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #0F172A;
    margin: 0;
    line-height: 1.25;
  }

  h2 {
    font-size: 13pt;
    font-weight: 700;
    color: #1E293B;
    margin: 20px 0 8px 0;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 4px;
    page-break-after: avoid;
    break-after: avoid;
  }

  h3 {
    font-size: 10.5pt;
    font-weight: 700;
    color: #334155;
    margin: 14px 0 6px 0;
    page-break-after: avoid;
    break-after: avoid;
  }

  p {
    margin: 0 0 9px 0;
    text-align: justify;
  }

  ul, ol {
    margin: 0 0 10px 0;
    padding-left: 20px;
  }

  li {
    margin-bottom: 4px;
  }

  .callout {
    background-color: #F8FAFC;
    border-left: 4px solid #3B82F6;
    padding: 10px 14px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
    font-size: 9pt;
    page-break-inside: avoid;
    break-inside: avoid;
  }

  .callout.tip {
    border-left-color: #10B981;
    background-color: #F0FDF4;
  }

  .callout.warning {
    border-left-color: #F59E0B;
    background-color: #FFFBEB;
  }

  .callout.math {
    border-left-color: #8B5CF6;
    background-color: #F5F3FF;
  }

  .callout-title {
    font-weight: 700;
    margin-bottom: 4px;
    color: #0F172A;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .formula-box {
    background: #F1F5F9;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 10px 0;
    text-align: center;
    font-family: 'Cambria Math', 'Times New Roman', serif;
    font-size: 10.5pt;
    page-break-inside: avoid;
    break-inside: avoid;
  }

  .formula-box .eq-desc {
    display: block;
    font-family: -apple-system, sans-serif;
    font-size: 8pt;
    color: #64748B;
    margin-top: 4px;
  }

  pre {
    background-color: #0F172A;
    color: #E2E8F0;
    padding: 10px 12px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 8pt;
    line-height: 1.45;
    overflow-x: hidden;
    margin: 10px 0;
    border: 1px solid #1E293B;
    page-break-inside: avoid;
    break-inside: avoid;
    white-space: pre-wrap;
    word-break: break-all;
  }

  code {
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 8.5pt;
    background: #F1F5F9;
    color: #0F172A;
    padding: 1px 4px;
    border-radius: 3px;
    border: 1px solid #E2E8F0;
  }

  pre code {
    background: transparent;
    color: inherit;
    padding: 0;
    border: none;
    font-size: 8pt;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
    break-inside: avoid;
  }

  th, td {
    border: 1px solid #CBD5E1;
    padding: 6px 9px;
    text-align: left;
    vertical-align: top;
  }

  th {
    background-color: #F1F5F9;
    font-weight: 700;
    color: #1E293B;
  }

  tr:nth-child(even) {
    background-color: #F8FAFC;
  }

  .arch-diagram {
    background: #0B132B;
    color: #A3CEF1;
    padding: 12px;
    border-radius: 6px;
    font-family: 'Consolas', monospace;
    font-size: 7.5pt;
    line-height: 1.25;
    margin: 12px 0;
    border: 1px solid #1C2541;
    page-break-inside: avoid;
    break-inside: avoid;
  }

  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 10px 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }

  .toc-entry {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px dotted #CBD5E1;
    padding: 4px 0;
    margin-bottom: 4px;
    font-size: 9.5pt;
  }

  .toc-entry .title {
    font-weight: 600;
    color: #1E293B;
  }

  .toc-entry .dots {
    flex-grow: 1;
    border-bottom: 1px dotted #94A3B8;
    margin: 0 8px;
  }

  .toc-entry .num {
    font-weight: 700;
    color: #059669;
  }

  .page-subbreak {
    page-break-before: always;
    break-before: page;
    padding-top: 4mm;
  }
`;

