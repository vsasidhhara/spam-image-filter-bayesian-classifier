"""
Generates an HTML / printable PDF-ready version of the Project Report
DS307 Computer Vision - Project 12
"""

import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_MD = os.path.join(BASE_DIR, "PROJECT_REPORT.md")
REPORT_HTML = os.path.join(BASE_DIR, "PROJECT_REPORT.html")

def markdown_to_html(md_text):
    # Basic markdown parsing with styling suitable for academic report
    html = md_text

    # Escape HTML special characters except markdown tags
    # Convert code blocks
    def replace_code_block(match):
        code = match.group(2)
        return f'<pre class="code-block"><code>{code}</code></pre>'
    html = re.sub(r'```([a-zA-Z0-9_]*)\n(.*?)```', replace_code_block, html, flags=re.DOTALL)

    # Convert inline code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

    # Convert headers
    html = re.sub(r'^# (.*?)$', r'<h1 class="report-title">\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*?)$', r'<h2 class="section-title">\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.*?)$', r'<h3 class="subsection-title">\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.*?)$', r'<h4 class="subsubsection-title">\1</h4>', html, flags=re.MULTILINE)

    # Convert bold and italics
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)

    # Convert horizontal rules
    html = re.sub(r'^---$', r'<hr class="page-break">', html, flags=re.MULTILINE)

    # Convert markdown tables
    lines = html.split("\n")
    in_table = False
    table_html = []
    processed_lines = []

    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            if not in_table:
                in_table = True
                table_html = ['<table class="academic-table">']
            
            # Check if separator line
            if re.match(r'^\|[\s\:\-\|]+\|$', line.strip()):
                continue

            cells = [c.strip() for c in line.strip()[1:-1].split("|")]
            tag = "th" if len(table_html) == 1 else "td"
            row_str = "<tr>" + "".join([f"<{tag}>{c}</{tag}>" for c in cells]) + "</tr>"
            table_html.append(row_str)
        else:
            if in_table:
                table_html.append('</table>')
                processed_lines.append("\n".join(table_html))
                in_table = False
            processed_lines.append(line)

    if in_table:
        table_html.append('</table>')
        processed_lines.append("\n".join(table_html))

    html = "\n".join(processed_lines)

    # Convert lists
    html = re.sub(r'^\s*-\s+(.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^\s*\d+\.\s+(.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)

    # Convert paragraphs
    paragraphs = html.split("\n\n")
    wrapped_paras = []
    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        if p_clean.startswith("<h") or p_clean.startswith("<table") or p_clean.startswith("<pre") or p_clean.startswith("<hr") or p_clean.startswith("<li>") or p_clean.startswith("<div"):
            wrapped_paras.append(p_clean)
        else:
            wrapped_paras.append(f"<p>{p_clean}</p>")

    return "\n\n".join(wrapped_paras)

def build_full_html():
    with open(REPORT_MD, "r", encoding="utf-8") as f:
        md_content = f.read()

    body_content = markdown_to_html(md_content)

    full_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DS307 Project Report - Bayesian Spam Image Filter</title>
    <style>
        @page {{
            size: A4;
            margin: 20mm 15mm 20mm 15mm;
        }}
        body {{
            font-family: 'Times New Roman', Times, serif, Cambria;
            font-size: 11.5pt;
            line-height: 1.5;
            color: #111;
            margin: 0 auto;
            max-width: 900px;
            padding: 40px;
            background: #fff;
        }}
        .report-title {{
            text-align: center;
            font-size: 20pt;
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 10px;
            color: #0d3b66;
            text-transform: uppercase;
        }}
        .section-title {{
            font-size: 15pt;
            font-weight: bold;
            color: #0d3b66;
            border-bottom: 2px solid #0d3b66;
            padding-bottom: 4px;
            margin-top: 30px;
            margin-bottom: 12px;
        }}
        .subsection-title {{
            font-size: 12.5pt;
            font-weight: bold;
            color: #1f4e78;
            margin-top: 20px;
            margin-bottom: 8px;
        }}
        p {{
            text-align: justify;
            margin-bottom: 10px;
        }}
        .code-block {{
            background: #f4f6f8;
            border-left: 4px solid #0d3b66;
            padding: 10px 14px;
            font-family: Consolas, 'Courier New', monospace;
            font-size: 9.5pt;
            overflow-x: auto;
            margin: 14px 0;
            border-radius: 4px;
        }}
        code {{
            font-family: Consolas, 'Courier New', monospace;
            background: #eef1f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 10pt;
        }}
        .academic-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 18px 0;
            font-size: 10pt;
        }}
        .academic-table th, .academic-table td {{
            border: 1px solid #aaa;
            padding: 8px 10px;
            text-align: left;
        }}
        .academic-table th {{
            background-color: #e9eff5;
            font-weight: bold;
            color: #0d3b66;
        }}
        .academic-table tr:nth-child(even) {{
            background-color: #fafbfc;
        }}
        hr.page-break {{
            border: 0;
            height: 1px;
            background: #ccc;
            margin: 30px 0;
            page-break-after: always;
        }}
        ul, ol {{
            margin-bottom: 12px;
            padding-left: 25px;
        }}
        li {{
            margin-bottom: 5px;
            text-align: justify;
        }}
        @media print {{
            body {{
                padding: 0;
                max-width: 100%;
            }}
            .page-break {{
                page-break-after: always;
            }}
        }}
    </style>
</head>
<body>
{body_content}
</body>
</html>"""

    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(full_template)

    print(f"HTML report successfully generated at: {REPORT_HTML}")

if __name__ == "__main__":
    build_full_html()
