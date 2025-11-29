import markdown
from xhtml2pdf import pisa
import os

def convert_md_to_pdf(source_md, output_pdf):
    # 1. Read Markdown
    with open(source_md, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Convert to HTML
    # Add some basic CSS for better formatting
    css = """
    <style>
        body { font-family: Helvetica, sans-serif; font-size: 12px; }
        h1 { color: #2E3E50; border-bottom: 1px solid #ccc; }
        h2 { color: #2E3E50; margin-top: 20px; }
        h3 { color: #E67E22; }
        code { background-color: #f4f4f4; padding: 2px; font-family: monospace; }
        pre { background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd; white-space: pre-wrap; }
        img { max-width: 100%; height: auto; margin: 20px 0; }
    </style>
    """
    
    # Update image paths to be absolute or relative to execution
    # The report uses /c:/Users/... which might be an issue for xhtml2pdf if not handled right.
    # xhtml2pdf usually expects local paths.
    # Let's replace the absolute path prefix if necessary or ensure it works.
    # The current paths in MD are like /c:/Users/nefta/Develop/Paralela-Proyecto/analysis/time_vs_processes.png
    # We might need to strip the leading / for Windows paths if xhtml2pdf complains.
    
    html_content = markdown.markdown(text, extensions=['fenced_code', 'tables'])
    full_html = f"<html><head>{css}</head><body>{html_content}</body></html>"

    # Fix paths: xhtml2pdf on Windows might not like /c:/...
    # We will replace "/c:/" with "c:/"
    full_html = full_html.replace('src="/c:/', 'src="c:/')

    # 3. Write PDF
    with open(output_pdf, "wb") as result_file:
        pisa_status = pisa.CreatePDF(
            full_html,                # the HTML to convert
            dest=result_file          # file handle to recieve result
        )

    if pisa_status.err:
        print(f"Error converting to PDF: {pisa_status.err}")
    else:
        print(f"Successfully created {output_pdf}")

if __name__ == "__main__":
    # Path to the artifact
    md_path = "C:/Users/nefta/.gemini/antigravity/brain/ca040b7e-3747-41c1-8dab-14a72db68523/reporte_proyecto.md"
    pdf_path = "c:/Users/nefta/Develop/Paralela-Proyecto/docs/Reporte_Proyecto_Paralela.pdf"
    
    # Ensure docs dir exists
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    convert_md_to_pdf(md_path, pdf_path)
