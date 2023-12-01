import markdown
import sys
from weasyprint import HTML

def markdown_to_pdf(md_file_path, pdf_file_path):
    # Read the Markdown file
    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

    # Convert Markdown to HTML
    html_content = markdown.markdown(md_content)

    # Convert HTML to PDF and save
    HTML(string=html_content).write_pdf(pdf_file_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python markdown_to_pdf.py [input.md] [output.pdf]")
        sys.exit(1)

    markdown_to_pdf(sys.argv[1], sys.argv[2])
