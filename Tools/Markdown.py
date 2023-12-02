import markdown2
import pdfkit
import sys

def markdown_to_pdf(input_file, output_file):
    # Read Markdown content from a file
    with open(input_file, 'r', encoding='utf-8') as file:
        markdown_text = file.read()

    # Convert Markdown to HTML
    html_text = markdown2.markdown(markdown_text)

    # Convert HTML to PDF and save the PDF file
    pdfkit.from_string(html_text, output_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python markdown_to_pdf.py [input_file.md] [output_file.pdf]")
        sys.exit(1)

    input_md = sys.argv[1]
    output_pdf = sys.argv[2]
    markdown_to_pdf(input_md, output_pdf)
    print(f"Converted {input_md} to {output_pdf}")
