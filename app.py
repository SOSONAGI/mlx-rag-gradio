import gradio as gr
from vdb import VectorDB
from mlx_lm import load, generate
import PyPDF2
from io import BytesIO

def vdb_from_pdf(pdf_file):
    model = VectorDB()
    reader = PyPDF2.PdfReader(BytesIO(pdf_file))
    content = "\n\n".join([reader.pages[i].extract_text() for i in range(len(reader.pages))])
    model.ingest(content=content)
    return model

def answer_question(model_repo, pdf_file, question):
    
    vdb = vdb_from_pdf(pdf_file)

    
    model, tokenizer = load(model_repo)
    context = vdb.query(question)
    prompt = f"CONTEXT:\n\n{context}\n\nQuestion: {question}\nAnswer:\n"
    answer = generate(model, tokenizer, prompt=prompt, max_tokens=512)
    return answer


iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(label="Enter Hugging Face Repo"),  
        gr.File(type="binary", label="Upload the PDF file"),
        gr.Textbox(label="Ask your question about pdf here")
    ],
    outputs=gr.Textbox(label="Answer")
)


iface.launch(share=True)
