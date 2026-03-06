import gradio as gr
from ocr_engine import recognize_text

with gr.Blocks(title="Noise2Text") as iface:
    gr.Markdown(
        """
        # Noise2Text:- Robust Distorted Text Recognizer
        Upload a distorted or noisy image containing text.
        The system applies a multi-layer preprocessing pipeline and performs OCR using **EasyOCR** — entirely offline.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            run_btn = gr.Button("Recognize Text", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Processed Image")
            output_text = gr.Textbox(label="Recognized Text", lines=3)
            output_conf = gr.Textbox(label="Confidence Score")

    run_btn.click(
        fn=recognize_text,
        inputs=input_image,
        outputs=[output_image, output_text, output_conf],
    )

    input_image.change(
        fn=recognize_text,
        inputs=input_image,
        outputs=[output_image, output_text, output_conf],
    )

if __name__ == "__main__":
    iface.launch()
