"""Local image-upload frontend for the supplied OmniRestore checkpoint."""
import os
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import argparse
from functools import lru_cache
import json
from pathlib import Path
import threading
import time
from uuid import uuid4
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import torch
from benchmark_comparison import load_comparison, comparison_table
from infer import DEFAULT_EMBEDDER, load_models, restore
from utils.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'results_uploads'
MODEL_LOCK = threading.Lock()
WEATHER = [('Clear', 'clear'), ('Haze / fog', 'haze'), ('Rain', 'rain'),
           ('Snow', 'snow'), ('Low light', 'low'), ('Haze + rain', 'haze_rain'),
           ('Haze + snow', 'haze_snow'), ('Low light + haze', 'low_haze'),
           ('Low light + rain', 'low_rain'), ('Low light + snow', 'low_snow'),
           ('Low light + haze + rain', 'low_haze_rain'),
           ('Low light + haze + snow', 'low_haze_snow')]

CSS = '''
.gradio-container {max-width: 1240px !important; margin: auto;}
#hero {padding: 28px 32px; border-radius: 20px; background: linear-gradient(115deg,#102d3d,#164e52); margin-bottom: 20px;}
#hero h1 {color: #f3faf9; font-size: 34px; margin: 4px 0 10px; letter-spacing: -1px;}
#hero p {color: #d2e8e5; font-size: 16px;}
#hero .eyebrow {font-size: 12px; letter-spacing: 2px; color: #8de2c3; font-weight: 700;}
#restore-button {min-height: 48px; border-radius: 12px;}
#caption {font-size: 13px; color: #5b6f78;}
.gradio-container {width:100% !important; min-width:0 !important; box-sizing:border-box;} .main,main,.wrap,.tabs,.tabitem,.row,.column,.form {min-width:0 !important; max-width:100%;} .prose table {display:block;max-width:100%;overflow-x:auto;} @media(max-width:600px){.main{padding:12px !important;} #hero{padding:20px 18px;} #hero h1{font-size:28px;} #hero .eyebrow{letter-spacing:1px;} }
@media(max-width:600px){#restore-layout{flex-direction:column !important;} #restore-layout > .column{width:100% !important;flex:1 1 auto !important;}}
'''


@lru_cache(maxsize=1)
def get_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    return load_models(str(ROOT/DEFAULT_EMBEDDER), str(ROOT/'ckpts/best.ckpt'), device), device


def run_restoration(image, mode='image', weather='haze', reference=None, original_size=False):
    if image is None:
        raise gr.Error('Upload an image first.')
    if mode not in ('image', 'fused'):
        raise gr.Error('Choose a valid restoration mode.')
    if mode == 'fused' and weather not in dict((v, k) for k, v in WEATHER):
        raise gr.Error('Choose the weather condition for fused restoration.')
    image = ImageOps.exif_transpose(image).convert('RGB')
    if reference is not None:
        reference = ImageOps.exif_transpose(reference).convert('RGB')
        if reference.size != image.size:
            raise gr.Error('The clean reference must have the same original dimensions and show the same scene.')
    with MODEL_LOCK:
        models, device = get_models()
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = restore(image, models, device, size=(224, 224), conditioning=mode,
                         weather=weather if mode == 'fused' else None)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    processed = image.resize((224, 224), Image.Resampling.BICUBIC)
    scores = []
    if reference is not None:
        target = np.asarray(reference.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float64)/255
        for name, sample in [('Input', processed), ('Restored PNG', result)]:
            array = np.asarray(sample, dtype=np.float64)/255
            psnr = peak_signal_noise_ratio(target, array)
            scores.append([name, 'infinite (identical)' if np.isinf(psnr) else f'{psnr:.4f}',
                           f'{structural_similarity(target, array):.6f}'])
        quality_note = 'PSNR/SSIM measured against your clean reference at 224x224, using the exported 8-bit image.'
    else:
        scores = [['Input', 'Needs clean reference', 'Needs clean reference'],
                  ['Restored PNG', 'Needs clean reference', 'Needs clean reference']]
        quality_note = 'Upload a matching clean reference to measure PSNR/SSIM for this image.'
    display_input = image if original_size else processed
    displayed = result.resize(image.size, Image.Resampling.BICUBIC) if original_size else result
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT/f'restored_{uuid4().hex}.png'
    displayed.save(destination)
    metadata = {'conditioning': mode, 'weather': weather if mode == 'fused' else None,
                'device': device, 'processing_size': [224, 224], 'input_size': list(image.size),
                'output_size': list(displayed.size), 'elapsed_seconds': elapsed,
                'has_clean_reference': reference is not None, 'metrics_8bit_png': scores}
    destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    status = f'**Restoration complete** · {displayed.width} × {displayed.height} · {elapsed:.2f}s on {device.upper()}\n\n{quality_note}'
    return (display_input, displayed), displayed, str(destination), scores, status


def build_app():
    reference_metrics, report, report_path = load_comparison()
    with gr.Blocks(title='OmniRestore Studio', analytics_enabled=False) as demo:
        gr.HTML('<div id="hero"><div class="eyebrow">ADVERSE-WEATHER IMAGE RESTORATION</div><h1>OmniRestore Studio</h1><p>Upload a photo. Restore visibility. Compare the result.</p></div>')
        with gr.Tab('Restore an image'):
            with gr.Row(equal_height=False, elem_id="restore-layout"):
                with gr.Column(scale=4):
                    upload = gr.Image(label='Your image', type='pil', sources=['upload'], height=290, format='png')
                    mode = gr.Radio([('Image only · weather unknown', 'image'),
                                     ('Image + weather label', 'fused')], value='image', label='Restoration mode')
                    weather = gr.Dropdown(WEATHER, value='haze', label='Weather condition', visible=False)
                    mode.change(lambda selected: gr.update(visible=selected == 'fused'), mode, weather, queue=False, api_name=False)
                    original = gr.Checkbox(label='Save at original dimensions', value=False,
                                           info='Processing is 224×224. This resizes the saved result back to your image dimensions.')
                    with gr.Accordion('Optional: measure quality with a clean reference', open=False):
                        gr.Markdown('Use the clean version of the **same scene and original dimensions**. A reference is not needed for restoration.')
                        clean = gr.Image(label='Clean reference', type='pil', sources=['upload'], height=180, format='png')
                    button = gr.Button('Restore image', variant='primary', elem_id='restore-button')
                with gr.Column(scale=6):
                    slider = gr.ImageSlider(label='Before / after — drag the divider', type='pil', height=390, format='png', interactive=False)
                    status = gr.Markdown('Your restored image will appear here.')
                    download = gr.DownloadButton('Download restored PNG', interactive=False)
                    quality = gr.Dataframe(headers=['Image', 'PSNR (dB)', 'SSIM'], datatype=['str', 'str', 'str'],
                                           interactive=False, label='Quality of this upload')
                    with gr.Accordion('View restored image', open=False):
                        result = gr.Image(type='pil', label='Restored output', interactive=False, format='png')
            gr.Markdown('**Try a real test image** — selecting an example fills in its matching clean reference.', elem_id='caption')
            examples = []
            for condition in ['haze', 'low']:
                source = ROOT/'data/cdd11/test'/condition/'00002.png'
                target = ROOT/'data/cdd11/test/clear/00002.png'
                if source.is_file() and target.is_file():
                    examples.append([str(source), 'fused', condition, str(target), False])
            if examples:
                gr.Examples(examples, inputs=[upload, mode, weather, clean, original], cache_examples=False,
                            example_labels=['Haze · CDD-11', 'Low light · CDD-11'])
            event = button.click(run_restoration, inputs=[upload, mode, weather, clean, original],
                                 outputs=[slider, result, download, quality, status], api_name='restore', concurrency_limit=1)
            event.then(lambda: gr.update(interactive=True), outputs=download, api_name=False)
        with gr.Tab('Benchmark comparison'):
            gr.Markdown('## Current execution vs owner-reported results\n\nHigher PSNR and SSIM are better. The following values describe the full saved test run, not your uploaded image.')
            gr.Markdown(comparison_table(reference_metrics, report))
            gr.Markdown(f'**Dataset:** {report["pair_count"]:,} CDD-11 test pairs, including 200 clean identity cases. '
                        '**Settings:** 224×224, CUDA, non-shifted windows. Fused mode uses known dataset weather labels. '
                        'The original README does not fully specify the same evaluation protocol; this is a numerical comparison, not verified paper reproduction.')
            gr.Markdown('An “accuracy percentage” is not reported for restoration. For your own uploads, PSNR and SSIM require a matching clean reference.')
            gr.File(value=str(ROOT/'README_COMPARISON.md'), label='Download comparison README', interactive=False)
            gr.File(value=str(report_path), label='Download measured execution report', interactive=False)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    theme = gr.themes.Soft(primary_hue='teal', neutral_hue='slate', font=['Arial', 'sans-serif'], font_mono=['Consolas', 'monospace'])
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', server_port=args.port,
        share=False, inbrowser=False, theme=theme, css=CSS, max_file_size='30mb',
        allowed_paths=[str(OUTPUT)], show_error=False)


if __name__ == '__main__':
    main()
