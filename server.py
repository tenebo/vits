import os
# import requests
from flask import (
    Flask,
    request,
    send_file,
    render_template,
    jsonify,
    Response,
    redirect,
    url_for,
)

import sys
sys.path.append('.')

import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from server_text_processer import normalize_multiline_text

from scipy.io.wavfile import write

download_root = os.getenv(
    "YAIN_TTS_CACHE_HOME", 
    os.path.join(os.path.expanduser("~"), ".cache", "YainTTS")
)

models_path = os.listdir(download_root)

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def synthesize(text,hps,net_g):
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        write("tmp/out.wav",hps.data.sampling_rate, audio)
    return "tmp/out.wav"

hps = utils.get_hparams_from_file("./configs/ljs_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cpu()
_ = net_g.eval()

_ = utils.load_checkpoint("models/NAR_G_60000.pth", net_g, None)

app = Flask(__name__)


@app.after_request
def allow_cors(response):
    response.headers['Access-Control-Allow-Origin'] = "*"
    return response


@app.route("/")
def index():
    return redirect(url_for("text_inference"))


@app.route("/tts-server/text-inference")
def text_inference():
    return render_template("text-inference.html")


@app.route("/tts-server/oc-overlay")
def open_captions_overlay():
    return render_template("oc-overlay.html")


@app.route("/tts-server/api/process-text", methods=["POST"])
def text():
    text = request.json.get("text", "")
    texts = normalize_multiline_text(text)

    return jsonify(texts)


@app.route("/tts-server/api/infer-glowtts")
def infer_glowtts():
    text = request.args.get("text", "")

    if not text:
        return "text shouldn't be empty", 400

    try:
        wav = synthesize(text, hps, net_g)
        return send_file(wav, mimetype="audio/wav", attachment_filename="audio.wav", as_attachment=True)

    except Exception as e:
        return f"Cannot generate audio: {str(e)}", 500


@app.route("/favicon.ico")
def favicon():
    return "I don't have favicon :p", 404


# @app.route("/<path:path>")
# def twip_proxy(path):
#     new_url = request.url.replace(request.host, "twip.kr")
#     resp = requests.request(
#         method=request.method,
#         url=new_url,
#         headers={key: value for (key, value) in request.headers if key != "Host"},
#         data=request.get_data(),
#         cookies=request.cookies,
#         allow_redirects=False,
#     )
#     excluded_headers = [
#         "content-encoding",
#         "content-length",
#         "transfer-encoding",
#         "connection",
#     ]
#     headers = [
#         (name, value)
#         for (name, value) in resp.raw.headers.items()
#         if name.lower() not in excluded_headers
#     ]
#     content = resp.content
#     if new_url.startswith("http://twip.kr/assets/js/alertbox/lib-"):
#         content = (
#             resp.text
#         )
#     response = Response(content, resp.status_code, headers)
#     return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=os.environ.get("TTS_DEBUG", "0") == "1", port=3000)
