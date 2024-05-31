###################################
# Sample a speaker from Gaussian.
import gradio as gr
import ChatTTS
from ChatTTS.core import Chat
import numpy as np
#from ChatTTS.experimental.llm import llm_api

import torch
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

'''
#大语言模型问答
API_KEY = ''
client = llm_api(api_key=API_KEY,
        base_url="",
        model="gpt-3.5-turbo")

user_question = '四川有哪些好吃的美食呢?'
text = client.call(user_question)
'''

chat = Chat()
chat.load_models()

styles = {'prompt': '[oral_3][laugh_1][break_2]'}


rand_spk = torch.randn(768)


params_infer_code = {
  'spk_emb': rand_spk, # 随机音色 
  'temperature': .5, # using custom temperature
  'top_P': 0.5, # top P decode
  'top_K': 10, # top K decode
  'prompt': '[speed_5]'
}

params_refine_text = styles

def change_style(sty):
    index = int(sty.split(' ')[1]) - 1
    global params_refine_text
    global styles
    params_refine_text = styles[index]

def change_temperature(temp):
    global params_infer_code
    params_infer_code['temperature'] = temp

def change_top_P(top_p):
    global params_infer_code
    params_infer_code['top_P'] = top_p

def change_top_K(top_k):
    global params_infer_code
    params_infer_code['top_K'] = top_k


def change_speed(speed):
    global params_infer_code
    params_infer_code['prompt'] = f'[speed_{speed*10+5}]'


def change_olb(oral, laugh, break_):
    global styles
    styles['prompt'] = f'[oral_{oral}][laugh_{laugh}][break_{break_}]'
    



###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7) 
# to generate special token in text to synthesize.
# 句子级别的控制
def sentence_level_control(text):
    global params_refine_text
    global params_infer_code
    global chat
    wav = chat.infer(text, params_refine_text=params_refine_text, params_infer_code=params_infer_code)
    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000
    return (sample_rate, audio_data)

###################################
# 单词级别的控制
def word_level_control(text):
    global params_infer_code
    wav = chat.infer(text, skip_refine_text=True, params_infer_code=params_infer_code)
    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000
    return (sample_rate, audio_data)

with gr.Blocks(theme=gr.themes.Soft(),title = "ChatTTS-UI") as demo:
    gr.Markdown(
    """
    # My ChatTTS-UI
    Text-to-speech synthesis with ChatTTS.Only use for testing.
    """)
    with gr.Column():
        text_input = gr.Textbox(lines=1, label="Input Text")
        covert_btn = gr.Button("Convert Text to Speech",scale=0)
        covert_btn.click(fn = sentence_level_control, inputs = [text_input],outputs = [gr.Audio(label = '生成语音')])
        

    with gr.Row():
        temperature = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.5, label="Temperature")
        top_P = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.5, label="Top P")
        top_K = gr.Slider(minimum=1, maximum=50, step=1, value=10, label="Top K")

        temperature.change(fn = change_temperature, inputs = [temperature], outputs = None)
        top_P.change(fn = change_top_P, inputs = [top_P], outputs = None)
        top_K.change(fn = change_top_K, inputs = [top_K], outputs = None)
        
    
    # use oral_(0-9), laugh_(0-2), break_(0-7)
    with gr.Row():
        oral = gr.Slider(minimum=0, maximum=9, step=1, value=3, label="Oral")
        laugh = gr.Slider(minimum=0, maximum=2, step=1, value=1, label="Laugh")
        break_ = gr.Slider(minimum=0, maximum=7, step=1, value=3, label="Break")
        speed = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Speaker's speed")

        @gr.on(inputs = [oral, laugh, break_], outputs = None)
        def change_olb(oral, laugh, break_):
            global styles
            styles['prompt'] = f'[oral_{oral}][laugh_{laugh}][break_{break_}]'
                
        speed.change(fn = change_speed, inputs = [speed], outputs = None)


        
demo.launch()
