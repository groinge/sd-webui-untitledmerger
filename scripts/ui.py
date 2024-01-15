import gradio as gr
import os,yaml,torch,re,gc
from modules import sd_models,script_callbacks,scripts,shared,devices,sd_unet,sd_hijack,sd_models_config,timer,ui_components,paths_internal
from modules.ui_common import create_refresh_button,create_output_panel,plaintext_to_html
from scripts.untitled import merger,misc_util
import scripts.common as cmn
from safetensors.torch import save_file, safe_open
from copy import deepcopy

checkpoints_no_pickles = lambda: [checkpoint for checkpoint in sd_models.checkpoint_tiles() if checkpoint.split(' ')[0].endswith('.safetensors')]

ext2abs = lambda *x: os.path.join(scripts.basedir(),*x)

with open(ext2abs('scripts','examplemerge.yaml'), 'r') as file:
    EXAMPLE = file.read()

CALCMODE_PRESETS = {
'Weight-Sum':"""  'all':
    weight-sum:
      alpha: slider_a
      sources:
        checkpoint_a: model_a
        checkpoint_b: model_b""",

'Add Difference':"""  'all':
    add:
      alpha: slider_a
      sources:
        checkpoint_a: model_a
        sub:
          sources:
            checkpoint_b: model_b
            checkpoint_c: model_c""",

'Train Difference':"""  'all':
    traindiff:
      alpha: slider_a
      sources:
        checkpoint_a: model_a
        checkpoint_b: model_b
        checkpoint_c: model_c""",

'Extract':"""  'all':
    extract:
      alpha: slider_a
      beta: slider_b
      gamma: slider_c
      sources:
        checkpoint_a: model_a
        checkpoint_b: model_b
        checkpoint_c: model_c"""
#additional models can be inclu
}

def on_ui_tabs():
    with gr.Blocks() as ui:
        with ui_components.ResizeHandleRow():
            with gr.Column():

                #### MODEL SELECTION
                with gr.Row():
                    slider_scale = 5

                    model_a = gr.Dropdown(checkpoints_no_pickles(), label="model_a",scale=slider_scale)
                    swap_models_AB = gr.Button(value='⇆', elem_classes=["tool"],scale=1)

                    model_b = gr.Dropdown(checkpoints_no_pickles(), label="model_b",scale=slider_scale)
                    swap_models_BC = gr.Button(value='⇆', elem_classes=["tool"],scale=1)

                    model_c = gr.Dropdown(checkpoints_no_pickles(), label="model_c",scale=slider_scale)
                    refresh_button = create_refresh_button([model_a,model_b,model_c], lambda:None,lambda: {"choices": checkpoints_no_pickles()},"refresh_checkpoints")
                    refresh_button.update(scale=1)

                    def swapvalues(x,y): return gr.update(value=y), gr.update(value=x)
                    swap_models_AB.click(fn=swapvalues,inputs=[model_a,model_b],outputs=[model_a,model_b])
                    swap_models_BC.click(fn=swapvalues,inputs=[model_b,model_c],outputs=[model_b,model_c])

                #### MODE SELECTION
                mode_selector = gr.Radio(label='Merge mode',choices=list(CALCMODE_PRESETS.keys()),value=list(CALCMODE_PRESETS.keys())[0])

                ##### MAIN SLIDERS
                with gr.Row():
                    alpha = gr.Slider(minimum=-1,maximum=2,label="slider_a (alpha)",value=0.5)
                    beta = gr.Slider(minimum=-1,maximum=2,label="slider_b (beta)",value=0.5)
                    gamma = gr.Slider(minimum=0,maximum=100,label="slider_c (gamma)",value=5)

                status = gr.Textbox(max_lines=1,label="",info="",interactive=False)

                #### MERGE BUTTONS
                with gr.Row():
                    mergebutton = gr.Button(value='Merge',variant='primary')
                    emptycachebutton = gr.Button(value='Empty Cache')
                
                #### BLOCK SLIDERS
                                    
                def createslider(name):
                    slider = gr.Slider(label=name,minimum=0,maximum=1,value=0.5,interactive=False,scale=5,min_width=120)

                    def updateslidervalue(slider):
                        cmn.slidervalues[name] = slider

                    slider.release(fn=updateslidervalue,inputs=slider)
                    cmn.slidervalues[name] = None
                    return slider

                def createblocksliders(side):
                    created_sliders = {}
                    for block_numb in range(0,12):
                        name = side+str(block_numb)
                        created_sliders[name] = createslider(name)
                    return created_sliders
                        
                sliders = {}

                with gr.Accordion(label='Block Weight Sliders',open=True):
                    with gr.Row():
                        gr.Column(scale=4,min_width=0)
                        sliders['clip'] = createslider('clip')
                        gr.Column(scale=4,min_width=0)
                    with gr.Row():
                        gr.Column(scale=1,min_width=0)

                        with gr.Column(scale=5):
                            sliders.update(createblocksliders('in.'))

                        gr.Column(scale=1,min_width=0)
    
                        with gr.Column(scale=5):
                            sliders.update(createblocksliders('out.'))
                                
                        gr.Column(scale=1,min_width=0)
                    with gr.Row():
                        gr.Column(scale=4,min_width=0)
                        sliders['mid'] = createslider('mid')
                        gr.Column(scale=4,min_width=0)
                    with gr.Row():
                        slidertoggles = gr.CheckboxGroup(choices=list(sliders.keys()))

                        def toggle_slider(inputs):
                            enabled_sliders = inputs.pop(slidertoggles)
                            updates = {}
                            for component,value in inputs.items():
                                if component.label in enabled_sliders:
                                    updates[component] = gr.update(interactive=True)
                                    cmn.slidervalues[component.label] = value
                                else:
                                    updates[component] = gr.update(interactive=False)
                                    cmn.slidervalues[component.label] = None
                            return updates

                        slidertoggles.change(fn=toggle_slider,inputs={slidertoggles,*list(sliders.values())},outputs={*list(sliders.values())},show_progress='hidden')
                    with gr.Row():
                        enable_all = gr.Button(value="Enable all")
                        disable_all = gr.Button(value="Disable all")

                        enable_all.click(fn=lambda x: gr.update(value = list(sliders.keys())),outputs=slidertoggles)
                        disable_all.click(fn=lambda: gr.update(value = []),outputs=slidertoggles)
                with gr.Accordion(label='Recipe editor',open=True):
                    recipe_editor = gr.Code(value=EXAMPLE,lines=20,language='yaml',label='yaml')
                    with gr.Accordion(label='Test target selector',open=False):
                        target_tester = gr.Textbox(max_lines=1,label="",info="",interactive=True,placeholder='out.4.tran.norm.weight')
                        target_tester_display = gr.Textbox(max_lines=40,lines=40,label="Targeted keys:",info="",interactive=False)
                        target_tester.change(fn=test_regex,inputs=[target_tester,model_a],outputs=target_tester_display,show_progress='minimal')

            with gr.Column():
                result_gallery, html_info_x, html_info, html_log = create_output_panel("txt2img", shared.opts.outdir_txt2img_samples)    

            emptycachebutton.click(fn=clear_cache)
            mergebutton.click(fn=start_merge, inputs=[mode_selector,model_a,model_b,model_c,alpha,beta,gamma,recipe_editor],outputs=status)
    return [(ui, "Untitled merger", "untitled_merger")]

script_callbacks.on_ui_tabs(on_ui_tabs)


def start_merge(calcmode,model_a,model_b,model_c,slider_a,slider_b,slider_c,editor):
    mTimer = timer.Timer()

    model_variables = {'model_a':model_a.split(' ')[0],'model_b':model_b.split(' ')[0],'model_c':model_c.split(' ')[0]}

    if editor == '':
        editor = 'targets:\n'

    recipe_str = re.sub(r'(?<=targets:\n)',CALCMODE_PRESETS[calcmode]+'\n',editor)

    for model in model_variables.copy().keys():
        if not model_variables[model] or not re.search(f"^[^#\\n]*:\\s*{model}\\b$",recipe_str,re.M):
            del model_variables[model]

    for variable, sub in {'slider_a':float(slider_a),'slider_b':float(slider_b),'slider_c':float(slider_c)}.items():
        recipe_str = re.sub(f"\\b{variable}\\b",str(sub),recipe_str,flags=re.I|re.M)

    recipe = yaml.safe_load(recipe_str)

    additional_models = recipe.get('checkpoints') or {}
    checkpoints = []

    for alias,name in {**model_variables, **additional_models}.items():
        checkpoint_info = sd_models.get_closet_checkpoint_match(name)

        assert checkpoint_info != None, 'Couldn\'t find checkpoint: '+name
        assert checkpoint_info.filename.endswith('.safetensors'), 'This extension only supports safetensors checkpoints: '+name
        
        checkpoints.append(checkpoint_info.filename)
        recipe_str = re.sub(f"\\b{alias}\\b",name,recipe_str,flags=re.I|re.M)

    recipe = yaml.safe_load(recipe_str)

    for name,value in cmn.slidervalues.items():
        if value is not None:
            recipe['targets'][name] = value

    recipe['checkpoints'] = checkpoints
    recipe['primary_checkpoint'] = checkpoints[0]

    sd_models.unload_model_weights(shared.sd_model)
    sd_unet.apply_unet("None")
    sd_hijack.model_hijack.undo_hijack(shared.sd_model)
    devices.torch_gc()

    #Merge starts here:
    state_dict = merger.prepare_merge(recipe,mTimer)

    checkpoint_info = deepcopy(sd_models.get_closet_checkpoint_match(os.path.basename(recipe['primary_checkpoint'])))
    checkpoint_info.name_for_extra = str(hash(cmn.last_merge_tasks))

    #save_file(state_dict,os.path.join(paths_internal.models_path,'Stable-diffusion','crap.safetensors'))
    
    load_merged_state_dict(state_dict,checkpoint_info)
    mTimer.record('Load model')
    del state_dict
    devices.torch_gc()

    message = 'Merge completed in ' + mTimer.summary()
    print(message)
    return message


def clear_cache():
    cmn.tensor_cache.__init__(cmn.cache_size)
    gc.collect()
    devices.torch_gc()
    torch.cuda.empty_cache()


def load_merged_state_dict(state_dict,checkpoint_info):
    config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)

    if shared.sd_model.used_config == config:
        print('Loading weights using already loaded model...')

        sd_models.load_model_weights(shared.sd_model, checkpoint_info, state_dict, timer.Timer())
        
        sd_hijack.model_hijack.hijack(shared.sd_model)

        script_callbacks.model_loaded_callback(shared.sd_model)

        sd_models.model_data.set_sd_model(shared.sd_model)
        sd_unet.apply_unet()
    else:
        sd_models.model_data.__init__()
        sd_models.load_model(checkpoint_info=checkpoint_info, already_loaded_state_dict=state_dict)


def test_regex(input,model_a):
    regex = misc_util.target_to_regex(input)

    path = sd_models.get_closet_checkpoint_match(model_a).filename

    with safe_open(path,framework='pt',device='cpu') as file:
        keys = file.keys()

    selected_keys = re.findall(regex,'\n'.join(keys),re.M)
    joined = '\n'.join(selected_keys)
    return  f'Matched keys: {len(selected_keys)}\n{joined}'