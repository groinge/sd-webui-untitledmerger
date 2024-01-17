import gradio as gr
import os,yaml,re
import torch,safetensors,safetensors.torch
from modules import sd_models,script_callbacks,scripts,shared,devices,sd_unet,sd_hijack,sd_models_config,ui_components,paths_internal,ui_loadsave,script_loading,paths
from modules.timer import Timer
from modules.ui_common import create_refresh_button,create_output_panel
from scripts.untitled import merger,misc_util
import scripts.common as cmn
from copy import deepcopy

networks = script_loading.load_module(os.path.join(paths.extensions_builtin_dir,'Lora','networks.py'))

checkpoints_no_pickles = lambda: [checkpoint for checkpoint in sd_models.checkpoint_tiles() if checkpoint.split(' ')[0].endswith('.safetensors')]

extension_path = scripts.basedir()

ext2abs = lambda *x: os.path.join(extension_path,*x)

with open(ext2abs('scripts','examplemerge.yaml'), 'r') as file:
    EXAMPLE = file.read()

CALCMODE_PRESETS = {
'Weight-Sum':"""  'all':
    weight-sum:
      alpha: slider_a
      sources:
        checkpoint_a: model_a
        checkpoint_b: model_b""",

'Combined similarity':"""  'all':
    similarity:
      alpha: slider_a
      beta: 0
      gamma: slider_c
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

'Extract (dis)similarity':"""  'all':
    extract:
      alpha: slider_a
      beta: slider_b
      gamma: slider_c
      sources:
        checkpoint_a: model_a
        checkpoint_b: model_b
        checkpoint_c: model_c""",

'Add disimilarity':"""  'all':
    add:
      alpha: slider_b
      sources:
        checkpoint_a: model_a
        similarity:
          alpha: slider_a
          beta: 1
          gamma: slider_c
          sources:
            checkpoint_b: model_b
            checkpoint_c: model_c""",

'Smooth Add Difference':"""  'all':
    add:
      alpha: slider_a
      sources:
        checkpoint_a: model_a
        smooth:
          sources:
            sub:
              sources:
                checkpoint_b: model_b
                checkpoint_c: model_c""",

'Smooth Add disimilarity':"""  'all':
    add:
      alpha: slider_b
      sources:
        checkpoint_a: model_a
        smooth:
          sources:
            similarity:
              alpha: slider_a
              beta: 1
              gamma: slider_c
              sources:
                checkpoint_b: model_b
                checkpoint_c: model_c"""
}

def centered_plaintext_html(x):
    return f"<html><head><style>p {{text-align: center;}}</style></head><body><p>{x}</p></body></html>"
    


def on_ui_tabs():
    with gr.Blocks() as cmn.ui:
        with ui_components.ResizeHandleRow():
            with gr.Column():
                status = gr.Textbox(max_lines=1,label="",info="",interactive=False,render=False)
                #### MODEL SELECTION
                with gr.Row():
                    slider_scale = 5
                    with gr.Column(variant='compact',min_width=150,scale=slider_scale):
                        with gr.Row():
                            model_a = gr.Dropdown(checkpoints_no_pickles(), label="model_a",scale=slider_scale)
                            swap_models_AB = gr.Button(value='⇆', elem_classes=["tool"],scale=1)
                        model_a_info = gr.HTML(centered_plaintext_html('None | None'))
                        model_a.change(fn=checkpoint_changed,inputs=model_a,outputs=model_a_info)

                    with gr.Column(variant='compact',min_width=150,scale=slider_scale):
                        with gr.Row():
                            model_b = gr.Dropdown(checkpoints_no_pickles(), label="model_b",scale=slider_scale)
                            swap_models_BC = gr.Button(value='⇆', elem_classes=["tool"],scale=1)
                        model_b_info = gr.HTML(centered_plaintext_html('None | None'))
                        model_b.change(fn=checkpoint_changed,inputs=model_b,outputs=model_b_info)

                    with gr.Column(variant='compact',min_width=150,scale=slider_scale):
                        with gr.Row():
                            model_c = gr.Dropdown(checkpoints_no_pickles(), label="model_c",scale=slider_scale)
                            refresh_button = create_refresh_button([model_a,model_b,model_c], sd_models.list_models,lambda: {"choices": checkpoints_no_pickles()},"refresh_checkpoints")
                            refresh_button.update(scale=1,min_width=50)
                        model_c_info = gr.HTML(centered_plaintext_html('None | None'))
                        model_c.change(fn=checkpoint_changed,inputs=model_c,outputs=model_c_info)

                    def swapvalues(x,y): return gr.update(value=y), gr.update(value=x)
                    swap_models_AB.click(fn=swapvalues,inputs=[model_a,model_b],outputs=[model_a,model_b])
                    swap_models_BC.click(fn=swapvalues,inputs=[model_b,model_c],outputs=[model_b,model_c])


                #### MODE SELECTION
                mode_selector = gr.Radio(label='Merge mode:',choices=list(CALCMODE_PRESETS.keys()),value=list(CALCMODE_PRESETS.keys())[0])
                
                ##### MAIN SLIDERS
                with gr.Row():
                    alpha = gr.Slider(minimum=-1,step=0.01,maximum=2,label="slider_a (alpha)",value=0.5)
                    beta = gr.Slider(minimum=-1,step=0.01,maximum=2,label="slider_b (beta)",value=0.5)
                    gamma = gr.Slider(minimum=0,step=0.01,maximum=2,label="slider_c (gamma)",value=0.25)   

                with gr.Row():
                    with gr.Column(variant='panel'):
                        save_name = gr.Textbox(max_lines=1,label='Save checkpoint as:',lines=1,placeholder='Enter name...',scale=2)
                        with gr.Row():
                            save_settings = gr.CheckboxGroup(label = " ",choices=["Autosave","Overwrite","fp16"],value=['fp16'],interactive=True,scale=2,min_width=100)
                            save_loaded = gr.Button(value='Save loaded checkpoint',size='sm',scale=1)
                            save_loaded.click(fn=save_loaded_model, inputs=[save_name,save_settings],outputs=status)
                    with gr.Column():
                        #### MERGE BUTTONS
                        merge_button = gr.Button(value='Merge',variant='primary')
                        with gr.Row():
                            empty_cache_button = gr.Button(value='Empty Cache')

                discard = gr.Textbox(max_lines=1,label='Discard:',info="Targets will be removed from the model, separate with whitespace",value='model_ema',lines=1,scale=2)
                    

                with gr.Row(variant='panel'):
                    device_selector = gr.Radio(label='Preferred device/dtype for merging:',info='float16 is probably useless',choices=['cuda/float16', 'cuda/float32', 'cpu/float32'],value = 'cuda/float32' )
                    worker_count = gr.Slider(step=2,minimum=2,value=cmn.threads,maximum=16,label='Worker thread count:',info=('Relevant for both cuda and CPU merging. Using too many threads can harm performance.'))
                    def worker_count_fn(x): cmn.threads = int(x)
                    worker_count.release(fn=worker_count_fn,inputs=worker_count)
                    device_selector.change(fn=change_preferred_device,inputs=device_selector)

               
                #Block sliders
                                    
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

                with gr.Accordion(label='Block Weight Sliders',open=False):
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
                status.render()
                with gr.Tab(label='😫'):
                    result_gallery, html_info_x, html_info, html_log = create_output_panel("txt2img", shared.opts.outdir_txt2img_samples)   


            empty_cache_button.click(fn=merger.clear_cache,outputs=status)
            merge_button.click(fn=start_merge, inputs=[mode_selector,model_a,model_b,model_c,alpha,beta,gamma,recipe_editor,save_name,save_settings,discard],outputs=status)

        

    return [(cmn.ui, "Untitled merger", "untitled_merger")]

script_callbacks.on_ui_tabs(on_ui_tabs)


def start_merge(calcmode,model_a,model_b,model_c,slider_a,slider_b,slider_c,editor,save_name,save_settings,discard):
    timer = Timer()

    model_variables = {'model_a':model_a.split(' ')[0],'model_b':model_b.split(' ')[0],'model_c':model_c.split(' ')[0]}

    if editor == '':
        editor = 'targets:\n'

    if not re.search(r'''^  all|^  'all'|^  "all"''',editor,flags=re.I|re.M):
        recipe_str = re.sub(r'(?<=targets:\n)',CALCMODE_PRESETS[calcmode]+'\n',editor)
    else:
        recipe_str = editor

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
        recipe_str = re.sub(f"\\b{re.escape(alias)}\\b",name,recipe_str,flags=re.I|re.M)

    recipe = yaml.safe_load(recipe_str)

    for name,value in cmn.slidervalues.items():
        if value is not None:
            recipe['targets'][name] = value

    recipe['checkpoints'] = checkpoints
    recipe['primary_checkpoint'] = checkpoints[0]
    recipe['discard'] = re.findall(r'[^\s]+', discard, flags=re.I|re.M)

    sd_models.unload_model_weights(shared.sd_model)
    sd_unet.apply_unet("None")
    sd_hijack.model_hijack.undo_hijack(shared.sd_model)
    devices.torch_gc()

    #Actual main merge process begins here:

    state_dict = merger.prepare_merge(recipe,timer)

    merge_name = create_name(checkpoints,calcmode,slider_a)
    checkpoint_info = deepcopy(sd_models.get_closet_checkpoint_match(os.path.basename(recipe['primary_checkpoint'])))
    checkpoint_info.short_title = cmn.last_merge_tasks_hash
    checkpoint_info.name_for_extra = '_TEMP_MERGE_'+merge_name

    if 'Autosave' in save_settings:
        checkpoint_info = save_state_dict(state_dict,save_name or merge_name,save_settings,timer)
    
    load_merged_state_dict(state_dict,checkpoint_info)
    timer.record('Load model')
    del state_dict
    devices.torch_gc()

    message = 'Merge completed in ' + timer.summary()
    print(message)
    return message


def load_merged_state_dict(state_dict,checkpoint_info):
    config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)

    if shared.sd_model.used_config == config:
        print('Loading weights using already loaded model...')

        load_timer = Timer()
        sd_models.load_model_weights(shared.sd_model, checkpoint_info, state_dict, load_timer)
        print('Loaded weights in: '+load_timer.summary())

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

    with safetensors.torch.safe_open(path,framework='pt',device='cpu') as file:
        keys = file.keys()

    selected_keys = re.findall(regex,'\n'.join(keys),re.M)
    joined = '\n'.join(selected_keys)
    return  f'Matched keys: {len(selected_keys)}\n{joined}'


def create_name(checkpoints,calcmode,alpha):
    names = []
    try:
        checkpoints = checkpoints[0:3]
    except:pass
    for filename in checkpoints:
        name = os.path.basename(os.path.splitext(filename)[0]).lower()
        segments = re.findall(r'^\w{0,10}|[ev]\d{1,3}|(?<=\D)\d{1,3}(?=.*\.)|xl',name)
        abridgedname = segments.pop(0).title()
        for segment in set(segments):
            abridgedname += "-"+segment.upper()
        names.append(abridgedname)
    new_name = f'{"~".join(names)}_{calcmode.replace(" ","-").upper()}x{alpha}'
    return new_name
        

def save_loaded_model(name,settings):
    if shared.sd_model.sd_checkpoint_info.short_title != cmn.last_merge_tasks_hash:
        gr.Warning('Loaded model is not a unsaved merged model.')
        return

    sd_unet.apply_unet("None")
    sd_hijack.model_hijack.undo_hijack(shared.sd_model)

    with torch.no_grad():
        for module in shared.sd_model.modules():
            networks.network_restore_weights_from_backup(module)

    state_dict = shared.sd_model.state_dict()

    name = name or shared.sd_model.sd_checkpoint_info.name_for_extra.replace('_TEMP_MERGE_','')

    checkpoint_info = save_state_dict(state_dict,name,settings)
    load_merged_state_dict(state_dict,checkpoint_info)
    return 'Model saved as: '+checkpoint_info.filename

    
def save_state_dict(state_dict,name,settings,timer=None):
    fileext = ".fp16.safetensors" if 'fp16' in settings else '.safetensors'

    filename_no_ext = os.path.join(paths_internal.models_path,'Stable-diffusion',name)
    try:
        filename_no_ext = filename_no_ext[0:225]
    except: pass

    filename = filename_no_ext+fileext
    if 'Overwrite' not in settings:
        n = 1
        while os.path.exists(filename):
            filename = f"{filename_no_ext}_{n}{fileext}"
            n+=1

    if 'fp16' in settings:
        for key,tensor in state_dict.items():
            state_dict[key] = tensor.type(torch.float16)

    try:
        safetensors.torch.save_file(state_dict,filename)
    except safetensors.SafetensorError:
        print('Failed to save checkpoint. Applying contiguous to tensors and trying again...')
        for key,tensor in state_dict.items():
            state_dict[key] = tensor.contiguous()
        safetensors.torch.save_file(state_dict,filename)

    try:
        timer.record('Save checkpoint')
    except: pass

    checkpoint_info = sd_models.CheckpointInfo(filename)
    checkpoint_info.register()

    gr.Info('Model saved as '+filename)
    return checkpoint_info


def change_preferred_device(input):
    cmn.device,dtype = input.split('/')
                     
    if dtype == 'float16': cmn.precision=torch.float16
    else: cmn.precision = torch.float32

def checkpoint_changed(name):
    sdversion, dtype = misc_util.id_checkpoint(name)
    return centered_plaintext_html(f"{sdversion} | {str(dtype).split('.')[1]}")