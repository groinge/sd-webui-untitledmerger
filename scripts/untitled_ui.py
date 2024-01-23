import gradio as gr
import os,yaml,re
import torch,safetensors,safetensors.torch
from modules import sd_models,script_callbacks,scripts,shared,devices,sd_unet,sd_hijack,sd_models_config,ui_components,paths_internal,ui_loadsave,script_loading,paths
from modules.timer import Timer
from modules.ui_common import create_refresh_button,create_output_panel,plaintext_to_html
from scripts.untitled import merger,misc_util,calcmodes
import scripts.untitled.common as cmn
from copy import deepcopy

networks = script_loading.load_module(os.path.join(paths.extensions_builtin_dir,'Lora','networks.py'))

checkpoints_no_pickles = lambda: [checkpoint for checkpoint in sd_models.checkpoint_tiles() if checkpoint.split(' ')[0].endswith('.safetensors')]

extension_path = scripts.basedir()

ext2abs = lambda *x: os.path.join(extension_path,*x)

symlink_path = ext2abs("scripts","untitled","symlink")
sd_checkpoints_path = os.path.join(paths.models_path,'Stable-diffusion')
        
with open(ext2abs('scripts','examplemerge.yaml'), 'r') as file:
    EXAMPLE = file.read()

calcmode_selection = {}
for calcmode in calcmodes.CALCMODES_LIST:
    calcmode_selection.update({calcmode.name: calcmode})

def on_ui_tabs():
    with gr.Blocks() as ui:
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
                        model_a_info = gr.HTML(plaintext_to_html('None | None',classname='untitled_sd_version'))
                        model_a.change(fn=checkpoint_changed,inputs=model_a,outputs=model_a_info)

                    with gr.Column(variant='compact',min_width=150,scale=slider_scale):
                        with gr.Row():
                            model_b = gr.Dropdown(checkpoints_no_pickles(), label="model_b",scale=slider_scale)
                            swap_models_BC = gr.Button(value='⇆', elem_classes=["tool"],scale=1)
                        model_b_info = gr.HTML(plaintext_to_html('None | None',classname='untitled_sd_version'))
                        model_b.change(fn=checkpoint_changed,inputs=model_b,outputs=model_b_info)

                    with gr.Column(variant='compact',min_width=150,scale=slider_scale):
                        with gr.Row():
                            model_c = gr.Dropdown(checkpoints_no_pickles(), label="model_c",scale=slider_scale)
                            refresh_button = gr.Button(value='🔄', elem_classes=["tool"],scale=1)
                        model_c_info = gr.HTML(plaintext_to_html('None | None',classname='untitled_sd_version'))
                        model_c.change(fn=checkpoint_changed,inputs=model_c,outputs=model_c_info)

                    def swapvalues(x,y): return gr.update(value=y), gr.update(value=x)
                    swap_models_AB.click(fn=swapvalues,inputs=[model_a,model_b],outputs=[model_a,model_b])
                    swap_models_BC.click(fn=swapvalues,inputs=[model_b,model_c],outputs=[model_b,model_c])
                    refresh_button.click(fn=refresh_models,outputs=[model_a,model_b,model_c])

                #### MODE SELECTION
                with gr.Row():
                    mode_selector = gr.Radio(label='Merge mode:',choices=list(calcmode_selection.keys()),value=list(calcmode_selection.keys())[0],scale=3)
                    smooth = gr.Checkbox(label='Smooth Add',info='Filter additions to prevent burning at high weights',show_label=True,scale=1)
                
                ##### MAIN SLIDERS
                with gr.Row(equal_height=True):
                    alpha = gr.Slider(minimum=-1,step=0.01,maximum=2,label="slider_a",info='model_a - model_b',value=0.5,elem_classes=['main_sliders'])
                    beta = gr.Slider(minimum=-1,step=0.01,maximum=2,label="slider_b",info='-',value=0.5,elem_classes=['main_sliders'])
                    gamma = gr.Slider(minimum=-1,step=0.01,maximum=2,label="slider_c",info='-',value=0.25,elem_classes=['main_sliders'])
                    delta = gr.Slider(minimum=-1,step=0.01,maximum=2,label="slider_d",info='-',value=0.25,elem_classes=['main_sliders'])

                mode_selector.change(fn=calcmode_changed, inputs=[mode_selector], outputs=[mode_selector,alpha,beta,gamma,delta],show_progress='hidden')

                with gr.Row(equal_height=True):
                    with gr.Column(variant='panel'):
                        save_name = gr.Textbox(max_lines=1,label='Save checkpoint as:',lines=1,placeholder='Enter name...',scale=2)
                        with gr.Row():
                            save_settings = gr.CheckboxGroup(label = " ",choices=["Autosave","Overwrite","fp16"],value=['fp16'],interactive=True,scale=2,min_width=100)
                            save_loaded = gr.Button(value='Save loaded checkpoint',size='sm',scale=1)
                            save_loaded.click(fn=save_loaded_model, inputs=[save_name,save_settings],outputs=status).then(fn=refresh_models,outputs=[model_a,model_b,model_c])

                    with gr.Column():
                        #### MERGE BUTTONS
                        merge_button = gr.Button(value='Merge',variant='primary')
                        with gr.Row():
                            empty_cache_button = gr.Button(value='Empty Cache')
                            #stop_button = gr.Button(value='Stop merge')
                with gr.Accordion(label='Include/Exclude/Discard',open=False):
                    with gr.Row():
                        with gr.Column():
                            clude = gr.Textbox(max_lines=4,label='Include/Exclude:',info='Entered targets will remain as model_a when set to \'Exclude\', and will be the only ones to be merged if set to \'Include\'. Separate with withspace.',value='',lines=4,scale=4)
                            clude_mode = gr.Radio(label="",info="",choices=["Exclude",("Include exclusively",'include')],value='Exclude',min_width=300,scale=1)
                        discard = gr.Textbox(max_lines=5,label='Discard:',info="Targets will be removed from the model, separate with whitespace.",value='model_ema',lines=5,scale=1)
                    

                with gr.Row(variant='panel'):
                    device_selector = gr.Radio(label='Preferred device/dtype for merging:',info='',choices=['cuda/float16', 'cuda/float32', 'cpu/float32'],value = 'cuda/float16' )
                    worker_count = gr.Slider(step=2,minimum=2,value=cmn.threads,maximum=16,label='Worker thread count:',info=('Relevant for both cuda and CPU merging. Using too many threads can harm performance.'))
                    def worker_count_fn(x): cmn.threads = int(x)
                    worker_count.release(fn=worker_count_fn,inputs=worker_count)
                    device_selector.change(fn=change_preferred_device,inputs=device_selector)

               
                #with gr.Accordion(label='Weight editor',open=True):
                    
            with gr.Column():
                status.render()
                weight_editor = gr.Code(value=EXAMPLE,lines=20,language='yaml',label='')
                #with gr.Tab(label='😫'):
                    #result_gallery, html_info_x, html_info, html_log = create_output_panel("txt2img", shared.opts.outdir_txt2img_samples)   
                with gr.Tab(label='Model keys'):
                    target_tester = gr.Textbox(max_lines=1,label="",info="",interactive=True,placeholder='out.4.tran.norm.weight')
                    target_tester_display = gr.Textbox(max_lines=40,lines=40,label="Targeted keys:",info="",interactive=False)
                    target_tester.change(fn=test_regex,inputs=[target_tester,model_a],outputs=target_tester_display,show_progress='minimal')


            empty_cache_button.click(fn=merger.clear_cache,outputs=status)
            merge_button.click(fn=start_merge, inputs=[mode_selector,model_a,model_b,model_c,alpha,beta,gamma,delta,weight_editor,save_name,save_settings,discard,clude,clude_mode,smooth],outputs=status)

        

    return [(ui, "Untitled merger", "untitled_merger")]

script_callbacks.on_ui_tabs(on_ui_tabs)

WEIGHT_NAMES = ('alpha','beta','gamma','delta')

def start_merge(calcmode,model_a,model_b,model_c,slider_a,slider_b,slider_c,slider_d,editor,save_name,save_settings,discard,clude,clude_mode,smooth):
    calcmode = calcmode_selection[calcmode]
    timer = Timer()
    cmn.stop = False

    targets = re.sub(r'#.*$','',editor.lower(),flags=re.M)
    targets = re.sub(r'\bslider_a\b',str(slider_a),targets,flags=re.M)
    targets = re.sub(r'\bslider_b\b',str(slider_b),targets,flags=re.M)
    targets = re.sub(r'\bslider_c\b',str(slider_c),targets,flags=re.M)
    targets = re.sub(r'\bslider_d\b',str(slider_d),targets,flags=re.M)

    targets_list = targets.split('\n')
    parsed_targets = {}
    for target in targets_list:
        if target != "":
            target = re.sub(r'\s+','',target)
            selector, weights = target.split(':')
            parsed_targets[selector] = {'smooth':smooth}
            for n,weight in enumerate(weights.split(',')):
                try:
                    parsed_targets[selector][WEIGHT_NAMES[n]] = float(weight)
                except ValueError:pass


    checkpoints = []
    for n, model in enumerate((model_a,model_b,model_c)):
        if n+1 > calcmode.input_models:
            checkpoints.append("")
            continue
        name = model.split(' ')[0]
        checkpoint_info = sd_models.get_closet_checkpoint_match(name)
        assert checkpoint_info != None, 'Couldn\'t find checkpoint. '+name
        assert checkpoint_info.filename.endswith('.safetensors'), 'This extension only supports safetensors checkpoints: '+name
        
        checkpoints.append(checkpoint_info.filename)
    else:
        assert n+1 >= calcmode.input_models, "Missing input models"


    discards = re.findall(r'[^\s]+', discard, flags=re.I|re.M)
    cludes = [clude_mode.lower(),*re.findall(r'[^\s]+', clude, flags=re.I|re.M)]

    sd_models.unload_model_weights(shared.sd_model)
    sd_unet.apply_unet("None")
    sd_hijack.model_hijack.undo_hijack(shared.sd_model)
    devices.torch_gc()

    #Actual main merge process begins here:
    state_dict = merger.prepare_merge(calcmode,parsed_targets,checkpoints,discards,cludes,timer)

    merge_name = create_name(checkpoints,calcmode.name,slider_a)

    checkpoint_info = deepcopy(sd_models.get_closet_checkpoint_match(model_a))
    checkpoint_info.short_title = hash(cmn.last_merge_tasks)
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
        segments = re.findall(r'^.{0,10}|[ev]\d{1,3}|(?<=\D)\d{1,3}(?=.*\.)|xl',name) #update this to reduce number spam
        abridgedname = segments.pop(0).title()
        for segment in set(segments):
            abridgedname += "-"+segment.upper()
        names.append(abridgedname)
    new_name = f'{"~".join(names)}_{calcmode.replace(" ","-").upper()}x{alpha}'
    return new_name
        

def save_loaded_model(name,settings):
    if shared.sd_model.sd_checkpoint_info.short_title != hash(cmn.last_merge_tasks):
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
    shared.sd_model.sd_checkpoint_info = checkpoint_info
    shared.sd_model_file = checkpoint_info.filename
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
    if name == "":
        return plaintext_to_html('None | None',classname='untitled_sd_version')
    sdversion, dtype = misc_util.id_checkpoint(name)
    return plaintext_to_html(f"{sdversion} | {str(dtype).split('.')[1]}",classname='untitled_sd_version')


def calcmode_changed(calcmode_name):
    calcmode = calcmode_selection[calcmode_name]

    slider_a_update = gr.update(
        minimum=calcmode.slid_a_config[0],
        maximum=calcmode.slid_a_config[1],
        step=calcmode.slid_a_config[2],
        info=calcmode.slid_a_info
    )

    slider_b_update = gr.update(
        minimum=calcmode.slid_b_config[0],
        maximum=calcmode.slid_b_config[1],
        step=calcmode.slid_b_config[2],
        info=calcmode.slid_b_info
    )

    slider_c_update = gr.update(
        minimum=calcmode.slid_c_config[0],
        maximum=calcmode.slid_c_config[1],
        step=calcmode.slid_c_config[2],
        info=calcmode.slid_c_info
    )

    slider_d_update = gr.update(
        minimum=calcmode.slid_d_config[0],
        maximum=calcmode.slid_d_config[1],
        step=calcmode.slid_d_config[2],
        info=calcmode.slid_d_info
    )

    return gr.update(info = calcmode.description),slider_a_update,slider_b_update,slider_c_update,slider_d_update


class NoHashing:
    def __init__(self):
        self.orig_setting = shared.cmd_opts.no_hashing

    def __enter__(self):
        shared.cmd_opts.no_hashing = True

    def __exit__(self,*args):
        shared.cmd_opts.no_hashing = self.orig_setting


def refresh_models():
    sd_models.list_models()
    checkpoints_list = checkpoints_no_pickles()
    return gr.update(choices=checkpoints_list),gr.update(choices=checkpoints_list),gr.update(choices=checkpoints_list)
