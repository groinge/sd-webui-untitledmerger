import gradio as gr
import os,yaml,torch,time
from modules import ui_components,sd_models,script_callbacks,scripts,shared,paths,paths_internal,devices,sd_unet,sd_hijack,sd_models_config,timer
from modules.ui_common import create_refresh_button
from scripts.merging.merger import prepare_merge
from safetensors.torch import save_file
from copy import deepcopy

checkpoints_no_pickles = lambda: [checkpoint for checkpoint in sd_models.checkpoint_tiles() if checkpoint.split(' ')[0].endswith('.safetensors')]

ext2abs = lambda *x: os.path.join(scripts.basedir(),*x)

with open(ext2abs('scripts','examplemerge.yaml'), 'r') as file:
    EXAMPLE = file.read()


def on_ui_tabs():
    with gr.Blocks() as ui:
        with gr.Column():
            with gr.Row():
                """model_a = gr.Dropdown(checkpoints_no_pickles(), label="Primary model (A)")
                create_refresh_button(model_a, sd_models.list_models, lambda: {"choices": checkpoints_no_pickles()}, "refresh_checkpoint_A")
                swap_models_AB = gr.Button(value='⇆', elem_classes=["tool"])

                model_b = gr.Dropdown(checkpoints_no_pickles(), label="Secondary model (B)")
                create_refresh_button(model_b, sd_models.list_models, lambda: {"choices": checkpoints_no_pickles()}, "refresh_checkpoint_B")
                swap_models_BC = gr.Button(value='⇆', elem_classes=["tool"])

                model_c = gr.Dropdown(checkpoints_no_pickles(), label="Tertiary model (C)")
                create_refresh_button(model_c, sd_models.list_models, lambda: {"choices": checkpoints_no_pickles()}, "refresh_checkpoint_C")"""
            status = gr.Textbox(max_lines=1,label="",info="")
            recipe = gr.Code(value=EXAMPLE,lines=30,language='yaml',label='Recipe')
            mergebutton = gr.Button(value='Merge',variant='primary')
            mergebutton.click(fn=start_merge, inputs=recipe,outputs=status)
    return [(ui, "Untitled merger", "untitled_merger")]

script_callbacks.on_ui_tabs(on_ui_tabs)

def start_merge(recipe_str):
    start_time = time.time()
    recipe = yaml.safe_load(recipe_str)

    secondary_checkpoints = recipe.get('checkpoints') or {}
    checkpoints = {}

    for alias,identifier in {**recipe['primary_checkpoint'], **secondary_checkpoints}.items():
        filename = sd_models.get_closet_checkpoint_match(identifier).filename
        assert filename.endswith('.safetensors'), 'This extension only supports safetensors checkpoints'
        checkpoints[alias] = filename
    
    recipe['checkpoints'] = checkpoints

    sd_models.unload_model_weights(shared.sd_model)
    devices.torch_gc()

    #Merge starts here:
    state_dict = prepare_merge(recipe)

    checkpoint_info = sd_models.get_closet_checkpoint_match(list(recipe['primary_checkpoint'].values())[0])
 
    #new_cpi = modifycpinfo(checkpoint_info)

    #save_file(state_dict,os.path.join(paths_internal.models_path,'Stable-diffusion','crap.safetensors'))
    
    load_merged_state_dict(state_dict,checkpoint_info)
    return f'Completed merge in {str(time.time() - start_time)[0:4]} seconds'


def load_merged_state_dict(state_dict,checkpoint_info):
    config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)

    if shared.sd_model.used_config == config:
        sd_unet.apply_unet("None")
        sd_hijack.model_hijack.undo_hijack(shared.sd_model)

        sd_models.load_model_weights(shared.sd_model, checkpoint_info, state_dict, timer.Timer())

        sd_hijack.model_hijack.hijack(shared.sd_model)

        script_callbacks.model_loaded_callback(shared.sd_model)

        sd_models.model_data.set_sd_model(shared.sd_model)
        sd_unet.apply_unet()
    else:
        sd_models.model_data.__init__()
        sd_models.load_model(checkpoint_info=checkpoint_info, already_loaded_state_dict=state_dict)

