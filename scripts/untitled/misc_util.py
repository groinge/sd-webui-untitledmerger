import gradio as gr
import re,safetensors.torch,safetensors,torch,os,re
from collections import OrderedDict
from modules.timer import Timer
from modules import sd_models,script_callbacks,shared,sd_unet,sd_hijack,sd_models_config,paths_internal,processing,script_loading,paths,ui_common

import scripts.untitled.common as cmn

networks = script_loading.load_module(os.path.join(paths.extensions_builtin_dir,'Lora','networks.py'))


BASE_SELECTORS = {
    "all":  "",
    "clip": "cond",
    "base": "cond",
    "unet": "model\\.diffusion_model",
    "in":   "model\\.diffusion_model\\.input_blocks",
    "out":  "model\\.diffusion_model\\.output_blocks",
    "mid":  "model\\.diffusion_model\\.middle_block"
}


def target_to_regex(target_input: str|list) -> re.Pattern:
    target_list = target_input if isinstance(target_input,list) else [target_input]

    targets = []
    for target_name in target_list:
        if target_name.endswith(('.','-')):
            target_name = target_name[:-1]
        target = re.split("\.|:",target_name.lower())

        regex = "^"

        if target[0] in BASE_SELECTORS:
            regex += BASE_SELECTORS[target.pop(0)]
            
        for selector in target:
            #Check if the selector qualifies as a number
            if re.search(r'^[\d,-]*$',selector):
                #Turns number inputs like "2-5,10" in to "2|3|4|5|10"
                splitnumeric = set(selector.split(','))
                for segment in splitnumeric.copy():
                    if '-' in segment:
                        a, b = segment.split('-')
                        valuerange = [str(i) for i in range(int(a),int(b)+1)]
                        splitnumeric.remove(segment)
                        splitnumeric.update(valuerange)
                formattedselector = '|'.join(splitnumeric)

                regex += f"\\D*\\.(?:{formattedselector})\\."
            else:
                regex += f".*{re.escape(selector)}"
        regex += ".*$"
        targets.append(regex)
    
    regex = '|'.join(targets)
    return regex
    

versions = {
    "v1":'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight',
    "v2":'cond_stage_model.model.token_embedding.weight',
    'xl':'conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight'
}

def id_checkpoint(name):
    filename = sd_models.get_closet_checkpoint_match(name).filename
    with safetensors.torch.safe_open(filename,framework='pt',device='cpu') as st_file:

        def gettensor(key):
            try:
                return st_file.get_tensor(key)
            except safetensors.SafetensorError:
                return None
            
        keys = st_file.keys()
        
        if versions['v1'] in keys:
            diffusion_model_input = gettensor('model.diffusion_model.input_blocks.0.0.weight')
            dtype = diffusion_model_input.dtype
            if diffusion_model_input.shape[1] == 9:
                return 'v1-inpainting',dtype
            if diffusion_model_input.shape[1] == 8:
                return 'v1-instruct-pix2pix',dtype
            return 'v1',dtype
        
        if versions['xl'] in keys:
            clip_embedder = gettensor('conditioner.embedders.1.model.ln_final.weight')
            if clip_embedder is not None:
                return 'SDXL',clip_embedder.dtype
            return 'SDXL-refiner',gettensor('conditioner.embedders.1.model.ln_final.weight').dtype
            
        if versions['v2'] in keys:
            diffusion_model_input = gettensor('model.diffusion_model.input_blocks.0.0.weight')
            dtype = diffusion_model_input.dtype
            if diffusion_model_input.shape[1] == 9:
                return 'v2-inpainting',dtype
            return 'v2',dtype
            
        
        return 'Unknown',gettensor(keys[0]).dtype
    

class NoCaching:
    def __init__(self):
        self.cachebackup = None

    def __enter__(self):
        self.cachebackup = sd_models.checkpoints_loaded
        sd_models.checkpoints_loaded = OrderedDict()

    def __exit__(self, *args):
        sd_models.checkpoints_loaded = self.cachebackup


def create_name(checkpoints,calcmode,alpha):
    names = []
    try:
        checkpoints = checkpoints[0:3]
    except:pass
    for filename in checkpoints:
        name = os.path.basename(os.path.splitext(filename)[0]).lower()
        segments = re.findall(r'^.{0,10}|[ev]\d{1,3}|(?<=\D)\d{1,3}(?=.*\.)|xl',name)
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


recently_saved = []
recent_save_prefix = '[Recent save] '

def save_state_dict(state_dict,name,settings,timer=None):
    global recently_saved
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

    """recently_saved.insert(0,recent_save_prefix+checkpoint_info.name)
    try:
        recently_saved = recently_saved[0:5]
    except IndexError: pass"""
    
    gr.Info('Model saved as '+filename)
    return checkpoint_info


def load_merged_state_dict(state_dict,checkpoint_info):
    config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    
    for key, weight in state_dict.items():
        state_dict[key] = weight.half()

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


def image_gen(task_id,promptbox,negative_promptbox,steps,sampler_name,width,height,batch_count,batch_size,cfg_scale,seed,
              enable_hr,hr_upscaler,hr_second_pass_steps,denoising_strength,hr_scale,hr_resize_x,hr_resize_y):
    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        prompt=promptbox,
        negative_prompt=negative_promptbox,
        seed=seed,
        sampler_name=sampler_name,
        batch_size=batch_size,
        n_iter=batch_count,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        denoising_strength=denoising_strength,
        do_not_save_grid=True,
        do_not_save_samples=True,
        do_not_reload_embeddings=True
    )

    p.cached_c = [None,None]
    p.cached_hr_c = [None,None]

    processed = processing.process_images(p)

    shared.total_tqdm.clear()
    return processed.images, ui_common.plaintext_to_html('\n'.join(processed.infotexts)), ui_common.plaintext_to_html(processed.comments)
