import gradio as gr
import os,yaml,torch,time,re,gc
from modules import sd_models,script_callbacks,scripts,shared,devices,sd_unet,sd_hijack,sd_models_config,timer,ui_components
from modules.ui_common import create_refresh_button,create_output_panel,plaintext_to_html
from scripts.untitled import merger
import scripts.common as cmn
from safetensors.torch import save_file
from copy import deepcopy

checkpoints_no_pickles = lambda: [checkpoint for checkpoint in sd_models.checkpoint_tiles() if checkpoint.split(' ')[0].endswith('.safetensors')]

ext2abs = lambda *x: os.path.join(scripts.basedir(),*x)

with open(ext2abs('scripts','examplemerge.yaml'), 'r') as file:
    EXAMPLE = file.read()

CALCMODENAMES = ('Weight-sum', 'Add difference', 'Train Difference', 'Extract')

visualize_slider = """(x) => {
    const sliders = document.getElementsByClassName('$$$');

    for (let i = 0; i < sliders.length; i++) {
        let slider = sliders[i].querySelector('[name="cowbell"]');
        if (slider.disabled) {
            slider.value = x;
        }
    }
}"""

def on_ui_tabs():
    with gr.Blocks() as ui:
        with ui_components.ResizeHandleRow():
            with gr.Column(variant='panel'):

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
                mode_selector = gr.Radio(label='Merge mode',choices=CALCMODENAMES,value=CALCMODENAMES[0])

                ##### MAIN SLIDERS
                with gr.Row():
                    alpha = gr.Slider(minimum=-1,maximum=2,label="alpha",value=0.5)
                    beta = gr.Slider(minimum=-1,maximum=2,label="beta",value=0.5)
                    gamma = gr.Slider(minimum=0,maximum=100,label="gamma",value=5)

                status = gr.Textbox(max_lines=1,label="",info="",interactive=False)
                """recipe = gr.Code(value=EXAMPLE,lines=30,language='yaml',label='Recipe')"""

                #### MERGE BUTTONS
                with gr.Row():
                    mergebutton = gr.Button(value='Merge',variant='primary')
                    emptycachebutton = gr.Button(value='Empty Cache')
                
                #### BLOCK SLIDERS
                with gr.Row():
                    blocksliderclass =  "blockslider"

                    def createsliders(side):
                        sliders = {}
                        checkboxes = []

                        for block_numb in range(0,12):
                            name = side+str(block_numb)
                            with gr.Row():
                                chkbox = gr.Checkbox(label=' ',info=' ',scale=1,min_width=5)

                                def toggleinteractive(chkbox):
                                    cmn.slidervalues[name][0] = chkbox
                                    return gr.update(interactive=chkbox)

                                slider = gr.Slider(label=name,elem_classes=[blocksliderclass],minimum=0,maximum=1,value=0.5,interactive=False,scale=20)

                                def updateslidervalue(slider):
                                    cmn.slidervalues[name][1] = slider

                                slider.release(fn=updateslidervalue,inputs=slider)
                                chkbox.input(fn=toggleinteractive,inputs=chkbox,outputs=slider)

                                cmn.slidervalues[name] = [False,0.5]
                                sliders[name] = slider
                                checkboxes.append(chkbox)
                        return sliders
                    
                    blocksliders = {}

                    with gr.Column(scale=2,min_width=0):pass

                    with gr.Column(scale=4):
                        blocksliders.update( createsliders('in.') )

                    with gr.Column(scale=1,min_width=0):pass
   
                    with gr.Column(scale=4):
                        blocksliders.update( createsliders('out.') )
                            
                    with gr.Column(scale=2,min_width=0):pass
                    

                alpha.release(fn=None,_js=visualize_slider.replace('$$$',blocksliderclass),inputs=alpha)

            with gr.Column():
                result_gallery, html_info_x, html_info, html_log = create_output_panel("txt2img", shared.opts.outdir_txt2img_samples)    

            emptycachebutton.click(fn=clear_cache)
            mergebutton.click(fn=basic_merge, inputs=[mode_selector,model_a,model_b,model_c,alpha,beta,gamma],outputs=status)
    return [(ui, "Untitled merger", "untitled_merger")]

script_callbacks.on_ui_tabs(on_ui_tabs)


def basic_merge(calcmode,model_a,model_b,model_c,slider_a,slider_b,slider_c):
    blockweights = {}

    for name,values in cmn.slidervalues.items():
        enabled,value = values
        if enabled:
            blockweights[name] = value

    recipe = {}

    #https://youtu.be/2617GBhF3lY
    a = model_a.split(' ')[0]
    b = model_b.split(' ')[0]
    c = model_c.split(' ')[0]
    if calcmode in ['Weight-sum']:
        def calcrecipe(x): 
            return {'weight-sum': {
                'alpha': x,
                'sources': {
                    'checkpoint_a':a,
                    'checkpoint_b':b}}}
        
        checkpoints = (a,b)
    elif calcmode in ('Add difference', ):
        diffmode = {'sub':{'sources':{'checkpoint_b':b,'checkpoint_c':c}}}
        def calcrecipe(x): 
            return {'add': {
                'alpha': x,
                'sources': {
                    'checkpoint':a,
                    **diffmode}}}
        
        checkpoints = (a,b,c)
    elif calcmode == 'Extract':
        def calcrecipe(x): 
            return {'extract':{
                    'alpha': slider_a,
                    'beta': slider_b,
                    'gamma': slider_c,
                    'sources':{'checkpoint_a':a,'checkpoint_b':b,'checkpoint_c':c}}}
        
        checkpoints = (a,b,c)
    elif calcmode == 'Train Difference':
        def calcrecipe(x): 
            return {'traindiff':{
                'alpha': x,
                'sources':{'checkpoint_a':a,'checkpoint_b':b,'checkpoint_c':c}}}
        
        checkpoints = (a,b,c)

    checkpoint_paths = []
    for checkpoint in checkpoints:
        checkpoint_info = sd_models.get_closet_checkpoint_match(checkpoint)
   
        assert checkpoint_info != None, 'Couldn\'t find checkpoint: '+checkpoint
        assert checkpoint_info.filename.endswith('.safetensors'), 'This extension only supports safetensors checkpoints: '+checkpoint

        checkpoint_paths.append(checkpoint_info.filename)

    targets = dict({'all': calcrecipe(slider_a)}|{k:calcrecipe(v) for k,v in blockweights.items()})
    recipe['checkpoints'] = checkpoint_paths
    recipe['primary_checkpoint'] = checkpoint_paths[0]
    recipe['targets'] = targets

    return start_merge(recipe)

"""def start_merge(recipe_str,model_a,model_b,model_c,slider_a,slider_b,slider_c):
    model_variables = {'model_a':model_a.split(' ')[0],'model_b':model_b.split(' ')[0],'model_c':model_c.split(' ')[0]}

    for model in model_variables.copy().keys():
        if not model_variables[model] or not re.search(f"^[^#\\n]*:\\s*{model}\\b$",recipe_str,re.M):
            del model_variables[model]


    for variable, sub in {'slider_a':float(slider_a),'slider_c':float(slider_b),'slider_c':float(slider_c)}.items():
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
    recipe['checkpoints'] = checkpoints

    #The primary checkpoint is the source of the keys for the merged state_dict and is used as a fallback if another model lacks a tensor
    recipe['primary_checkpoint'] = checkpoints[0]"""


def start_merge(recipe):
    mTimer = timer.Timer()

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





