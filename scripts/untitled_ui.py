import gradio as gr
import os,re,functools
import torch,safetensors,safetensors.torch
from modules import sd_models,script_callbacks,scripts,shared,ui_components,paths,sd_samplers,ui,call_queue
from modules.ui_common import create_output_panel,plaintext_to_html
from modules.ui import create_sampler_and_steps_selection
from scripts.untitled import merger,misc_util
import scripts.untitled.common as cmn
from copy import deepcopy

checkpoints_no_pickles = lambda: [checkpoint for checkpoint in sd_models.checkpoint_tiles() if checkpoint.split(' ')[0].endswith('.safetensors')]

extension_path = scripts.basedir()

ext2abs = lambda *x: os.path.join(extension_path,*x)

sd_checkpoints_path = os.path.join(paths.models_path,'Stable-diffusion')
        
with open(ext2abs('scripts','examplemerge.yaml'), 'r') as file:
    EXAMPLE = file.read()

model_a_keys = []

recently_saved = []
recent_save_prefix = '[Recent save] '

def on_ui_tabs():
    with gr.Blocks() as blocksui:
        dummy_component = gr.Textbox(visible=False,interactive=True)
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
                        model_a.change(fn=checkpoint_changed,inputs=model_a,outputs=model_a_info).then(fn=update_model_a_keys, inputs=model_a)

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
                    mode_selector = gr.Radio(label='Merge mode:',choices=list(merger.calcmode_selection.keys()),value=list(merger.calcmode_selection.keys())[0],scale=3)
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
                            save_loaded.click(fn=misc_util.save_loaded_model, inputs=[save_name,save_settings],outputs=status).then(fn=refresh_models,outputs=[model_a,model_b,model_c])

                    with gr.Column():
                        #### MERGE BUTTONS
                        merge_button = gr.Button(value='Merge',variant='primary')
                        merge_and_gen_button = gr.Button(value='Merge & Gen',variant='primary')
                        gen_button = gr.Button(value='Gen',variant='primary')
                        with gr.Row():
                            empty_cache_button = gr.Button(value='Empty Cache')
                            #stop_button = gr.Button(value='Stop merge')
                with gr.Accordion(label='Include/Exclude/Discard',open=False):
                    with gr.Row():
                        with gr.Column():
                            clude = gr.Textbox(max_lines=4,label='Include/Exclude:',info='Entered targets will remain as model_a when set to \'Exclude\', and will be the only ones to be merged if set to \'Include\'. Separate with withspace.',value='first_stage_model',lines=4,scale=4)
                            clude_mode = gr.Radio(label="",info="",choices=["Exclude",("Include exclusively",'include')],value='Exclude',min_width=300,scale=1)
                        discard = gr.Textbox(max_lines=5,label='Discard:',info="Targets will be removed from the model, separate with whitespace.",value='model_ema',lines=5,scale=1)
                    
                with gr.Accordion("Supermerger Adjust", open=False) as acc_ad:
                    with gr.Row(variant="compact"):
                        finetune = gr.Textbox(label="Adjust", show_label=False, info="Adjust IN,OUT,OUT2,Contrast,Brightness,COL1,COL2,COL3", visible=True, value="", lines=1)
                        finetune_write = gr.Button(value="↑", elem_classes=["tool"])
                        finetune_read = gr.Button(value="↓", elem_classes=["tool"])
                        finetune_reset = gr.Button(value="\U0001f5d1\ufe0f", elem_classes=["tool"])
                    with gr.Row(variant="compact"):
                        with gr.Column(scale=1, min_width=100):
                            detail1 = gr.Slider(label="IN", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                        with gr.Column(scale=1, min_width=100):
                            detail2 = gr.Slider(label="OUT", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                        with gr.Column(scale=1, min_width=100):
                            detail3 = gr.Slider(label="OUT2", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                    with gr.Row(variant="compact"):
                        with gr.Column(scale=1, min_width=100):
                            contrast = gr.Slider(label="Contrast", minimum=-10, maximum=10, step=0.01, value=0, info="Contrast/Detail")
                        with gr.Column(scale=1, min_width=100):
                            bri = gr.Slider(label="Brightness", minimum=-10, maximum=10, step=0.01, value=0, info="Dark(Minius)-Bright(Plus)")
                    with gr.Row(variant="compact"):
                        with gr.Column(scale=1, min_width=100):
                            col1 = gr.Slider(label="Cyan-Red", minimum=-10, maximum=10, step=0.01, value=0, info="Cyan(Minius)-Red(Plus)")
                        with gr.Column(scale=1, min_width=100):
                            col2 = gr.Slider(label="Magenta-Green", minimum=-10, maximum=10, step=0.01, value=0, info="Magenta(Minius)-Green(Plus)")
                        with gr.Column(scale=1, min_width=100):
                            col3 = gr.Slider(label="Yellow-Blue", minimum=-10, maximum=10, step=0.01, value=0, info="Yellow(Minius)-Blue(Plus)")
                    
                        finetune.change(fn=lambda x:gr.update(label = f"Supermerger Adjust : {x}"if x != "" and x !="0,0,0,0,0,0,0,0" else "Supermerger Adjust"),inputs=[finetune],outputs = [acc_ad])

                    def finetune_update(finetune, detail1, detail2, detail3, contrast, bri, col1, col2, col3):
                        arr = [detail1, detail2, detail3, contrast, bri, col1, col2, col3]
                        tmp = ",".join(map(lambda x: str(int(x)) if x == 0.0 else str(x), arr))
                        if finetune != tmp:
                            return gr.update(value=tmp)
                        return gr.update()

                    def finetune_reader(finetune):
                        try:
                            tmp = [float(t) for t in finetune.split(",") if t]
                            assert len(tmp) == 8, f"expected 8 values, received {len(tmp)}."
                        except ValueError as err: gr.Warning(str(err))
                        except AssertionError as err: gr.Warning(str(err))
                        else: return [gr.update(value=x) for x in tmp]
                        return [gr.update()]*8
        
                    # update finetune
                    finetunes = [detail1, detail2, detail3, contrast, bri, col1, col2, col3]
                    finetune_reset.click(fn=lambda: [gr.update(value="")]+[gr.update(value=0.0)]*8, inputs=[], outputs=[finetune, *finetunes])
                    finetune_read.click(fn=finetune_reader, inputs=[finetune], outputs=[*finetunes])
                    finetune_write.click(fn=finetune_update, inputs=[finetune, *finetunes], outputs=[finetune])
                    detail1.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                    detail2.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                    detail3.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                    contrast.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                    bri.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                    col1.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                    col2.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
                    col3.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)

                with gr.Row(variant='panel'):
                    device_selector = gr.Radio(label='Preferred device/dtype for merging:',info='',choices=['cuda/float16', 'cuda/float32', 'cpu/float32'],value = 'cuda/float16' )
                    worker_count = gr.Slider(step=2,minimum=2,value=cmn.threads,maximum=16,label='Worker thread count:',info=('Relevant for both cuda and CPU merging. Using too many threads can harm performance.'))
                    def worker_count_fn(x): cmn.threads = int(x)
                    worker_count.release(fn=worker_count_fn,inputs=worker_count)
                    device_selector.change(fn=change_preferred_device,inputs=device_selector)

            gen_elem_id = 'untitled_merger'
            with gr.Column():
                status.render()
                with gr.Accordion('Weight editor'):
                    weight_editor = gr.Code(value=EXAMPLE,lines=20,language='yaml',label='')
                with gr.Tab(label='Image gen'):
                    with gr.Column(variant='panel'):
                        output_panel = create_output_panel('untitled_merger', shared.opts.outdir_txt2img_samples)
                        with gr.Accordion('Generation info',open=False):
                            infotext = gr.HTML(elem_id=f'html_info_{gen_elem_id}', elem_classes="infotext")

                    with gr.Row(elem_id=f"{gen_elem_id}_prompt_container", elem_classes=["prompt-container-compact"],equal_height=True):
                            promptbox = gr.Textbox(label="Prompt", elem_id=f"{gen_elem_id}_prompt", show_label=False, lines=3, placeholder="Prompt", elem_classes=["prompt"])
                            negative_promptbox = gr.Textbox(label="Negative prompt", elem_id=f"{gen_elem_id}_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt", elem_classes=["prompt"])
                    steps, sampler_name = create_sampler_and_steps_selection(sd_samplers.visible_sampler_names(), gen_elem_id)


                    with ui_components.FormRow():
                        with gr.Column(elem_id=f"{gen_elem_id}_column_size", scale=4):
                            width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id=f"{gen_elem_id}_width")
                            height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id=f"{gen_elem_id}_height")

                        with gr.Column(elem_id=f"{gen_elem_id}_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                            res_switch_btn = gr.Button(value='⇅', elem_id=f"{gen_elem_id}_res_switch_btn", tooltip="Switch width/height", elem_classes=["tool"])
                            res_switch_btn.click(fn=swapvalues, inputs=[width,height], outputs=[width,height])

                        with gr.Column(elem_id=f"{gen_elem_id}_column_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id=f"{gen_elem_id}_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id=f"{gen_elem_id}_batch_size")

                    with gr.Row():
                        cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id=f"{gen_elem_id}_cfg_scale")
                        
                    with gr.Row():
                        seed = gr.Number(label='Seed', value=-1, elem_id=gen_elem_id+" seed", min_width=100, precision=0)

                        random_seed = ui_components.ToolButton(ui.random_symbol, elem_id=gen_elem_id+" random_seed", tooltip="Set seed to -1, which will cause a new random number to be used every time")
                        random_seed.click(fn=None, _js="function(){setRandomSeed('" + gen_elem_id+" seed" + "')}", show_progress=False, inputs=[], outputs=[])
                        #reuse_seed = ui_components.ToolButton(ui.reuse_symbol, elem_id=gen_elem_id+" reuse_seed", tooltip="Reuse seed from last generation, mostly useful if it was randomized")


                    with ui_components.InputAccordion(False, label="Hires. fix", elem_id=f"{gen_elem_id}_hr") as enable_hr:
                        with enable_hr.extra():
                            hr_final_resolution = ui_components.FormHTML(value="", elem_id=f"{gen_elem_id}_hr_finalres", label="Upscaled resolution", interactive=False, min_width=0)

                        with ui_components.FormRow(elem_id=f"{gen_elem_id}_hires_fix_row1", variant="compact"):
                            hr_upscaler = gr.Dropdown(label="Upscaler", elem_id=f"{gen_elem_id}_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                            hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id=f"{gen_elem_id}_hires_steps")
                            denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id=f"{gen_elem_id}_denoising_strength")

                        with ui_components.FormRow(elem_id=f"{gen_elem_id}_hires_fix_row2", variant="compact"):
                            hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id=f"{gen_elem_id}_hr_scale")
                            hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id=f"{gen_elem_id}_hr_resize_x")
                            hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id=f"{gen_elem_id}_hr_resize_y")

                        hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]

                        for component in hr_resolution_preview_inputs:
                            event = component.release if isinstance(component, gr.Slider) else component.change

                            event(
                                fn=ui.calc_resolution_hires,
                                inputs=hr_resolution_preview_inputs,
                                outputs=[hr_final_resolution],
                                show_progress=False,
                            )

                with gr.Tab(label='Model keys'):
                    target_tester = gr.Textbox(max_lines=1,label="",info="",interactive=True,placeholder='out.4.tran.norm.weight')
                    target_tester_display = gr.Textbox(max_lines=40,lines=40,label="Targeted keys:",info="",interactive=False)
                    target_tester.change(fn=test_regex,inputs=[target_tester],outputs=target_tester_display,show_progress='minimal')


            empty_cache_button.click(fn=merger.clear_cache,outputs=status)

            merge_args = [
                mode_selector,
                model_a,
                model_b,
                model_c,
                alpha,
                beta,
                gamma,
                delta,
                weight_editor,
                save_name,
                save_settings,
                discard,
                clude,
                clude_mode,
                smooth,
                finetune
                ]
            
            gen_args = [
                dummy_component, 
                promptbox,
                negative_promptbox,
                steps,
                sampler_name,
                width,
                height,
                batch_count,
                batch_size,
                cfg_scale,
                seed,
                enable_hr,
                hr_upscaler,
                hr_second_pass_steps,
                denoising_strength,
                hr_scale,
                hr_resize_x,
                hr_resize_y
            ]

            merge_button.click(fn=merger.start_merge,inputs=merge_args,outputs=status)

            def merge_interrupted(func):
                @functools.wraps(func)
                def inner(*args):
                    if not cmn.interrupted:
                        return func(*args)
                    else:
                        return gr.update(),gr.update(),gr.update()
                return inner

            merge_and_gen_button.click(fn=merger.start_merge,
                                       inputs=merge_args,
                                       outputs=status).then(
                                        fn=merge_interrupted(call_queue.wrap_gradio_gpu_call(misc_util.image_gen, extra_outputs=[None, '', ''])),
                                        _js='submit_imagegen',
                                        inputs=gen_args,
                                        outputs=[output_panel.gallery,infotext,output_panel.html_log]
                                        
            )
            gen_button.click(fn=call_queue.wrap_gradio_gpu_call(misc_util.image_gen, extra_outputs=[None, '', '']),
                            _js='submit_imagegen',
                            inputs=gen_args,
                            outputs=[output_panel.gallery,infotext,output_panel.html_log])

    return [(blocksui, "Untitled merger", "untitled_merger")]

script_callbacks.on_ui_tabs(on_ui_tabs)


def test_regex(input):
    regex = misc_util.target_to_regex(input)
    selected_keys = re.findall(regex,'\n'.join(model_a_keys),re.M)
    joined = '\n'.join(selected_keys)
    return  f'Matched keys: {len(selected_keys)}\n{joined}'


def update_model_a_keys(model_a):
    global model_a_keys
    path = sd_models.get_closet_checkpoint_match(model_a).filename
    with safetensors.torch.safe_open(path,framework='pt',device='cpu') as file:
        model_a_keys = file.keys()


def change_preferred_device(input):
    cmn.device,dtype = input.split('/')
                     
    if dtype == 'float16': cmn.precision=torch.float16
    elif dtype == 'float8': cmn.precision=torch.float8_e4m3fn
    else: cmn.precision = torch.float32


def checkpoint_changed(name):
    if name == "":
        return plaintext_to_html('None | None',classname='untitled_sd_version')
    sdversion, dtype = misc_util.id_checkpoint(name)
    return plaintext_to_html(f"{sdversion} | {str(dtype).split('.')[1]}",classname='untitled_sd_version')


def calcmode_changed(calcmode_name):
    calcmode = merger.calcmode_selection[calcmode_name]

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


def refresh_models():
    sd_models.list_models()
    checkpoints_list = recently_saved + checkpoints_no_pickles()
    return gr.update(choices=checkpoints_list),gr.update(choices=checkpoints_list),gr.update(choices=checkpoints_list)
