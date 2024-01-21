import re,safetensors.torch,safetensors,torch
from modules import sd_models

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


