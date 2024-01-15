import re

BASE_SELECTORS = {
    "all":  "",
    "clip": "cond",
    "unet": "model\\.diffusion_model",
    "in":   "model\\.diffusion_model\\.input_blocks",
    "out":  "model\\.diffusion_model\\.output_blocks",
    "mid":  "model\\.diffusion_model\\.middle_block"
}


def target_to_regex(target_name: str) -> str:
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
        elif re.search(r'^\w*$',selector):
            regex += f".*{selector}"
    regex += ".*$"
    return regex