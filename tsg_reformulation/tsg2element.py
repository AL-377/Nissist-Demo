import os
import json
import sys
import re
# Get the parent directory of the current script (my_folder)
current_directory = os.path.dirname(os.path.realpath(__file__))
# Add the parent directory to sys.path
sys.path.append(os.path.join(current_directory, ".."))
from llm_components import get_oai_completion_gpt_unified
from tqdm import tqdm


def tsg_to_element(tsg_md):
    system_prompt = '''
    You are a helpful assistant that extract the information from the troubleshooting guide in <TSG>. <TSG> is in markdown format and each first level header is a section such as Terminology, Background, FAQ, etc. The **steps** of troubleshooting the incident are in the section named related to the incident. You should extract the second level header in the section and write each element following the below format as json by STRICTLY following the requirement in each field, keep '#' in the json keys:
    {
        "#type#": "the type of the element, select from the following types: terminology, background, faq, steps.",
        "#title#": "the first level header of the **steps** section where describe the incident or problem. All type of the element should use the first level header of the step section as the title.",
        "#intent#": "for other type of sections, the #intent# can be the question or terminology that is asked or defined. For the **steps** type section, the #intent# is a summarization of both the third level intent context (if existed) and the header of the **steps** type section. DO NOT include the item number in the #intent#.",
        "#action#": "for the **steps** section, the action is the content which troubleshoots the incident or problem, including the **code block** and '\\n'. for other sections, the action can be the answer or definition of the intent.",
        "#output#": "the output of the action. if there is no output, make it empty. for example, the faq section may not have output. If there is output, you SHOULD extract the output following the below format: -If **condition**, then **should_do**. It can contain multiple if-then cases. If this step is the last step and you cannot extract the 'if-then' format, you can use the output as the **condition** and make the **should_do** as 'you should mitigate this incident.'"
    }
    
    You should extract elements following the above json format from each second level header and write them in json contains a list, following the below json format:
    {
        "extracted_elements":
        [
            element1,
            element2,
            ...
        ]
    }

    Here is the TSG in markdown format:
    <TSG>:
    '''
    message_list = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "{markdown_content}".format(markdown_content=tsg_md)
        }
    ]
    output=get_oai_completion_gpt_unified(message_list, json_mode=True)
    return output


def extract_code_template(code):
    system_prompt = '''
    You are a helpful assistant that extract the code template and the default parameters from the prvoided code instance in <CODE>. <CODE> is a code block contains several parameters. You should replace those parameters with placeholders and output the code template with placeholders and default parameters. For example, in the code instance below:
    ```kusto
    ***
    ```

    You response should be in the json format as below:
    {
        "#CODE_TEMPLATE#": where you replace the parameters in <CODE> with placeholders,
        "#DEFAULT_PARAMETERS#": where you keep the parameters in <CODE> as default values. 
    }    
    where the #CODE_TEMPLATE# should be:
    ```kusto
    **
    ```
    and the #DEFAULT_PARAMETERS# should be in json format:
    {
        "<**>": "**"
    }

    Your response should contain the code template and the default parameters extracted from <CODE>.

    If you think the <CODE> is not a vailid code block or you cannot extract the code template, please say "Sorry, I cannot give a confident answer".
    
    Your output should ONLY be the json format in <RESPONSE>.    
    '''

    message_list = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "Here is the code instance:\n<CODE>:\n{code_instance}\n<RESPONSE>:\n".format(code_instance=code)
        }
    ]
    output=get_oai_completion_gpt_unified(message_list, json_mode=True)
    
    print(output)
    if is_no_answer(output):
        print('Not valid code block')
        return code, {}
    
    def filter_list(output):
        start_index = output.find('{')
        if start_index != -1:
            list_part = output[start_index:]
            return list_part
        else:
            return code

    output =filter_list(output)
    output = json.loads(output)
    code = output['#CODE_TEMPLATE#']
    default_parameters = output['#DEFAULT_PARAMETERS#']
    # default_parameters = json.loads(default_parameters)
    return code, default_parameters


def extract_markdown_code_blocks(md_string):
    code_blocks = re.findall(r'```.*?```', md_string, re.DOTALL)
    return code_blocks

def replace_code_blocks(md_string, modified_code_blocks):
    def replace_code_block(match):
        return modified_code_blocks.pop(0)

    modified_md_string = re.sub(r'```.*?```', replace_code_block, md_string, flags=re.DOTALL)
    return modified_md_string


def is_no_answer(response):
    NO_ANSWER_PAT=[
        'Sorry, I do not understand your question',
        'Sorry, I cannot give a confident answer',
        'I cannot confidently confirm',
        'could you please',
    ]
    for pat in NO_ANSWER_PAT:
        if pat.lower() in response.lower():
            return 1
    return 0

def modify_code_template(code_string):
    code_blocks_list = extract_markdown_code_blocks(code_string)
    if len(code_blocks_list)==0:
        return code_string, {}
    # print('code_blocks_list', code_blocks_list)
    modified_code_blocks=[]
    default_parameter_list=[]
    for code_block in code_blocks_list:
        code_block, default_parameters = extract_code_template(code_block)
        modified_code_blocks.append(code_block)
        default_parameter_list.append(default_parameters)

    def unify_parameters(default_parameter_list):
        if len(default_parameter_list)==0:
            return []
        else:
            unified_parameter_list=default_parameter_list[0]
            for default_parameters in default_parameter_list[1:]:
                for key in default_parameters:
                    if key not in unified_parameter_list:
                        unified_parameter_list[key]=default_parameters[key]
            return unified_parameter_list

    modified_string = replace_code_blocks(code_string, modified_code_blocks)
    default_parameters = unify_parameters(default_parameter_list)
    return modified_string, default_parameters

def save_elements_with_llm(tsg_folder, elements_folder):
    for path_file in tqdm(os.listdir(tsg_folder)):
        path_file=os.path.join(tsg_folder, path_file)
        save_file_path=os.path.join(elements_folder, path_file.split('\\')[-1].split('.')[0]+'.json')
        if os.path.exists(save_file_path):
            continue
        with open(path_file, 'r', encoding='utf-8') as f:
            path_md = f.read()
        # print(path_md)
        output=tsg_to_element(path_md)
        print(output)

        if output is None:
            print('Failed generation {file_path}'.format(file_path=path_file))
            continue
        # output=filter_list(output)
        try:
            output_json=json.loads(output['extracted_elements'])
            print(output_json)
            with open(save_file_path, 'w', encoding='utf-8') as f:
                json.dump(output_json, f, indent=4)
        except json.JSONDecodeError as e:
            with open('test.txt','w', encoding='utf-8') as f:
                f.write(output)
            print(f"Error parsing JSON: {e}")
        except Exception as e:
            print(f"Error occurred: {e}")

def save_elements(tsg_folder, elements_folder):
    # assume the TSGs are well formulated
    for path_file in tqdm(os.listdir(tsg_folder)):
        path_file=os.path.join(tsg_folder, path_file)
        save_file_path=os.path.join(elements_folder, path_file.split('\\')[-1].split('.')[0]+'.json')
        # if os.path.exists(save_file_path):
        #     continue
        with open(path_file, 'r', encoding='utf-8') as f:
            path_md = f.read()

        output_json = []
        # loop through the first header starts with # of the markdown file
        sections = re.split(r'(?=\n# )', path_md)
        def check_string_start(s, header_pattern="##"):
            # Construct the regex pattern with the provided header pattern
            # pattern = fr'^(?:{re.escape(header_pattern)}|\n{re.escape(header_pattern)}|\n\s*{re.escape(header_pattern)})'
            pattern = fr'^(?:{re.escape(header_pattern)}|\n\s*{re.escape(header_pattern)})'
            return bool(re.match(pattern, s))
        for section in sections:
            if section.strip():
                section_lines = section.strip().split('\n')
                section_type = section_lines[0].replace('#', '').strip().lower()
                sub_section = re.split(r'(?=\n## )', section)
                if section_type in ['terminology','background','faq','appendix']:
                    for sub in sub_section:
                        if sub.strip() and check_string_start(sub):
                            sub_lines = sub.strip().split('\n')
                            intent = sub_lines[0].replace('##', '').strip()
                            action = ' '.join(sub_lines[1:]).strip()
                            output_json.append({
                                "#type#": section_type,
                                "#intent#": intent,
                                "#action#": action,
                                "#output#": ""
                            })
                else:
                    sub_section = re.split(r'(?=\n## )', section)
                    for sub in sub_section:
                        subsub_section = re.split(r'(?=\n### )', sub)
                        step={}
                        step["#type#"]="steps"
                        for subsub in subsub_section:
                            if subsub.strip() and check_string_start(subsub,"###"):
                                subsub_lines = subsub.strip().split('\n')
                                subsub_title = subsub_lines[0].replace('###', '').strip().lower()
                                if subsub_title == 'intent':
                                    step["#intent#"] = ' '.join(subsub_lines[1:]).strip()
                                elif subsub_title == 'action':
                                    step["#action#"] = ' '.join(subsub_lines[1:]).strip()
                                elif subsub_title == 'output':
                                    step["#output#"] = ' '.join(subsub_lines[1:]).strip()
                        if len(step)>1:
                            output_json.append(step)
        # print(output_json)
        with open(save_file_path, 'w', encoding='utf-8') as f:
            # dump the list of json into json file
            json.dump(output_json, f, indent=4)
            


def amplify_intent(l_json):
    for idx, element in enumerate(l_json):
        if element['#type#']=='steps':
            element['#intent#']=element['#title#']+'\n'+element['#intent#']
            l_json[idx]=element
            return l_json

def convert2template(elements_folder, save_folder, monitor_map=None):
    os.makedirs(save_folder, exist_ok=True)
    for path_file in tqdm(os.listdir(elements_folder)):
        save_path=os.path.join(save_folder, path_file)
        if os.path.exists(save_path):
            continue
        path_file=os.path.join(elements_folder, path_file)
        with open(path_file, 'r', encoding='utf-8') as f:
            path_json = json.load(f)
        # get the tsg name from the path_file
        tsg_title = path_file.split('\\')[-1].split('.')[0]        
        monitor_id = monitor_map[tsg_title] if monitor_map is not None else None
        new_json=[]
        is_first_node = True
        for element in path_json:
            new_element=element
            action = element['#action#']
            try:
                code, default_parameters=modify_code_template(action)
            except:
                code = element['#action#']
                default_parameters = ""
            new_element["#default_parameters#"]=default_parameters
            new_element["#action#"]=code
            new_element["#title#"]=tsg_title
            if new_element["#type#"] == 'steps' and is_first_node:
                new_element['#isfirst#']="Yes"
                is_first_node = False
            else:
                new_element['#isfirst#']="No"
            if monitor_id is not None:
                new_element['#monitor#']=monitor_id
            new_json.append(new_element)
        new_json=amplify_intent(new_json)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(new_json, f, indent=4)       

def node_db_generation(tsg_folder_path, tmp_folder_path, node_folder_path, monitor_map=None):
    save_elements(tsg_folder_path, tmp_folder_path)
    convert2template(tmp_folder_path, node_folder_path, monitor_map)


if __name__ == '__main__':   

    tsg_folder_path=os.path.join(os.path.dirname(__file__), 'used_reformulated_tsgs') # inputs
    tmp_folder_path=os.path.join(os.path.dirname(__file__), 'tsg_kb_tmp')
    output_folder_path=os.path.join(os.path.dirname(__file__), 'tsg_kb')  # outputs
    os.makedirs(tmp_folder_path, exist_ok=True)    
    os.makedirs(output_folder_path, exist_ok=True)
    # load monitor map
    monitor_map_path=os.path.join(os.path.dirname(__file__), 'monitor_map.json')
    with open(monitor_map_path, 'r', encoding='utf-8') as f:
        monitor_map = json.load(f)
    node_db_generation(tsg_folder_path, tmp_folder_path, output_folder_path, monitor_map)