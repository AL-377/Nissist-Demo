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
import ast
import time
from pathlib import Path
import random
import glob


def reformulate2json(raw_tsg):
    system_prompt = """
    You are a helpful troubleshooting guide assistant that helps the user to formulate the manual unstructured troubleshooting guide <TSG> into structured one. The <TSG> is in markdown format, with the first level header describing the incident or problem, and the following each second level header providing information related to the incident or problem.
    
    Each second level subsection can be categoried into the following types: Terminology, Background, FAQ, STEPS and Appendix. Your reformulation should be strictly comply the following definition:
    - Terminology: firstly, it should be the relationship or connection between terminology about the incident, if not, is can be the explanation or concept of the incident. Sometimes it should extract and summarize by yourself.
    - Background: the information about the importance or context of the incident.
    - FAQ: frequently asked questions which help to understand the incident.
    - STEPS: the processes to resolve the incident, and you should make sure its completeness. Usually, steps have causally inner connection, the former step will trigger the next step.
    - Appendix: the supplement of the incident which is not important or labeled by TSG, usually providing additional 。resources, data, links and so on.

    1. You need to identify each second level subsection, includng third level subsection if it needed, analyze its content or purpose, and categorize it accordingly. For those belonging to STEPS, you should capture the inner connections, such as Causality or Temporal relations, and present them in the correct order.

    2. Your returned formulated TSG should be in JSON format. Make sure that the keys originate from these categories: Terminology, Background, FAQ, STEPS ad Appendix. Each value should be a list of dictionaries. The keys for Terminology, Background, FAQ and Appendix are "question" and "answer", and the keys for STEPS are "intent", "action", and "linker". All values within the lists need to align with the original context, with truthful meaning instead of simple words like "Introduction", and including the **code block** and '\\n'.
    
    3. Importantly, the "linker" is used to imply dual role of providing the action's result and connecting to the next step using the "if-then" sentence format. 
        You should formulate each steps's linker to be "If any results are obtained by executing the corresponding action in the previous step, then **the true intent of the following step** provided here". Implicit linkers like "proceed to the next step." or "then the intent of the following step should be taken into consideration." will cause HUGE accidents, since user DO NOT know what is the actual next step, so please includes next step's actual intent explicitly.

    4. For each "if" condition at every step in the STEPS, it is necessary to add a special token behind the "then" condition within the "linker". The options for these tokens are "[CONTINUE]", "[CROSS]", and "[MITIGATE]". 
        - The token "[CONTINUE]" indicates that the actions corresponding to this "if" condition are part of the continuum within the same TSG's STEPS. 
        - The token "[CROSS]" signifies that the subsequent actions require a transition to a different set of steps that are external to the current TSG's STEPS. 
        - The token "[MITIGATE]" implies that the actions following the "if" condition convey that the incident is mitigated, or necessitate communication with on-call engineers or teams. 
        
        The use of this special token is instrumental in verifying the completeness and structural integrity of the STEPS section.

    **An Example Here**

    Here is the TSG in markdown format you need to formulate:
    <TSG>: 
    """

    message_list = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "{markdown_content}   \n\n JSON does not allow true line breaks in anywhere. You must replace all line breaks with ascii characters '\\n', and your category type must strictly originate from [Terminology, Background, FAQ, STEPS ad Appendix], without any other catrgory.".format(markdown_content=raw_tsg)
        }
    ]
    output=get_oai_completion_gpt_unified(message_list,json_mode=True)
    print(f'refoumulated raw llm output: {output}')

    return output


def refine_autotsg(raw_tsg, auto_tsg):
    system_prompt = """
    As a valuable troubleshooting guide assistant, your role is to help users convert the unstructured manual troubleshooting guide <RAW_TSG> in MARKDOWN format into a structured one <RET_TSG>. However, some of the <RAW_TSG> information has been ignored and lost, and even classified into incorrect categories. So now please adhere to the following requirments while referencing <RAW_TSG> to polish the <RET_TSG> to ensure the original content is fully preserved and correctly classified.

    It should be claimed initially that the <RAW_TSG> can not be assessed by user, so avoid using "Please follow the steps provided in the <RAW_TSG> for each method" or "follow the appropriate steps provided in the original TSG." instead the actual contents in <RAW_TSG> as action or answer to refine the <RET_TSG>. Moreover, each elements in <RET_TSG> will be seperated later, so avoid using representation such as "proceed to the next step" and instead provide specific contents.

    First, confirm that all subsections, including second and third-level sections in <RAW_TSG>, are present in <RET_TSG>. Compare the number of subsections in <RAW_TSG> with the element count of all lists in <RET_TSG>. If they don't match, <RET_TSG> must ignore many subsectiosn, please examine each subsection one by one, and check if it exists in <RET_TSG>, if not, recategorize them strictly from [Terminology, Background, FAQ, STEPS ad Appendix], without any other catrgory. The keys for Terminology, Background, FAQ and Appendix are "question" and "answer", and the keys for STEPS are "intent", "action", and "linker".

    Second, verify that every element in the <RET_TSG> list contains complete information corresponding to its respective subsection in <RAW_TSG>. Ensure that all content under a subsection heading is fully preserved in <RET_TSG>, especially the **code block** and the sublist. If any content is missing, rewrite the element while maintaining its integrity.

    Third, reflect by yourself on your generated <RET_TSG>. Please think step by step to check the resonability of your reformulation. For STEPS, you should check if there is a close connection between two consecutive steps, and all steps can form a solution flow about mitigating the incident. At the same time, it is also necessary to identify nodes that have been misclassified, such as those that should belong to FAQ but were classified into STEPS. Usually, these nodes are not related to the flow in steps and should be classified into question-answer form instead of the intent-action-linker form.
    
    After you have verified STEPS' rationality and completeness, please ensure that the "linker" of each step has a continuation relationship. 
        - You should formulate each steps's linker to be "If any results are obtained by executing the corresponding action in the previous step, then **the true intent of the following step** provided here". Implicit linkers like "proceed to the next step." or "then the intent of the following step should be taken into consideration." will cause HUGE accidents, since user DO NOT know what is the actual next step, so please includes next step's actual intent explicitly.

        - The linker of the final step often involves the event being successfully resolved or being handed over to another team for further resolution. You should to POLISH the linker of EACH STEP to realize the format, since you have fully understood the connection between each step. And if you find some steps are isolated or not linked to previous or next step, they must be misclassified into STEPS, please reclassify these nodes into considerable categories.

        - For each "if" condition at every step in the STEPS, it is necessary to add a special token behind the "then" condition within the "linker". The options for these tokens are "[CONTINUE]", "[CROSS]", and "[MITIGATE]". 
            - The token "[CONTINUE]" indicates that the actions corresponding to this "if" condition are part of the continuum within the same TSG's STEPS. 
            - The token "[CROSS]" signifies that the subsequent actions require a transition to a different set of steps that are external to the current TSG's STEPS. 
            - The token "[MITIGATE]" implies that the actions following the "if" condition convey that the incident is mitigated, or necessitate communication with on-call engineers or teams. 
        
            The use of this special token is instrumental in verifying the completeness and structural integrity of the STEPS section.

    Lastly, you should double check if the <RET_TSG> only provides actions like "run the Kusto or other format code" or "Follow the provided steps and commands" but not actually includes **code block** , please go through the <RAW_TSG> and fill in the necessary **code block** or commands.

    Remember to keep the **code block** from <RAW_TSG>, do not lose any of them.

    Return with the updated TSG. I am sure that the original <RET_TSG> is not a complete, sufficient and well-formed version of <RAW_TSG>. More than one subsections are missed and many elements' content is incomplete. So if the revised version is identical to the original <RET_TSG>, you must have misunderstood the task. Please reevaluate the question and attempt it again with caution.  
    
    Below are the <RAW_TSG> in markdown format and the <RET_TSG> in the desired JSON format, please return with your final updated TSG:
    """

    message_list = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "\n <RAW_TSG>: {markdown_content} \n <RET_TSG>: {auto_tsg} \n\n  There are many missed subsections in it, and the content of the nodes are incomplete, especially missing the **codeblock**, please carefully categorize them strictly from [Terminology, Background, FAQ, STEPS ad Appendix], without any other catrgory, and make sure each element has complete content.".format(markdown_content=raw_tsg, auto_tsg=auto_tsg)
        }
    ]
    output=get_oai_completion_gpt_unified(message_list,json_mode=True)
    print(f'refine raw llm output: {output}')

    return output


def reformulate_tsg(raw_tsg, save_path):
    # gpt_reformulation = reformulate2md(raw_tsg, gptversion)
    print("******   reformulating   *******")
    gpt_reformulation = reformulate2json(raw_tsg)

    # ****  whether use refining  ****
    print("******   refining   *******")
    gpt_reformulation = refine_autotsg(raw_tsg, gpt_reformulation)
    refine_save_path = save_path
    with open(refine_save_path,'w', encoding='utf-8') as f:
        f.write(gpt_reformulation)

def json2md(tsg_file, save_path):
    tsg_title = save_path.split("\\")[-1].split(".json")[0] + ".md"
    try: 
        # Read the JSON file
        with open(tsg_file, 'r') as file:
            data = json.load(file)
        # Prepare the Markdown content
        markdown_content = """"""
        if "Terminology" in data.keys():
            markdown_content += "# Terminology\n\n"
            for term in data['Terminology']:
                question = term['question']
                answer = term['answer']
                markdown_content += f"## {question}\n\n{answer}\n\n"
        if "Background" in data.keys():
            markdown_content += "\n# Background\n\n"
            for term in data['Background']:
                question = term['question']
                answer = term['answer']
                markdown_content += f"## {question}\n\n{answer}\n\n"
        if "FAQ" in data.keys():
            markdown_content += "\n# FAQ\n\n"
            for term in data['FAQ']:
                question = term['question']
                answer = term['answer']
                markdown_content += f"## {question}\n\n{answer}\n\n"
        if "STEPS" in data.keys():
            markdown_content += f"\n# How to Investigate {tsg_title} Incident \n\n"
            for index, step in enumerate(data['STEPS']):
                intent = step['intent']
                action = step['action']
                linker = step['linker']
                markdown_content += f"## {index+1}.{intent}\n\n### Intent\n\n{intent}\n\n###  Action\n\n{action}\n\n### Output\n\n{linker}\n\n"

        if "Appendix" in data.keys():
            markdown_content += "\n# Appendix\n\n"
            for term in data['Appendix']:
                question = term['question']
                answer = term['answer']
                markdown_content += f"## {question}\n\n{answer}\n\n"
    except Exception as e:
        print(tsg_file, e)
        return
    
    # Write the Markdown content to a file
    # markdown_file = f'{tsg_title}.md'
    with open(save_path, 'w') as file:
        file.write(markdown_content)
    # get the tsg title this from the path like D:XX\\XX\\XX\\tsg_title.json

    print(f"Markdown content successfully written to {tsg_title}")


if __name__ == '__main__':       
    raw_tsgs_path = os.path.join(os.path.dirname(__file__), 'used')
    save_path=os.path.join(os.path.dirname(__file__), 'used_reformulated_tsgs')    
    tmp_path = os.path.join(os.path.dirname(__file__), 'tmp')

    raw_tsgs = glob.glob(os.path.join(raw_tsgs_path, "**/*.md"), recursive=True)
    print(raw_tsgs)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path, exist_ok=True)
    for raw_tsg in raw_tsgs:
        tmp_file_path=os.path.join(tmp_path, raw_tsg.split('/')[-1].split('.')[0] + '.json')
        save_file_path=os.path.join(save_path, raw_tsg.split('/')[-1].split('.')[0] +'.md')

        with open(raw_tsg, 'r', encoding='utf-8') as f:
            raw_tsg_md = f.read()
        reformulate_tsg(raw_tsg_md, tmp_file_path)
        json2md(tmp_file_path,save_file_path)