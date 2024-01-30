
"""
#Install 
pip install --upgrade pip
pip install --disable-pip-version-check \
    torch==1.13.1 \
    torchdata==0.5.1 --quiet

pip install \
    transformers==4.27.2 \
    datasets==2.11.0  --quiet

pip install langchain
"""

# ### Imports 
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate

#Load Dataset, Model, Tokenizer
dataset_name = "knkarthick/dialogsum"
model_name='google/flan-t5-base'

dataset = load_dataset(dataset_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


def print_data(input_text,
               ground_truth,
               model_output=None,
               model_output_type=None,
               index_number=0):
    dash_line = "_".join(['']*100)
    print(dash_line)
    print('Example ', index_number + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{input_text}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{ground_truth}')
    if(model_output):
        print(dash_line)
        print(model_output_type + f'\n{model_output}\n')

def run_example(model,tokenizer,input_text):
    tokenized_inputs = tokenizer(input_text,return_tensors="pt")
    model_output =  model.generate(tokenized_inputs["input_ids"],max_new_tokens=50)
    predicted_output = tokenizer.decode(model_output[0],skip_special_tokens=True)
    return predicted_output 

def print_dataset(dataset,indices):
    dash_line = "_".join(['']*100)
    for i,index in enumerate(indices):
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        print_data(dialogue,summary,index_number=i)

class PromptsManager(object):
    def __init__(self,prompts_list=None):
        self.prompt_dict = {}
        self.add_all(prompts_list)

    def get(self,prompt_type):
        if(not self.prompt_dict):
            return None
        if(prompt_type in self.prompt_dict):
            return self.prompt_dict[prompt_type]
        raise Exception

    def get_text(self,prompt_type):
        if(not self.prompt_dict):
            return None
        if(prompt_type in self.prompt_dict):
            return self.prompt_dict[prompt_type]
        raise Exception
    
    def add(self,prompt_type,input_variables,prompt_text):
        self.prompt_dict[prompt_type] = PromptTemplate(
                        input_variables=input_variables,
                        template=prompt_text
                    )
    
    def add_all(self,prompts_list):
        for template_name,template_description,template_text,input_variables in prompts_list:
                self.prompt_dict[template_name] = Prompt(template_name= template_name,
                                                      input_variables=input_variables,
                                                      template_text=template_text,
                                                      template_description=template_description)

class Prompt(object):
    def __init__(self,template_text,template_name,template_description,input_variables):
        self.template_text = template_text
        self.template_name = template_name
        self.template_description = template_description
        self.input_variables = input_variables
        self.template = PromptTemplate(
                        input_variables=self.input_variables,
                        template=self.template_text
                    )

prompt_0 = """{dialogue}"""
prompt_1 = """
            Summarize the following conversation.

            {dialogue}

            Summary:
            """
prompt_2 = """
            Dialogue:

            {dialogue}

            What was going on?
            """

prompt_3 = """
            Dialogue:

            {dialogue}

            What was going on?
            {summary}

            

           """

prompts_list = (("no_prompt","MODEL GENERATION - WITHOUT PROMPT ENGINEERING:",prompt_0,["dialogue"]),
                ("zero_shot_1","MODEL GENERATION - Zero Shot:",prompt_1,["dialogue"]),
                ("zero_shot_2","MODEL GENERATION - Zero Shot:",prompt_2,["dialogue"]),
                ("few_shot_1","MODEL GENERATION - Few Shot:",prompt_3,["dialogue","summary"]))
prompts_manager = PromptsManager(prompts_list)


def make_prompt(example_indices_full, example_index_to_summarize):
    prompt_name = "few_shot_1"
    example_prompt = prompts_manager.get(prompt_name).template
    examples  = []
    for index in example_indices_full:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        examples.append({"dialogue":dialogue,"summary":summary})

    dialogue = dataset['test'][example_index_to_summarize]['dialogue']
    prefix = ""
    suffix = """
                Dialogue:

                {dialogue}

                What was going on?
            """
    few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["dialogue"],
    example_separator="\n\n"
     )     
    dialogue = dataset['test'][example_index_to_summarize]['dialogue']
    prompt = few_shot_prompt_template.format(dialogue=dialogue)
    return prompt


if __name__ == "__main__":
    #Explore Dataset
    example_indices = [40, 200]
    print_dataset(dataset,example_indices)

    #Run the model using zero shot
    print("******** Zero Shot Inference ******")
    prompt_type = "no_prompt"
    for i, index in enumerate(example_indices):
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        prompt = prompts_manager.get(prompt_type).template.format(dialogue=dialogue)
        model_summary = run_example(model,tokenizer,prompt)
        print_data(prompt,summary,model_summary,index_number=i,model_output_type=prompts_manager.get(prompt_type).template_description)
    
    print("******** Few Shot Inference ******")
    # Run the model using one shot & few shot template
    prompt_type = "few_shot_1"
    example_indices_full = [40, 80, 120]
    example_index_to_summarize = 200
    prompt = make_prompt(example_indices_full, example_index_to_summarize)
    model_summary = run_example(model,tokenizer,prompt)
    print_data(prompt,"",model_summary,index_number=i,model_output_type=prompts_manager.get(prompt_type).template_description)




