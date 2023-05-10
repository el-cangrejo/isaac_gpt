from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from base import *

import os
import math
import numpy as np
import torch

import openai

def subscribe_events():
    robot.gym.subscribe_viewer_keyboard_event(robot.viewer, gymapi.KEY_R, "r")
    robot.gym.subscribe_viewer_keyboard_event(robot.viewer, gymapi.KEY_A, "a")

def load_prompt(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return text 

def get_llm_response(prompt):
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt,
      temperature=0,
      max_tokens=250,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=["###"]
    )

    return response["choices"][0]["text"]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# parse arguments
args = gymutil.parse_arguments(description="")

robot = BaseSim(args)

subscribe_events()

items = ["coke_can", "pear", "banana", "meat_can", "orange"]
places = ["coke_can", "pear", "banana", "meat_can", "orange", "bowl", "drawer",
          "table_1", "table_2"]

openai.api_key = os.getenv("OPENAI_API_KEY")

main_prompt = load_prompt('base_prompt.txt')

prompt_end = \
'''
## INSTRUCTION
items = ['banana', 'pear', 'coke_can', 'meat_can', 'orange']
places = ["coke_can", "pear", "banana", "meat_can", "orange", "bowl", "drawer", "table_1", "table_2"]
'''
llm_response = None

# Simulate
while not robot.gym.query_viewer_has_closed(robot.viewer):
    robot.step()
    for evt in robot.gym.query_viewer_action_events(robot.viewer):
        if evt.action == "r" and evt.value > 0:
            robot.reset()

        if evt.action == "a" and evt.value > 0:
            instruction = input(bcolors.BOLD + bcolors.HEADER + '\nGive me an instruction: ' + bcolors.OKCYAN)

            prompt_tmp = main_prompt.replace('INSTRUCTION', instruction)

            print ("\n")
            print (bcolors.BOLD + bcolors.FAIL + bcolors.UNDERLINE + "\nLLMs prompt:" + bcolors.ENDC)
            print (bcolors.OKGREEN + prompt_tmp + bcolors.ENDC)

            llm_response = get_llm_response(prompt_tmp)

            print (bcolors.BOLD + bcolors.FAIL + bcolors.UNDERLINE + "\nLLMs response:" + bcolors.ENDC)
            print (bcolors.WARNING + llm_response + bcolors.ENDC)

            print (bcolors.BOLD + bcolors.FAIL + bcolors.UNDERLINE + "\nExecuting LLMs response.." + bcolors.ENDC)
            try:
                exec(llm_response)
                print (bcolors.BOLD + bcolors.FAIL + bcolors.UNDERLINE + "\nFinished execution .." + bcolors.ENDC)
                ### Add output to prompt
                # main_prompt = prompt_tmp + llm_response + prompt_end
            except:
                print (bcolors.BOLD + bcolors.FAIL + bcolors.UNDERLINE + "\nError in code." + bcolors.ENDC)


print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
