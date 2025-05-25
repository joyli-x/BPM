PARSE_INSTRUCTION_PROMPT = """
Given an instruction for image editing, output the parts of the original image that should be modified and the modified parts in the edited image compared to the original image. If there are no parts that should be modified, output None. Don't output the reasons.
Here are some examples:
1. Instruction: "Replace the bike with bear." Output: Original image: bike. Edited image: bear.
2. Instruction: "Add a toy on the table." Output: Original image: None. Edited image: toy.
3. Instruction: "Add a car on the screen." Output: Original image: None. Edited image: car.
4. Instruction: "Remove the bike." Output: Original image: bike. Edited image: None.
5. Instruction: "Change the image to a cartoon style." Output: Original image: all. Edited image: all.
Here is the instruction: 
"""

EDIT_TYPE_PROMPT = """
Given an editing instruction, please determine its editing type. The available editing types are: add, remove, replace, change color, change texture, change size, other. Don't output the reasons. Here are some examples:
1. Instruction: put a beauty queen seated on top of bus. Output: add.
2. Instruction: let the tie be yellow. Output: change color.
3. Instruction: turn the wine into absinthe. Output: replace.
4. Instruction: make the flower bigger. Output: change size.
Here is the instruction to be determined: 
"""

BEING_ADDED_TO_OBJECT_PROMPT = """
Given an editing instruction of adding an object, please provide the object that is being added to. Here are some examples:
1. Instruction: put a car on the screen of the laptop. Output: screen of the laptop.
2. Instruction: let a pizza be inside the microwave. Output: microwave.
3. Instruction: What if the cat has glasses. Output: cat.
4. Instruction: Add a gas pump. Output: None.

Please output only the objects; do not output anything else and do not use bold.
Here is the instruction to be analyzed: """

ADDED_OBJECT_PROMPT = """
Given an editing instruction of adding an object, please provide the added object. Here are some examples:
1. Instruction: put a car on the screen of the laptop. Output: car.
2. Instruction: let a pizza be inside the microwave. Output: pizza.
3. Instruction: What if the cat has glasses. Output: glasses.
4. Instruction: Add a gas pump. Output: gas pump.

Please output only the objects; do not output anything else and do not use bold.
Here is the instruction to be analyzed: 
"""

SIZE_PROMPT = """
Given an instruction for image editing, output what size changes of the objects are involved, such as bigger, taller. If there are no size changes involved, output None. Don't output the reasons.
Here are some examples:
1. Instruction: "give the bear bigger claws." Output: bigger.
2. Instruction: "Make the vase thinner." Output: thinner.
3. Instruction: "Remove the bike." Output: None.
4. Instruction: "Let's add a cowboy hat to the giraffe." Output: None.
Here is the instruction: 
"""

POSITION_PROMPT = """
Given an instruction for image editing and edited area in the original and edited image, output the positional relationship between the edited area in the edited image and the edited area in the original image. The optional positional relationships are: left, right, above, below, inside, unchanged, None.
Here are some examples:
1. Instruction: "put a car on the screen of the laptop". Edited area in original image: screen of the laptop. Edited area in edited image: car. Output: inside.
2. Instruction: "Let a horse stand on the grass". Edited area in original image: grass. Edited area in edited image: horse. Output: None.
3. Instruction: "put a bird on top of the tower". Edited area in original image: tower. Edited area in edited image: bird. Output: above.
4. Instruction: "replace the cat with a dog". Edited area in original image: cat. Edited area in edited image: dog. Output: unchanged.
Here is the case to be analyzed: 
"""