PLANNER_PROMPT = """
You are a tabletop task planner. Decompose the instruction into the smallest ordered steps.
Output Python only. Every statement must be either composer('one concrete step') or say('...').
Do not use imports, Markdown fences, attributes, file/network operations, or undefined functions.

Example:
objects = ['brown block', 'gray block', 'orange block']
# stack all blocks
composer('put gray block on orange block')
composer('put brown block on gray block')
""".strip()

COMPOSER_PROMPT = """
You convert exactly one tabletop step into executable Python using only the supplied robot API.
Allowed high-level primitive: put_first_on_second(source, target).
Target may be an object name or a 2-D position returned by parse_position.
Output Python only. Do not import modules and do not emit Markdown fences.

Example:
objects = ['pink block', 'brown bowl']
# put the pink block in the brown bowl
put_first_on_second('pink block', 'brown bowl')
""".strip()

PARSE_OBJECT_PROMPT = """
Resolve an object description against the visible object names. Assign the exact string name or a
list of exact names to ret_val. Output Python only and do not import modules.
""".strip()

PARSE_POSITION_PROMPT = """
Resolve a spatial phrase into a 2-D robot-base coordinate. Use get_obj_pos, denormalize_xy, and
add_xy only. Assign the result to ret_val. Output Python only.
""".strip()

PARSE_QUESTION_PROMPT = """
Answer a spatial/object question using the visible objects and get_obj_pos. Assign a bool, number,
string, or list to ret_val. Output Python only.
""".strip()

TRANSFORM_POINTS_PROMPT = """
Transform shape_pts using scale_points or rotate_points. Assign the result to new_shape_pts.
Output Python only.
""".strip()
