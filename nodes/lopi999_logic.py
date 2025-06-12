import math

class node_RoundFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_float": ("FLOAT", {"forceInput": True}),
                "decimal_places": ("INT", {"min": 0, "default": 1})
                }
            }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("rounded_float",)
    FUNCTION = "round_value"
    CATEGORY = "lopi999/logic"

    def round_value(self, input_float: float, decimal_places: int) -> float:
        factor = 10.0 ** decimal_places
        shifted = abs(input_float) * factor
        rounded_shifted = math.floor(shifted + 0.5)
        result = math.copysign(rounded_shifted / factor, input_float)

        return result,

class node_InvertSign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["always_invert", "invert_if_positive", "invert_if_negative"],)
                },
            "optional": {
                "in_float": ("FLOAT", {"forceInput": True}),
                "in_int": ("INT", {"forceInput": True}),
                }
            }

    RETURN_TYPES = ("FLOAT","INT",)
    RETURN_NAMES = ("inv_float", "inv_int",)
    FUNCTION = "invert_value"
    CATEGORY = "lopi999/logic"

    def invert_value(self, mode, in_float=0.0, in_int=0):
        def invert(val, cond):
            return -val if cond(val) else val

        if mode == "always_invert":
            cond = lambda x: True
        elif mode == "invert_if_positive":
            cond = lambda x: x > 0
        else:
            cond = lambda x: x < 0

        return (invert(in_float, cond), invert(in_int, cond),)

