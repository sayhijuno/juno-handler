VALIDATIONS = {
    "messages": {
        "type": list,
        "required": True,
    },
    "temperature": {
        "type": float,
        "required": False,
        "default": None,
    },
    "max_tokens": {
        "type": int,
        "required": False,
        "default": None,
    },
    "top_p": {
        "type": float,
        "required": False,
        "default": None,
    },
    "tools": {
        "type": list,
        "required": False,
        "default": None,
    },
    "tool_choice": {
        "type": object,
        "required": False,
        "default": None,
    },
}
