import os

BASE_PATH = os.path.dirname(__file__)


def get_script(name: str) -> str:
    # get the path of the file
    if not name.endswith('.sql'):
        name = f"{name}.sql"
    path = os.path.join(BASE_PATH, 'sql', name.lower())

    # read the file
    with open(path, 'r') as f:
        return f.read()
    

def INIT() -> str:
    return get_script('init')
