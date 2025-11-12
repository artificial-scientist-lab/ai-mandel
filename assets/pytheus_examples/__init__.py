#%%
import pathlib

from agora.pytheus_tool.examples import PyTheusTask

__all__ = [
    "configs"
]

dir = pathlib.Path(__file__).parent
configs = {}

for file_path in dir.rglob('*config*'):
    try:
        if file_path.is_file() and file_path.suffix == '.json':
            config = PyTheusTask.parse_file(file_path)
            dir_rel = file_path.relative_to(dir)
            configs[dir_rel] = config

    except Exception as e:
        print(e)

