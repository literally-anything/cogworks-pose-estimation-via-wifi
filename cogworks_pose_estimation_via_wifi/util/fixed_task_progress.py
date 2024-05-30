from rich import filesize
from rich.progress import TaskProgressColumn, Task
from rich.themes import DEFAULT
from rich.style import Style
from rich.text import Text


class FixedTaskProgressColumn(TaskProgressColumn):
    def __new__(cls, *args, **kwargs) -> 'FixedTaskProgressColumn':
        if 'progress.speed' not in DEFAULT.styles:
            DEFAULT.styles['progress.speed'] = Style.parse('bold red')
        return super().__new__(cls)

    @classmethod
    def render_speed(cls, speed: float | None, unit_name: str = 'it') -> Text:
        if speed is None:
            return Text('')
        if speed < 1:
            return Text(f' {1/speed:.2f}s/{unit_name}', style='progress.speed')
        else:
            return Text(f' {speed:.2f} {unit_name}/s', style='progress.speed')

    def render(self, task: "Task") -> Text:
        text_format = (
            self.text_format_no_percentage if task.total is None else self.text_format
        )
        _text = text_format.format(task=task)
        if self.markup:
            text = Text.from_markup(_text, style=self.style, justify=self.justify)
        else:
            text = Text(_text, style=self.style, justify=self.justify)
        if self.highlighter:
            self.highlighter.highlight(text)
        if task.total is not None and self.show_speed:
            text = text + self.render_speed(task.finished_speed or task.speed, task.fields.get('unit', 'it'))
        return text
