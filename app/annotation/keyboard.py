from typing import Callable, Dict


class KeyboardController:
    def __init__(self, app: object) -> None:
        self.app = app

    def bind(self, root: object) -> None:
        root.bind("<Escape>", lambda event: root.destroy())
        root.bind("<Left>", lambda event: self.app.previous_image())
        root.bind("<Right>", lambda event: self.app.next_image())
        root.bind("s", lambda event: self.app.skip_image())
        root.bind("<Return>", lambda event: self.app.save_and_next())
