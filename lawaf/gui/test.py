import numpy as np
import matplotlib.pyplot as plt
from nicegui import app, ui

ui.label("Hello, world!")
ui.button("Click me", on_click=lambda: ui.label("Thanks!"))
with ui.pyplot():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)

ui.run()


# ui.label('Lawaf!')
# ui.run()
