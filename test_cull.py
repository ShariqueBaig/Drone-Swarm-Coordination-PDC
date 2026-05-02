from ursina import *

app = Ursina()

box = Entity(model='cube', scale=(-3, 3, 3), color=color.rgba(255, 0, 0, 100), unlit=True)
Entity(model='cube', scale=1, color=color.green)

# capture a screenshot after 1 sec and exit
def take_screenshot():
    base.screenshot('test_cull.png')
    application.quit()

invoke(take_screenshot, delay=1)

EditorCamera()
app.run()
