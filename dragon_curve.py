import turtle

dragon = 'FX'

def transform(symbol):
    if symbol == 'X':
        return 'X+YF+'
    elif symbol == 'Y':
        return '-FX-Y'
    else:
        return symbol

for i in range(15):
    dragon = ''.join(map(transform, list(dragon)))

dragon = dragon.replace('X', '').replace('Y', '')

turtle.speed('fastest')

wn = turtle.Screen()
wn.delay(1)
wn.bgcolor('black')

n_turns = 0

for col in ['lightblue', 'purple', 'blue', 'darkblue']:
    jerome = turtle.Turtle(visible=False)
    
    jerome.color(col)
    jerome.pendown()

    for turn in range(n_turns):
        jerome.right(90)
    wn.tracer(0, 0)

    for symbol in dragon:
        if symbol == 'F':  
            jerome.forward(1.5)
        elif symbol == '+':
            jerome.right(90)
        elif symbol == '-':
            jerome.left(90)
    n_turns += 1
        
wn.update()

wn.getcanvas().postscript(file="blue_dragon.eps")