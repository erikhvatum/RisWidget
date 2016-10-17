
def drawpix(a, x, y):
    if 0 <= x < a.shape[0] and 0 <= y < a.shape[1]:
        a[x, y] = 255
    else:
        print('!', x, y)

def circle(a, r):
    r = round(r)
    x = r
    y = 0
    cd2 = 0

    if r==0:
        return

    drawpix(a, r, r)
    drawpix(a, 0, 0)
    drawpix(a, 0, r*2)

    while x > y:
        x -= 1
        y += 1
        cd2 -= x - y
#       print(cd2, x, y)
        if cd2 < 0:
            cd2 += x
            x += 1
        drawpix(a, y, r-x) #upper upper right
        drawpix(a, x, r-y) #upper right right
        drawpix(a, y, r+x) #lower lower right
        drawpix(a, x, r+y) #lower right right
