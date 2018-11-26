import sys,tty,termios

class _Getch:
    def __call__(self):
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(3)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

def get():
        inkey = _Getch()
        while(1):
                k=inkey()
                if k!='':break
        if k=='\x1b[A':
            return 3 # UP
        elif k=='\x1b[B':
            return 1 # DOWN
        elif k=='\x1b[C':
            return 2 # RIGHT
        elif k=='\x1b[D':
            return 0 # LEFT
        else:
            return get()
