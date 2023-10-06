from camera import Camera

class Chicken:
    def __init__(self) -> None:
        self.banner()
        self.camera = Camera()
        self.camera.start_capture()

    def banner(self) -> None:
        print('''
           ________    _      __              ___    ____
          / ____/ /_  (_)____/ /_____  ____  /   |  /  _/
         / /   / __ \/ / ___/ //_/ _ \/ __ \/ /| |  / /  
        / /___/ / / / / /__/ ,< /  __/ / / / ___ |_/ /   
        \____/_/ /_/_/\___/_/|_|\___/_/ /_/_/  |_/___/                                        
        ''')


if __name__ == '__main__':
    chicken = Chicken()