from enum import Enum

class Scheduler_Type(Enum):
    DDIM = 1
    EULER = 2
    LCM = 3
 
class Model_Type(Enum):
    SDXL = 1
    SDXL_Turbo = 2
    LCM_SDXL = 3
    SD15 = 4
    SD21 = 5
    SD21_Turbo = 6
    SD14 = 7