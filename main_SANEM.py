from sanem_funcs import *
import warnings
warnings.filterwarnings("ignore")

mode=input('Full Execution or Custom Flow? [Full/Custom]')

if mode=='Custom':
    sanemMain.do_custom_flow()

elif mode=='Full':
    sanemMain.do_full_execution()

else:
    print('You must select one of these two options: [Full/Custom]')